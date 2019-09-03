import argparse
import os
import time
from math import log10

import pandas as pd
import torch
import torch.backends.cudnn
import torch.distributed
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.util import monkey_patch_bn

torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.enabled = False

monkey_patch_bn()

from metrics.metrics import AverageMeter
from metrics.ssim import ssim
from models.SRGAN import Generator, Discriminator, GeneratorLoss
from train.train_srgan_slow import ValDatasetFromFolder, TrainDatasetFromFolder

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--experiment-name", type=str)
parser.add_argument("--train-mx-path", default="/home/maksim/data/voc_train.rec")
parser.add_argument("--train-mx-index-path", default="/home/maksim/data/voc_train.idx")
parser.add_argument("--val-mx-path", default="/home/maksim/data/voc_val.rec")
parser.add_argument("--val-mx-index-path", default="/home/maksim/data/voc_val.idx")
parser.add_argument("--checkpoint-dir", default="/tmp")
parser.add_argument("--upscale-factor", type=int, default=2)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--prof", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--crop-size", type=int, default=88)
parser.add_argument("--workers", type=int, default=1)

args = parser.parse_args()
local_rank = args.local_rank
train_mx_path = args.train_mx_path
train_mx_index_path = args.train_mx_index_path
val_mx_path = args.val_mx_path
val_mx_index_path = args.val_mx_index_path
experiment_name = args.experiment_name
checkpoint_dir = args.checkpoint_dir
upscale_factor = args.upscale_factor
epochs = args.epochs
batch_size = args.batch_size
crop_size = args.crop_size
prof = args.prof
workers = args.workers
lr = args.lr
static_loss_scale = 1.0
dynamic_loss_scale = False
resume = False
dali_cpu = False
print_freq = 10

assert os.path.exists(train_mx_path)
assert os.path.exists(train_mx_index_path)
assert os.path.exists(val_mx_path)
assert os.path.exists(val_mx_index_path)
assert experiment_name
assert os.path.exists(checkpoint_dir)

if local_rank == 0:
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

distributed = False
world_size = 1

if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
    distributed = world_size > 1

print(f"distributed: {distributed}")
netG = Generator(scale_factor=upscale_factor, in_channels=3)
netD = Discriminator(in_channels=3)
generator_loss = GeneratorLoss()
total_batch_size = world_size * batch_size

if distributed:
    gpu = local_rank % torch.cuda.device_count()
    print(gpu)
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    assert world_size == torch.distributed.get_world_size()
    netG = netG.cuda(gpu)
    netD = netD.cuda(gpu)
    generator_loss = generator_loss.cuda(gpu)
    netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)
    netD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD)
    netG = torch.nn.parallel.DistributedDataParallel(
        netG, device_ids=[gpu], broadcast_buffers=False
    )
    netD = torch.nn.parallel.DistributedDataParallel(
        netD, device_ids=[gpu], broadcast_buffers=False
    )
    lr /= world_size
else:
    gpu = 0
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    netG = netG.cuda()
    netD = netD.cuda()
    generator_loss = generator_loss.cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=lr)
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr)

train_set = TrainDatasetFromFolder(
    "/home/maksim/data/VOC2012/train",
    crop_size=crop_size,
    upscale_factor=upscale_factor,
)
if distributed:
    dist_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
train_loader = DataLoader(
    dataset=train_set,
    num_workers=4,
    batch_size=batch_size,
    sampler=dist_sampler if distributed else None,
    shuffle=not distributed,
    pin_memory=True,
)
val_set = ValDatasetFromFolder(
    "/home/maksim/data/VOC2012/val", upscale_factor=upscale_factor
)
val_loader = DataLoader(
    dataset=val_set, num_workers=4, batch_size=1, shuffle=False, pin_memory=True
)


def train(epoch):
    netG.train()
    netD.train()

    running_results = {
        "batch_sizes": 0,
        "d_loss": 0,
        "g_loss": 0,
        "step": "",
        "batch_time": 0,
    }

    for i, (lr_image, hr_image) in enumerate(train_loader):
        start = time.time()

        batch_size = lr_image.size(0)
        running_results["batch_sizes"] += batch_size

        if gpu is not None:
            lr_image = lr_image.cuda(gpu, non_blocking=True)
            hr_image = hr_image.cuda(gpu, non_blocking=True)

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        fake_img = netG(lr_image)

        netD.zero_grad()
        real_out = netD(hr_image).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ###########################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ##########################
        netG.zero_grad()
        g_loss = generator_loss(fake_out, fake_img, hr_image)
        g_loss.backward()
        optimizerG.step()

        # record stats
        fake_img = netG(lr_image)
        fake_out = netD(fake_img).mean()
        g_loss = generator_loss(fake_out, fake_img, hr_image)
        running_results["g_loss"] += g_loss.item() * batch_size
        d_loss = 1 - real_out + fake_out
        running_results["d_loss"] += d_loss.item() * batch_size

        running_results["batch_time"] = time.time() - start

        if local_rank == 0:
            print(running_results)

    return running_results


def validate():
    netG.eval()
    val_bar = tqdm(val_loader, "validate")
    valing_results = {"mse": 0, "ssims": 0, "psnr": 0, "ssim": 0, "batch_sizes": 0}
    for lr, val_hr_restore, hr in val_bar:
        batch_size = lr.size(0)
        valing_results["batch_sizes"] += batch_size

        if gpu is not None:
            lr = lr.cuda(gpu, non_blocking=True)
            hr = hr.cuda(gpu, non_blocking=True)
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results["mse"] += batch_mse * batch_size
        batch_ssim = ssim(sr, hr).item()
        valing_results["ssims"] += batch_ssim * batch_size
        valing_results["psnr"] = 10 * log10(
            1 / (valing_results["mse"] / valing_results["batch_sizes"])
        )
        valing_results["ssim"] = valing_results["ssims"] / valing_results["batch_sizes"]

    return valing_results


def main():
    epoch_time = AverageMeter("epoch")
    end = time.time()
    results = {
        "mse": [],
        "psnr": [],
        "ssim": [],
        "g_lr": [],
        "d_lr": [],
        "g_loss": [],
        "d_loss": [],
        "epoch_time": [],
    }
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        running_results = train(epoch)
        if local_rank == 0:
            val_results = validate()
            val_results["epoch"] = epoch
            print(val_results)
            epoch_time.update(time.time() - end)
            end = time.time()

            torch.save(
                netG.state_dict(),
                f"{checkpoint_dir}/netG_epoch_{upscale_factor}_{epoch}.pth",
            )
            torch.save(
                netD.state_dict(),
                f"{checkpoint_dir}/netD_epoch_{upscale_factor}_{epoch}.pth",
            )
            results["psnr"].append(val_results["psnr"])
            results["ssim"].append(val_results["ssim"])
            results["mse"].append(val_results["mse"])
            results["g_lr"].append(optimizerG.param_groups[0]["lr"])
            results["d_lr"].append(optimizerD.param_groups[0]["lr"])
            results["d_loss"].append(
                running_results["d_loss"] / running_results["batch_sizes"]
            )
            results["g_loss"].append(
                running_results["g_loss"] / running_results["batch_sizes"]
            )
            results["epoch_time"].append(epoch_time.val)
            if epoch != 0 and not prof:
                data_frame = pd.DataFrame(data=results)
                data_frame.to_csv(
                    os.path.join(checkpoint_dir, "metrics.csv"), index_label="Epoch"
                )


if __name__ == "__main__":
    main()
