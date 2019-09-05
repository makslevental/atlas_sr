import argparse
import os
import time
from math import log10

import pandas as pd
import torch
import torch.backends.cudnn
import torch.distributed
from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import nn

from data_utils.dali import StupidDALIIterator, SRGANMXNetPipeline
from metrics.metrics import AverageMeter
from metrics.ssim import ssim
from models.SRGAN import Generator, Discriminator, GeneratorLoss
from util.util import reduce_tensor, snapshot


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

snapshot(checkpoint_dir)

distributed = False
world_size = 1

if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
    distributed = world_size > 1
if distributed:
    gpu = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    assert world_size == torch.distributed.get_world_size()
    lr *= world_size

total_batch_size = world_size * batch_size
static_loss_scale = 1.0
dynamic_loss_scale = False
resume = False
dali_cpu = False
print_freq = 10

netG = Generator(scale_factor=upscale_factor, in_channels=3)
netD = Discriminator(in_channels=3)

# because vgg excepts 3 channels
generator_criterion = GeneratorLoss()
generator_criterion = generator_criterion.cuda()
if distributed:
    # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
    # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
    netG = netG.cuda(gpu)
    netD = netD.cuda(gpu)
    netG = DDP(netG, delay_allreduce=True)
    netD = DDP(netD, delay_allreduce=True)
else:
    netG = netG.cuda()
    netD = netD.cuda()
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

# because vgg excepts 3 channels
# generator_criterion = lambda fake_out, fake_img, hr_image: generator_criterion_3(
#     fake_out,
#     torch.cat([fake_img, fake_img, fake_img], dim=1),
#     torch.cat([hr_image, hr_image, hr_image], dim=1),
# )

optimizerG = torch.optim.Adam(netG.parameters(), lr=lr)
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr)

train_pipe = SRGANMXNetPipeline(
    batch_size=batch_size,
    num_gpus=world_size,
    num_threads=workers,
    device_id=local_rank,
    crop=crop_size,
    dali_cpu=False,
    mx_path=train_mx_path,
    mx_index_path=train_mx_index_path,
    upscale_factor=upscale_factor,
)
train_pipe.build()
train_loader = StupidDALIIterator(
    pipelines=[train_pipe],
    output_map=["lr_image", "hr_image"],
    size=int(train_pipe.epoch_size("Reader") / world_size),
    auto_reset=False,
)
val_pipe = SRGANMXNetPipeline(
    batch_size=batch_size,
    num_gpus=world_size,
    num_threads=workers,
    device_id=local_rank,
    crop=crop_size,
    dali_cpu=False,
    mx_path=val_mx_path,
    mx_index_path=val_mx_index_path,
    upscale_factor=upscale_factor,
    random_shuffle=False,
)
val_pipe.build()
val_loader = StupidDALIIterator(
    pipelines=[val_pipe],
    output_map=["lr_image", "hr_image"],
    size=int(val_pipe.epoch_size("Reader") / world_size),
)


def train(epoch):
    netG.train()
    netD.train()

    running_results = {
        "batch_sizes": 0,
        "d_loss": 0,
        "g_loss": 0,
        "step": "",
        "batch_time": 0
    }

    for i, (lr_image, hr_image) in enumerate(train_loader):
        start = time.time()
        batch_size = lr_image.shape[0]
        running_results["batch_sizes"] += batch_size
        running_results["step"] = f"{i + 1}/{train_loader.size // batch_size}"
        #adjust_learning_rate(epoch, i, train_loader.size)

        if prof and i > 10:
            break

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ##########################
        fake_img = netG(lr_image)

        netD.zero_grad()
        real_out = netD(hr_image).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out

        if distributed:
            d_loss = reduce_tensor(d_loss, world_size)
        running_results["d_loss"] += d_loss.item() * batch_size
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, hr_image)
        if distributed:
            g_loss = reduce_tensor(g_loss, world_size)
        running_results["g_loss"] += g_loss.item() * batch_size
        g_loss.backward()
        optimizerG.step()

        torch.cuda.synchronize(device=torch.cuda.current_device())
        running_results["batch_time"] = time.time() - start

        if local_rank == 0:
            print(running_results)

    return running_results


def validate():
    netG.eval()
    valing_results = {"mse": 0, "ssims": 0, "psnr": 0, "ssim": 0, "batch_sizes": 0}
    for i, (lr_image, hr_image) in enumerate(val_loader):
        if prof and i > 10:
            break

        batch_size = lr_image.shape[0]
        valing_results["batch_sizes"] += batch_size
        with torch.no_grad():
            sr_image = netG(lr_image)

        batch_mse = ((sr_image - hr_image) ** 2).mean()
        batch_ssim = ssim(sr_image, hr_image)

        if distributed:
            batch_mse = reduce_tensor(batch_mse, world_size)
            batch_ssim = reduce_tensor(batch_ssim, world_size)

        valing_results["mse"] += batch_mse.item() * batch_size
        valing_results["ssims"] += batch_ssim.item() * batch_size

    valing_results["psnr"] = 10 * log10(
        1 / (valing_results["mse"] / valing_results["batch_sizes"])
    )
    valing_results["ssim"] = valing_results["ssims"] / valing_results["batch_sizes"]
    return valing_results


def adjust_learning_rate(epoch, step, len_epoch):
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)

    if args.local_rank == 0 and step % print_freq == 0 and step > 1:
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizerG.param_groups:
        param_group["lr"] = lr
    for param_group in optimizerD.param_groups:
        param_group["lr"] = lr


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
        running_results = train(epoch)
        val_results = validate()
        if local_rank == 0:
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

        val_loader.reset()
        train_loader.reset()


if __name__ == "__main__":
    main()
