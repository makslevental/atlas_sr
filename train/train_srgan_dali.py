import argparse
import os
import time
from math import log10

import apex
import pandas as pd
import torch
import torch.backends.cudnn
import torch.distributed
from apex.parallel import DistributedDataParallel, SyncBatchNorm as ApexSyncBatchNorm
from nvidia.dali import types
from torch import nn
from torch.nn import SyncBatchNorm as TorchSyncBatchNorm

from data_utils.dali import StupidDALIIterator, SRGANMXNetPipeline
from metrics.metrics import AverageMeter
from metrics.ssim import ssim
from models.SRGAN import Generator, Discriminator, GeneratorLoss
from train.tricks.lr_finder import LRFinder
from util.util import monkey_patch_bn, reduce_tensor, convert_sync_batchnorm, snapshot

monkey_patch_bn()

parser = argparse.ArgumentParser()
# paths
parser.add_argument("--train-mx-path")
parser.add_argument("--train-mx-index-path")
parser.add_argument("--val-mx-path")
parser.add_argument("--val-mx-index-path")
parser.add_argument("--checkpoint-dir")

# script params
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--use-apex", action="store_true", default=False)
parser.add_argument("--experiment-name", type=str, default="test")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--prof", action="store_true", default=False)

# hyperparams
parser.add_argument("--use-syncbn", action="store_true", default=False)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--upscale-factor", type=int, default=2)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--g-lr", type=float, default=1e-3)
parser.add_argument("--d-lr", type=float, default=1e-3)
parser.add_argument("--crop-size", type=int, default=88)

args = parser.parse_args()
local_rank = args.local_rank
experiment_name = args.experiment_name
use_apex = args.use_apex
prof = args.prof

train_mx_path = os.path.expanduser(args.train_mx_path)
train_mx_index_path = os.path.expanduser(args.train_mx_index_path)
val_mx_path = os.path.expanduser(args.val_mx_path)
val_mx_index_path = os.path.expanduser(args.val_mx_index_path)
checkpoint_dir = os.path.expanduser(args.checkpoint_dir)

use_syncbn = args.use_syncbn
channels = args.channels
upscale_factor = args.upscale_factor
epochs = args.epochs
batch_size = args.batch_size
workers = args.workers
g_lr = args.g_lr
d_lr = args.d_lr
crop_size = args.crop_size

print_freq = 10

assert os.path.exists(train_mx_path)
assert os.path.exists(train_mx_index_path)
assert os.path.exists(val_mx_path)
assert os.path.exists(val_mx_index_path)
assert experiment_name
assert os.path.exists(checkpoint_dir)

distributed = False
world_size = 1

print(f"GPU {local_rank} reporting for duty")

if local_rank == 0:
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

snapshot(checkpoint_dir)

if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
    distributed = world_size > 1

netG = Generator(scale_factor=upscale_factor, in_channels=channels)
netD = Discriminator(in_channels=channels)
g = GeneratorLoss()
netG.cuda(local_rank)
netD.cuda(local_rank)
g.cuda(local_rank)

if distributed:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    assert world_size == torch.distributed.get_world_size()
    g_lr *= world_size
    d_lr *= world_size
    if use_apex:
        if use_syncbn:
            netG = apex.parallel.convert_syncbn_model(netG)
            netD = apex.parallel.convert_syncbn_model(netD)
        netG = DistributedDataParallel(netG, delay_allreduce=True)
        netD = DistributedDataParallel(netD, delay_allreduce=True)
    else:
        if use_syncbn:
            netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG)
            netD = nn.SyncBatchNorm.convert_sync_batchnorm(netD)
        netG = nn.parallel.DistributedDataParallel(
            netG, device_ids=[local_rank], broadcast_buffers=False
        )
        netD = nn.parallel.DistributedDataParallel(
            netD, device_ids=[local_rank], broadcast_buffers=False
        )
else:
    netG = Generator(scale_factor=upscale_factor, in_channels=channels)
    netD = Discriminator(in_channels=channels)
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

# because vgg excepts 3 channels
if channels == 1:
    generator_loss = lambda fake_out, fake_img, hr_image: g(
        fake_out,
        torch.cat([fake_img, fake_img, fake_img], dim=1),
        torch.cat([hr_image, hr_image, hr_image], dim=1),
    )
else:
    generator_loss = g

optimizerG = torch.optim.Adam(netG.parameters(), lr=g_lr)
optimizerD = torch.optim.Adam(netD.parameters(), lr=d_lr)
# optimizerG = torch.optim.SGD(netG.parameters(), g_lr, momentum=0.9, weight_decay=1e-4)
# optimizerD = torch.optim.SGD(netD.parameters(), d_lr, momentum=0.9, weight_decay=1e-4)

train_pipe = SRGANMXNetPipeline(
    batch_size=batch_size,
    num_gpus=world_size,
    num_threads=workers,
    device_id=local_rank,
    crop=crop_size,
    mx_path=train_mx_path,
    mx_index_path=train_mx_index_path,
    upscale_factor=upscale_factor,
    image_type=types.DALIImageType.RGB,
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
    mx_path=val_mx_path,
    mx_index_path=val_mx_index_path,
    upscale_factor=upscale_factor,
    random_shuffle=False,
    image_type=types.DALIImageType.RGB,
)
val_pipe.build()
val_loader = StupidDALIIterator(
    pipelines=[val_pipe],
    output_map=["lr_image", "hr_image"],
    size=int(val_pipe.epoch_size("Reader") / world_size),
    auto_reset=False,
)

g_loss_meter = AverageMeter("g_loss")
d_loss_meter = AverageMeter("d_loss")
sample_speed_meter = AverageMeter("sample_speed")


def train(epoch):
    g_loss_meter.reset()
    d_loss_meter.reset()
    sample_speed_meter.reset()
    netG.train()
    netD.train()

    for i, (lr_image, hr_image) in enumerate(train_loader):
        start = time.time()
        batch_size = lr_image.shape[0]

        adjust_learning_rate(
            optimizerD, epoch, i, train_loader.size, d_lr
        )
        adjust_learning_rate(
            optimizerG, epoch, i, train_loader.size, g_lr
        )

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
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_loss(fake_out, fake_img, hr_image)
        g_loss.backward()
        optimizerG.step()

        if distributed:
            d_loss_meter.update(reduce_tensor(d_loss.data, world_size).item())
            g_loss_meter.update(reduce_tensor(g_loss.data, world_size).item())
        else:
            d_loss_meter.update(d_loss.item())
            g_loss_meter.update(g_loss.item())

        sample_speed_meter.update(world_size * batch_size / (time.time() - start))

        if local_rank == 0 and i % print_freq == 0:
            print(
                "\t".join(
                    [
                        f"epoch {epoch}",
                        f"step {i + 1}/{train_loader.size // batch_size}",
                        str(sample_speed_meter),
                        str(d_loss_meter),
                        str(g_loss_meter),
                    ]
                )
            )


mse_meter = AverageMeter("mse")
ssim_meter = AverageMeter("ssim")
psnr_meter = AverageMeter("psnr")


def validate(epoch):
    mse_meter.reset()
    ssim_meter.reset()
    psnr_meter.reset()
    netG.eval()
    for i, (lr_image, hr_image) in enumerate(val_loader):
        batch_size = lr_image.shape[0]
        if prof and i > 10:
            break

        with torch.no_grad():
            sr_image = netG(lr_image)

        batch_mse = ((sr_image - hr_image) ** 2).mean()
        batch_ssim = ssim(sr_image, hr_image)

        if distributed:
            mse_meter.update(reduce_tensor(batch_mse.data, world_size), batch_size)
            ssim_meter.update(reduce_tensor(batch_ssim.data, world_size), batch_size)
        else:
            mse_meter.update(batch_mse.item(), batch_size)
            ssim_meter.update(batch_ssim.item(), batch_size)

    psnr_meter.update(10 * log10(1 / mse_meter.avg))

    if local_rank == 0:
        print(
            "\t".join(
                [
                    "\033[1;31m" f"epoch {epoch}",
                    str(mse_meter),
                    str(ssim_meter),
                    str(psnr_meter),
                    "\033[1;0m",
                ]
            )
        )


def find_lr():
    print("find learning rates")

    def one_round_g(lr_image, _fake_img, hr_image):
        netD.train()
        fake_img = netG(lr_image)
        real_out = netD(hr_image).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        return generator_loss(fake_out, fake_img, hr_image)

    def one_round_d(lr_image, _real_out, hr_image):
        netG.train()
        fake_img = netG(lr_image)
        real_out = netD(hr_image).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        g_loss = generator_loss(fake_out, fake_img, hr_image)
        g_loss.backward(retain_graph=True)
        optimizerG.step()
        return d_loss

    optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4)

    lr_finder_g = LRFinder(netG, optimizerG, one_round_g)
    lr_finder_g.range_test(train_loader)
    train_loader.reset()
    lr_finder_g.plot()
    lr_finder_d = LRFinder(netD, optimizerD, one_round_d, retain_graph=True)
    lr_finder_d.range_test(train_loader)
    lr_finder_d.plot()


def adjust_learning_rate(optimizer, epoch, step, len_epoch, orig_lr):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = orig_lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)

    if local_rank == 0 and step % print_freq == 0 and step > 1:
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


epoch_time_meter = AverageMeter("epoch")

running_meters = {
    "g_loss": [],
    "d_loss": [],
    "sample_speed": [],
    "mse": [],
    "ssim": [],
    "psnr": [],
    "epoch_time": [],
}


def update_running_meters():
    global running_meters
    running_meters["g_loss"].append(g_loss_meter.avg)
    running_meters["d_loss"].append(d_loss_meter.avg)
    running_meters["sample_speed"].append(sample_speed_meter.avg)
    running_meters["mse"].append(mse_meter.avg)
    running_meters["ssim"].append(ssim_meter.avg)
    running_meters["psnr"].append(psnr_meter.avg)
    running_meters["epoch_time"].append(epoch_time_meter.val)


def main():
    for epoch in range(epochs):
        start = time.time()
        train(epoch)
        validate(epoch)
        if local_rank == 0:
            torch.save(
                netG.state_dict(),
                f"{checkpoint_dir}/netG_epoch_{upscale_factor}_{epoch}.pth",
            )
            torch.save(
                netD.state_dict(),
                f"{checkpoint_dir}/netD_epoch_{upscale_factor}_{epoch}.pth",
            )
            epoch_time_meter.update(time.time() - start)
            update_running_meters()
            if epoch != 0 and not prof:
                data_frame = pd.DataFrame(data=running_meters)
                data_frame.to_csv(
                    os.path.join(checkpoint_dir, "metrics.csv"), index_label="Epoch"
                )

        val_loader.reset()
        train_loader.reset()


if __name__ == "__main__":
    main()
    # find_lr()
