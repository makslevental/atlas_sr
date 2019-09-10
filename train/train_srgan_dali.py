import argparse
import csv
import inspect
import os
import time
from dataclasses import dataclass
from math import log10

import apex
import torch
import torch.backends.cudnn
import torch.distributed
from apex.parallel import DistributedDataParallel
from nvidia.dali import types
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from data_utils.dali import (
    StupidDALIIterator,
    SRGANMXNetTrainPipeline,
    SRGANMXNetValPipeline,
)
from metrics.metrics import AverageMeter
from metrics.ssim import ssim
from models.SRGAN import Generator, Discriminator, GeneratorLoss
from util.util import snapshot, clear_directory


@dataclass
class SRGANLearner:
    netG: nn.Module
    netD: nn.Module
    optimizerG: Optimizer
    optimizerD: Optimizer
    train_loader: StupidDALIIterator
    val_loader: StupidDALIIterator
    generator_loss: nn.Module
    mse_loss: nn.Module


@dataclass
class Metrics:
    g_loss = AverageMeter("g_loss")
    d_loss = AverageMeter("d_loss")
    g_score = AverageMeter("d(g(z))")
    d_score = AverageMeter("d(x)")
    sample_speed = AverageMeter("s/s")
    mse = AverageMeter("mse")
    ssim = AverageMeter("ssim")
    psnr = AverageMeter("psnr")
    epoch_time = AverageMeter("e_time")

    @classmethod
    def reset(cls):
        metrics = [
            (name, obj)
            for (name, obj) in inspect.getmembers(cls)
            if not name.startswith("__") and name != "reset"
        ]
        for _, metric in metrics:
            metric.reset()

    # def __setattr__(self, key, value):
    #     if isinstance(value, Tuple):
    #         assert len(value) == 2
    #         val, n = value
    #         getattr(self, key).update(val, n)
    #     else:
    #         getattr(self, key).update(value)


def setup():
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
    parser.add_argument("--print-freq", type=int, default=10)
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

    args.train_mx_path = os.path.expanduser(args.train_mx_path)
    args.train_mx_index_path = os.path.expanduser(args.train_mx_index_path)
    args.val_mx_path = os.path.expanduser(args.val_mx_path)
    args.val_mx_index_path = os.path.expanduser(args.val_mx_index_path)
    args.checkpoint_dir = os.path.expanduser(args.checkpoint_dir)

    assert os.path.exists(args.train_mx_path)
    assert os.path.exists(args.train_mx_index_path)
    assert os.path.exists(args.val_mx_path)
    assert os.path.exists(args.val_mx_index_path)
    assert os.path.exists(args.checkpoint_dir)
    assert args.experiment_name

    print(f"GPU {args.local_rank} reporting for duty")

    if args.local_rank == 0:
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        clear_directory(args.checkpoint_dir)
        snapshot(args.checkpoint_dir)

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1
        args.distributed = False

    # fundamental working rate is 128 @ 1e-3
    args.g_lr *= args.world_size
    args.d_lr *= args.world_size
    return args


def build_learner(args: argparse.Namespace):
    netG = Generator(scale_factor=args.upscale_factor)
    netD = Discriminator()
    g = GeneratorLoss()
    netG.cuda(args.local_rank)
    netD.cuda(args.local_rank)
    g.cuda(args.local_rank)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        assert args.world_size == torch.distributed.get_world_size()
        if args.use_apex:
            if args.use_syncbn:
                netG = apex.parallel.convert_syncbn_model(netG)
                netD = apex.parallel.convert_syncbn_model(netD)
            netG = DistributedDataParallel(netG, delay_allreduce=True)
            netD = DistributedDataParallel(netD, delay_allreduce=True)
        else:
            if args.use_syncbn:
                netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG)
                netD = nn.SyncBatchNorm.convert_sync_batchnorm(netD)
            netG = nn.parallel.DistributedDataParallel(
                netG, device_ids=[args.local_rank], broadcast_buffers=False
            )
            netD = nn.parallel.DistributedDataParallel(
                netD, device_ids=[args.local_rank], broadcast_buffers=False
            )
    else:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # because vgg excepts 3 channels
    if args.channels == 1:
        generator_loss = lambda fake_out, fake_img, hr_image: g(
            fake_out,
            torch.cat([fake_img, fake_img, fake_img], dim=1),
            torch.cat([hr_image, hr_image, hr_image], dim=1),
        )
    else:
        generator_loss = g

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.g_lr)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.d_lr)
    # optimizerG = torch.optim.SGD(netG.parameters(), g_lr, momentum=0.9, weight_decay=1e-4)
    # optimizerD = torch.optim.SGD(netD.parameters(), d_lr, momentum=0.9, weight_decay=1e-4)

    train_pipe = SRGANMXNetTrainPipeline(
        batch_size=args.batch_size,
        num_gpus=args.world_size,
        num_threads=args.workers,
        device_id=args.local_rank,
        crop=args.crop_size,
        mx_path=args.train_mx_path,
        mx_index_path=args.train_mx_index_path,
        upscale_factor=args.upscale_factor,
        image_type=types.DALIImageType.RGB,
    )
    train_pipe.build()
    train_loader = StupidDALIIterator(
        pipelines=[train_pipe],
        output_map=["lr_image", "hr_image"],
        size=int(train_pipe.epoch_size("Reader") / args.world_size),
        auto_reset=True,
    )
    val_pipe = SRGANMXNetValPipeline(
        batch_size=args.batch_size,
        num_gpus=args.world_size,
        num_threads=args.workers,
        device_id=args.local_rank,
        crop=args.crop_size,
        mx_path=args.val_mx_path,
        mx_index_path=args.val_mx_index_path,
        upscale_factor=args.upscale_factor,
        random_shuffle=False,
        image_type=types.DALIImageType.RGB,
    )
    val_pipe.build()
    val_loader = StupidDALIIterator(
        pipelines=[val_pipe],
        output_map=["lr_image", "hr_image"],
        size=int(val_pipe.epoch_size("Reader") / args.world_size),
        auto_reset=True,
    )

    return SRGANLearner(
        netG=netG,
        netD=netD,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        train_loader=train_loader,
        val_loader=val_loader,
        generator_loss=generator_loss,
        mse_loss=nn.MSELoss(),
    )


def train_srresnet(epoch, args: argparse.Namespace, l: SRGANLearner):
    Metrics.g_loss.reset()
    Metrics.sample_speed.reset()
    l.netG.train()

    if args.local_rank == 0:
        train_bar = tqdm(
            l.train_loader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=args.local_rank,
        )
    else:
        train_bar = l.train_loader

    for i, (lr_image, hr_image) in enumerate(train_bar):
        start = time.time()
        batch_size = lr_image.shape[0]
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        # adjust_learning_rate(l.optimizerD, epoch, i, l.train_loader.size, args.d_lr)
        # adjust_learning_rate(l.optimizerG, epoch, i, l.train_loader.size, args.g_lr)

        if args.prof and i > 10:
            break

        l.netG.zero_grad()
        fake_img = l.netG(lr_image)

        g_loss = l.mse_loss(fake_img, hr_image)
        g_loss.backward()
        l.optimizerG.step()

        Metrics.g_loss.update(g_loss.item(), batch_size)
        Metrics.sample_speed.update(
            args.world_size * batch_size / (time.time() - start)
        )

        if args.local_rank == 0 and i % args.print_freq == 0:
            train_bar.set_description_str(
                "  ".join([f"{epoch}", str(Metrics.g_loss), str(Metrics.sample_speed)])
            )


def train(epoch, args: argparse.Namespace, l: SRGANLearner):
    Metrics.g_loss.reset()
    Metrics.d_loss.reset()
    Metrics.g_score.reset()
    Metrics.d_score.reset()
    Metrics.sample_speed.reset()
    l.netG.train()
    l.netD.train()

    if args.local_rank == 0:
        train_bar = tqdm(
            l.train_loader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=args.local_rank,
        )
    else:
        train_bar = l.train_loader

    for i, (lr_image, hr_image) in enumerate(train_bar):
        start = time.time()
        batch_size = lr_image.shape[0]
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        # adjust_learning_rate(l.optimizerD, epoch, i, l.train_loader.size, args.d_lr)
        # adjust_learning_rate(l.optimizerG, epoch, i, l.train_loader.size, args.g_lr)

        if args.prof and i > 10:
            break

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ##########################
        fake_img = l.netG(lr_image)
        l.netD.zero_grad()
        real_out = l.netD(hr_image).mean()
        fake_out = l.netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        l.optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        l.netG.zero_grad()
        g_loss = l.generator_loss(fake_out, fake_img, hr_image)
        g_loss.backward()
        l.optimizerG.step()

        ############################
        # (3) Collect metrics
        ###########################
        fake_img = l.netG(lr_image)
        fake_out = l.netD(fake_img).mean()

        g_loss = l.generator_loss(fake_out, fake_img, hr_image)
        Metrics.g_loss.update(g_loss.item(), batch_size)
        d_loss = 1 - real_out + fake_out
        Metrics.d_loss.update(d_loss.item(), batch_size)
        Metrics.d_score.update(real_out.item(), batch_size)
        Metrics.g_score.update(fake_out.item(), batch_size)
        Metrics.sample_speed.update(
            args.world_size * batch_size / (time.time() - start)
        )

        if args.local_rank == 0 and i % args.print_freq == 0:
            train_bar.set_description_str(
                "  ".join(
                    [
                        f"{epoch}",
                        str(Metrics.d_loss),
                        str(Metrics.g_loss),
                        str(Metrics.d_score),
                        str(Metrics.g_score),
                        str(Metrics.sample_speed),
                    ]
                )
            )


def validate(epoch, args: argparse.Namespace, l: SRGANLearner):
    Metrics.mse.reset()
    Metrics.ssim.reset()
    Metrics.psnr.reset()
    l.netG.eval()
    if args.local_rank == 0:
        val_bar = tqdm(
            l.val_loader,
            "val",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=args.local_rank,
        )
    else:
        val_bar = l.val_loader

    for i, (lr_image, hr_image) in enumerate(val_bar):
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
        batch_size = lr_image.shape[0]
        if args.prof and i > 10:
            break

        with torch.no_grad():
            sr_image = l.netG(lr_image)

        batch_mse = ((sr_image - hr_image) ** 2).mean()
        batch_ssim = ssim(sr_image, hr_image).item()

        Metrics.mse.update(batch_mse, batch_size)
        Metrics.ssim.update(batch_ssim, batch_size)
        Metrics.psnr.set(10 * log10(1 / Metrics.mse.avg))
        if args.local_rank == 0 and i % args.print_freq == 0:
            val_bar.set_description_str(
                "  ".join(
                    [
                        "\033[1;31m",
                        f"{epoch}",
                        f"mse {Metrics.mse.sum}",
                        str(Metrics.ssim),
                        str(Metrics.psnr),
                        "\033[1;0m",
                    ]
                )
            )


def adjust_learning_rate(optimizer, epoch, step, len_epoch, orig_lr):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = orig_lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(epoch, start, args: argparse.Namespace, l: SRGANLearner):
    Metrics.epoch_time.update(time.time() - start)
    if not args.prof:
        with open(os.path.join(args.checkpoint_dir, "metrics.csv"), "a+") as csvfile:
            metrics_writer = csv.writer(csvfile)
            if epoch == 0:
                metrics_writer.writerow(
                    [
                        "epoch",
                        "psnr.val",
                        "ssim.val",
                        "d_loss.avg",
                        "d_score.avg",
                        "g_loss.avg",
                        "g_score.avg",
                        "epoch_time.val",
                        "sample_speed.avg",
                    ]
                )
            metrics_writer.writerow(
                [
                    epoch,
                    Metrics.psnr.val,
                    Metrics.ssim.val,
                    Metrics.d_loss.avg,
                    Metrics.d_score.avg,
                    Metrics.g_loss.avg,
                    Metrics.g_score.avg,
                    Metrics.epoch_time.val,
                    Metrics.sample_speed.avg,
                ]
            )

    Metrics.reset()
    torch.save(
        l.netG.state_dict(),
        f"{args.checkpoint_dir}/netG_epoch_{epoch:04}.pth",
    )
    torch.save(
        l.netD.state_dict(),
        f"{args.checkpoint_dir}/netD_epoch_{epoch:04}.pth",
    )


def main_srresnet_loop(args: argparse.Namespace, learner: SRGANLearner):
    for epoch in range(args.epochs):
        start = time.time()
        train_srresnet(epoch, args, learner)
        validate(epoch, args, learner)
        if args.local_rank == 0:
            save_checkpoint(epoch, start, args, learner)


def main_loop(args: argparse.Namespace, learner: SRGANLearner):
    for epoch in range(args.epochs):
        start = time.time()
        train(epoch, args, learner)
        validate(epoch, args, learner)
        if args.local_rank == 0:
            save_checkpoint(epoch, start, args, learner)


if __name__ == "__main__":
    args = setup()
    learner = build_learner(args)
    # main_loop(args, learner)
    main_srresnet_loop(args, learner)
