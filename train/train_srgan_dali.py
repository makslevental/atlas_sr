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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils.dali import (
    StupidDALIIterator,
    SRGANMXNetTrainPipeline,
    SRGANMXNetValPipeline,
)
from metrics.metrics import AverageMeter
from metrics.ssim import ssim
from models.SRGAN import Generator, Discriminator, GeneratorLoss
from util.util import (
    snapshot,
    clear_directory,
    dict_to_yaml_str,
    load_model_state,
    adjust_learning_rate,
)


@dataclass
class SRGANLearner:
    netG: Generator
    netD: Discriminator
    optimizerG: Optimizer
    optimizerD: Optimizer
    train_loader: StupidDALIIterator
    val_loader: StupidDALIIterator
    generator_loss: GeneratorLoss
    mse_loss: nn.MSELoss
    summary_writer: SummaryWriter


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
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--net-g-pth", default=None)
    parser.add_argument("--net-d-pth", default=None)
    parser.add_argument("--opt-g-pth", default=None)
    parser.add_argument("--opt-d-pth", default=None)

    # script params
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--use-apex", action="store_true", default=True)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--prof", action="store_true", default=False)
    parser.add_argument("--srresnet", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--start-epoch", type=int, default=0)

    # hyperparams
    parser.add_argument("--use-syncbn", action="store_true", default=False)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--upscale-factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--g-lr", type=float, default=1e-3)
    parser.add_argument("--d-lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=88)

    config = parser.parse_args()

    config.train_mx_path = os.path.expanduser(config.train_mx_path)
    config.train_mx_index_path = os.path.expanduser(config.train_mx_index_path)
    config.val_mx_path = os.path.expanduser(config.val_mx_path)
    config.val_mx_index_path = os.path.expanduser(config.val_mx_index_path)
    config.checkpoint_dir = os.path.expanduser(config.checkpoint_dir)
    config.tensorboard_dir = os.path.expanduser(config.tensorboard_dir)
    if config.net_g_pth is not None:
        config.net_g_pth = os.path.expanduser(config.net_g_pth)
    if config.net_d_pth is not None:
        config.net_d_pth = os.path.expanduser(config.net_d_pth)
    if config.opt_g_pth is not None:
        config.opt_g_pth = os.path.expanduser(config.opt_g_pth)
    if config.opt_d_pth is not None:
        config.opt_d_pth = os.path.expanduser(config.opt_d_pth)

    if config.resume:
        assert config.net_g_pth is not None
        assert config.net_d_pth is not None
        assert config.opt_g_pth is not None
        assert config.opt_d_pth is not None

    assert os.path.exists(config.train_mx_path)
    assert os.path.exists(config.train_mx_index_path)
    assert os.path.exists(config.val_mx_path)
    assert os.path.exists(config.val_mx_index_path)
    assert os.path.exists(config.checkpoint_dir)
    assert os.path.exists(config.tensorboard_dir)
    assert config.experiment_name

    print(f"GPU {config.local_rank} reporting for duty")

    if config.local_rank == 0:
        config.checkpoint_dir = os.path.join(
            config.checkpoint_dir, config.experiment_name
        )
        if not os.path.exists(config.checkpoint_dir):
            os.mkdir(config.checkpoint_dir)
        config.tensorboard_dir = os.path.join(
            config.tensorboard_dir, config.experiment_name
        )
        if not os.path.exists(config.tensorboard_dir):
            os.mkdir(config.tensorboard_dir)

        if not config.resume:
            clear_directory(config.checkpoint_dir)
            clear_directory(config.tensorboard_dir)
        snapshot(config.checkpoint_dir)

    return config


def build_learner(config: argparse.Namespace):
    if "WORLD_SIZE" in os.environ:
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.distributed = config.world_size > 1
    else:
        config.world_size = 1
        config.distributed = False

    netG = Generator(scale_factor=config.upscale_factor)
    netD = Discriminator()

    if config.net_g_pth is not None:
        netG = load_model_state(netG, config.net_g_pth)
    if config.net_d_pth is not None:
        netD = load_model_state(netD, config.net_d_pth)

    g = GeneratorLoss()
    netG.cuda(config.local_rank)
    netD.cuda(config.local_rank)
    g.cuda(config.local_rank)

    if config.distributed:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        assert config.world_size == torch.distributed.get_world_size()
        if config.use_apex:
            if config.use_syncbn:
                netG = apex.parallel.convert_syncbn_model(netG)
                netD = apex.parallel.convert_syncbn_model(netD)
            netG = DistributedDataParallel(netG, delay_allreduce=True)
            netD = DistributedDataParallel(netD, delay_allreduce=True)
        else:
            if config.use_syncbn:
                netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG)
                netD = nn.SyncBatchNorm.convert_sync_batchnorm(netD)
            netG = nn.parallel.DistributedDataParallel(
                netG, device_ids=[config.local_rank], broadcast_buffers=False
            )
            netD = nn.parallel.DistributedDataParallel(
                netD, device_ids=[config.local_rank], broadcast_buffers=False
            )
    else:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # because vgg excepts 3 channels
    if config.channels == 1:
        generator_loss = lambda fake_out, fake_img, hr_image: g(
            fake_out,
            torch.cat([fake_img, fake_img, fake_img], dim=1),
            torch.cat([hr_image, hr_image, hr_image], dim=1),
        )
    else:
        generator_loss = g

    optimizerG = torch.optim.Adam(netG.parameters(), lr=config.g_lr)
    optimizerD = torch.optim.SGD(netD.parameters(), lr=config.d_lr)
    if config.opt_g_pth is not None:
        optimizerG = load_model_state(optimizerG, config.opt_g_pth)
    if config.opt_d_pth is not None:
        optimizerD = load_model_state(optimizerD, config.opt_d_pth)

    train_pipe = SRGANMXNetTrainPipeline(
        batch_size=config.batch_size,
        num_gpus=config.world_size,
        num_threads=config.workers,
        device_id=config.local_rank,
        crop=config.crop_size,
        mx_path=config.train_mx_path,
        mx_index_path=config.train_mx_index_path,
        upscale_factor=config.upscale_factor,
        image_type=types.DALIImageType.RGB,
    )
    train_pipe.build()
    train_loader = StupidDALIIterator(
        pipelines=[train_pipe],
        output_map=["lr_image", "hr_image"],
        size=int(train_pipe.epoch_size("Reader") / config.world_size),
        auto_reset=True,
    )
    val_pipe = SRGANMXNetValPipeline(
        batch_size=config.batch_size,
        num_gpus=config.world_size,
        num_threads=config.workers,
        device_id=config.local_rank,
        crop=config.crop_size,
        mx_path=config.val_mx_path,
        mx_index_path=config.val_mx_index_path,
        upscale_factor=config.upscale_factor,
        random_shuffle=False,
        image_type=types.DALIImageType.RGB,
    )
    val_pipe.build()
    val_loader = StupidDALIIterator(
        pipelines=[val_pipe],
        output_map=["lr_image", "hr_image"],
        size=int(val_pipe.epoch_size("Reader") / config.world_size),
        auto_reset=True,
    )

    summary_writer = SummaryWriter(config.tensorboard_dir)
    if config.local_rank == 0:
        # https://stackoverflow.com/a/52784607
        args_text = "  \n".join(dict_to_yaml_str(config.__dict__).split("\n"))
        summary_writer.add_text("args", args_text)

    return SRGANLearner(
        netG=netG,
        netD=netD,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        train_loader=train_loader,
        val_loader=val_loader,
        generator_loss=generator_loss,
        mse_loss=nn.MSELoss(),
        summary_writer=summary_writer,
    )


def train_srresnet(epoch, config: argparse.Namespace, l: SRGANLearner):
    l.netG.train()

    if config.local_rank == 0:
        Metrics.g_loss.reset()
        Metrics.sample_speed.reset()
        train_bar = tqdm(
            l.train_loader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=config.local_rank,
        )
    else:
        train_bar = l.train_loader

    for i, (lr_image, hr_image) in enumerate(train_bar):
        start = time.time()
        batch_size = lr_image.shape[0]
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        # lr = adjust_learning_rate(
        #     l.optimizerG, epoch, i, l.train_loader.size, config.g_lr
        # )
        lr = config.g_lr

        if config.prof and i > 10:
            break

        l.netG.zero_grad()
        fake_img = l.netG(lr_image)

        g_loss = l.mse_loss(fake_img, hr_image)
        g_loss.backward()
        l.optimizerG.step()

        ############################
        # Collect metrics
        ###########################
        if config.local_rank == 0:
            Metrics.g_loss.update(g_loss.item(), batch_size)
            Metrics.sample_speed.update(
                config.world_size * batch_size / (time.time() - start)
            )

            step = i + len(l.train_loader) * epoch
            l.summary_writer.add_scalar(f"train/g_loss", Metrics.g_loss.val, step)
            l.summary_writer.add_scalar(
                f"train/sample_speed", Metrics.sample_speed.val, step
            )
            l.summary_writer.add_scalar(f"train/lr", lr, step)

            if i % config.print_freq == 0:
                train_bar.set_description_str(
                    "  ".join(
                        [f"{epoch}", str(Metrics.g_loss), str(Metrics.sample_speed)]
                    )
                )


def train(epoch, config: argparse.Namespace, l: SRGANLearner):
    l.netG.train()
    l.netD.train()

    if config.local_rank == 0:
        Metrics.g_loss.reset()
        Metrics.d_loss.reset()
        Metrics.g_score.reset()
        Metrics.d_score.reset()
        Metrics.sample_speed.reset()
        train_bar = tqdm(
            l.train_loader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=config.local_rank,
        )
    else:
        train_bar = l.train_loader

    for i, (lr_image, hr_image) in enumerate(train_bar):
        start = time.time()
        batch_size = lr_image.shape[0]
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        # d_lr = adjust_learning_rate(
        #     l.optimizerD, epoch, i, l.train_loader.size, config.d_lr
        # )
        # g_lr = adjust_learning_rate(
        #     l.optimizerG, epoch, i, l.train_loader.size, config.g_lr
        # )
        d_lr = config.d_lr
        g_lr = config.g_lr

        if config.prof and i > 10:
            break

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ##########################
        l.netD.zero_grad()
        fake_img = l.netG(lr_image)
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
        # Collect metrics
        ###########################
        fake_img = l.netG(lr_image)
        fake_out = l.netD(fake_img).mean()

        g_loss = l.generator_loss(fake_out, fake_img, hr_image)
        d_loss = 1 - real_out + fake_out

        if config.local_rank == 0:
            Metrics.g_loss.update(g_loss.item(), batch_size)
            Metrics.d_loss.update(d_loss.item(), batch_size)
            Metrics.d_score.update(real_out.item(), batch_size)
            Metrics.g_score.update(fake_out.item(), batch_size)
            Metrics.sample_speed.update(
                config.world_size * batch_size / (time.time() - start)
            )

            step = i + len(l.train_loader) * epoch
            l.summary_writer.add_scalar(f"train/g_loss", Metrics.g_loss.val, step)
            l.summary_writer.add_scalar(f"train/d_loss", Metrics.d_loss.val, step)
            l.summary_writer.add_scalar(f"train/g_score", Metrics.g_score.val, step)
            l.summary_writer.add_scalar(f"train/d_score", Metrics.d_score.val, step)
            l.summary_writer.add_scalar(
                f"train/sample_speed", Metrics.sample_speed.val, step
            )
            l.summary_writer.add_scalar(f"train/g_lr", g_lr, step)
            l.summary_writer.add_scalar(f"train/d_lr", d_lr, step)

            if i % config.print_freq == 0:
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


def validate(epoch, config: argparse.Namespace, l: SRGANLearner):
    l.netG.eval()
    if config.local_rank == 0:
        Metrics.mse.reset()
        Metrics.ssim.reset()
        Metrics.psnr.reset()
        val_bar = tqdm(
            l.val_loader,
            "val",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=config.local_rank,
        )
    else:
        val_bar = l.val_loader

    for i, (lr_image, hr_image) in enumerate(val_bar):
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
        batch_size = lr_image.shape[0]
        if config.prof and i > 10:
            break

        with torch.no_grad():
            sr_image = l.netG(lr_image)

        batch_mse = ((sr_image - hr_image) ** 2).mean()
        batch_ssim = ssim(sr_image, hr_image).item()

        ############################
        # Collect metrics
        ###########################
        if config.local_rank == 0:
            Metrics.mse.update(batch_mse, batch_size)
            Metrics.ssim.update(batch_ssim, batch_size)
            Metrics.psnr.set(10 * log10(1 / Metrics.mse.avg))

            if i % config.print_freq == 0:
                val_bar.set_description_str(
                    "  ".join(
                        [
                            "\033[1;31m",
                            f"{epoch}",
                            # sum is used to match leftthomas
                            f"mse {Metrics.mse.sum}",
                            str(Metrics.ssim),
                            str(Metrics.psnr),
                            "\033[1;0m",
                        ]
                    )
                )
    if config.local_rank == 0:
        l.summary_writer.add_scalar(f"val/mse", Metrics.mse.sum, epoch)
        l.summary_writer.add_scalar(f"val/ssim", Metrics.ssim.val, epoch)
        l.summary_writer.add_scalar(f"val/psnr", Metrics.psnr.val, epoch)


def save_checkpoint(epoch, config: argparse.Namespace, l: SRGANLearner):
    if not config.prof:
        with open(os.path.join(config.checkpoint_dir, "metrics.csv"), "a+") as csvfile:
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
        l.netG.state_dict(), f"{config.checkpoint_dir}/netG_epoch_{epoch:04}.pth"
    )
    torch.save(
        l.netD.state_dict(), f"{config.checkpoint_dir}/netD_epoch_{epoch:04}.pth"
    )
    torch.save(
        l.optimizerG.state_dict(), f"{config.checkpoint_dir}/optG_epoch_{epoch:04}.pth"
    )
    torch.save(
        l.optimizerD.state_dict(), f"{config.checkpoint_dir}/optD_epoch_{epoch:04}.pth"
    )


def main_srresnet_loop(config: argparse.Namespace, learner: SRGANLearner):
    for epoch in range(config.start_epoch, config.epochs):
        start = time.time()
        train_srresnet(epoch, config, learner)
        validate(epoch, config, learner)
        Metrics.epoch_time.update(time.time() - start)
        if config.local_rank == 0:
            save_checkpoint(epoch, config, learner)
            learner.summary_writer.add_scalar(f"epoch_time", Metrics.epoch_time.val, epoch)


def main_srgan_loop(config: argparse.Namespace, learner: SRGANLearner):
    for epoch in range(config.start_epoch, config.epochs):
        start = time.time()
        train(epoch, config, learner)
        validate(epoch, config, learner)
        Metrics.epoch_time.update(time.time() - start)
        if config.local_rank == 0:
            save_checkpoint(epoch, config, learner)
            learner.summary_writer.add_scalar(f"epoch_time", Metrics.epoch_time.val, epoch)


if __name__ == "__main__":
    config = setup()
    learner = build_learner(config)
    if config.srresnet:
        main_srresnet_loop(config, learner)
    else:
        main_srgan_loop(config, learner)
