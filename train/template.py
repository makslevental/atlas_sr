import argparse
import csv
import inspect
import os
import time
from dataclasses import dataclass

import apex
import torch
import torch.backends.cudnn
import torch.distributed
import torchvision.models as models
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
from util.util import (
    snapshot,
    clear_directory,
    dict_to_yaml_str,
    load_model_state,
    adjust_learning_rate,
)


@dataclass
class Learner:
    model: nn.Module
    optimizer: Optimizer
    train_loader: StupidDALIIterator
    val_loader: StupidDALIIterator
    loss: nn.Module
    summary_writer: SummaryWriter


@dataclass
class Metrics:
    loss = AverageMeter("loss")
    mse = AverageMeter("mse")
    sample_speed = AverageMeter("s/s")
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
    parser.add_argument("--model-pth", default=None)
    parser.add_argument("--opt-pth", default=None)

    # script params
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--use-apex", action="store_true", default=True)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--prof", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--start-epoch", type=int, default=0)

    # hyperparams
    parser.add_argument("--use-syncbn", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    config = parser.parse_args()

    config.train_mx_path = os.path.expanduser(config.train_mx_path)
    config.train_mx_index_path = os.path.expanduser(config.train_mx_index_path)
    config.val_mx_path = os.path.expanduser(config.val_mx_path)
    config.val_mx_index_path = os.path.expanduser(config.val_mx_index_path)
    config.checkpoint_dir = os.path.expanduser(config.checkpoint_dir)
    config.tensorboard_dir = os.path.expanduser(config.tensorboard_dir)
    if config.model_pth is not None:
        config.model_pth = os.path.expanduser(config.model_pth)
    if config.opt_pth is not None:
        config.opt_pth = os.path.expanduser(config.opt_pth)

    if config.resume:
        assert config.model_pth is not None
        assert config.opt_pth is not None

    assert os.path.exists(config.train_mx_path)
    assert os.path.exists(config.train_mx_index_path)
    assert os.path.exists(config.val_mx_path)
    assert os.path.exists(config.val_mx_index_path)
    assert os.path.exists(config.checkpoint_dir)
    assert os.path.exists(config.tensorboard_dir)
    assert config.experiment_name

    print(f"GPU {config.local_rank} reporting for duty")

    if "WORLD_SIZE" in os.environ:
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.distributed = config.world_size > 1
    else:
        config.world_size = 1
        config.distributed = False

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
    model = models.vgg16()

    if config.net_g_pth is not None:
        model = load_model_state(model, config.net_g_pth)

    loss = nn.CrossEntropyLoss()
    model.cuda(config.local_rank)
    loss.cuda(config.local_rank)

    if config.distributed:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        assert config.world_size == torch.distributed.get_world_size()
        if config.use_apex:
            if config.use_syncbn:
                model = apex.parallel.convert_syncbn_model(model)
            model = DistributedDataParallel(model, delay_allreduce=True)
        else:
            if config.use_syncbn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[config.local_rank], broadcast_buffers=False
            )
    else:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.g_lr)
    if config.opt_pth is not None:
        optimizer = load_model_state(optimizer, config.opt_pth)

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

    return Learner(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        summary_writer=summary_writer,
    )


def train(epoch, config: argparse.Namespace, l: Learner):
    l.model.train()

    if config.local_rank == 0:
        Metrics.loss.reset()
        Metrics.sample_speed.reset()
        train_bar = tqdm(
            l.train_loader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=config.local_rank,
        )
    else:
        train_bar = l.train_loader

    for i, (sample, target) in enumerate(train_bar):
        if config.prof and i > 10:
            break
        start = time.time()
        batch_size = sample.shape[0]
        if torch.cuda.is_available():
            sample = sample.cuda()
            target = target.cuda()

        lr = adjust_learning_rate(
            l.optimizer, epoch, i, l.train_loader.size, config.d_lr
        )

        l.model.zero_grad()
        output = l.model(sample)

        loss = l.loss(output, target)
        loss.backward()
        l.optimizer.step()

        ############################
        # Collect metrics
        ###########################
        if config.local_rank == 0:
            Metrics.loss.update(loss.item(), batch_size)
            Metrics.sample_speed.update(
                config.world_size * batch_size / (time.time() - start)
            )

            step = i + len(l.train_loader) * epoch
            l.summary_writer.add_scalar(f"train/loss", Metrics.loss.val, step)
            l.summary_writer.add_scalar(
                f"train/sample_speed", Metrics.sample_speed.val, step
            )
            l.summary_writer.add_scalar(f"train/lr", lr, step)

            if i % config.print_freq == 0:
                train_bar.set_description_str(
                    "  ".join(
                        [f"{epoch}", str(Metrics.loss), str(Metrics.sample_speed)]
                    )
                )


def validate(epoch, config: argparse.Namespace, l: Learner):
    l.model.eval()
    if config.local_rank == 0:
        Metrics.mse.reset()
        val_bar = tqdm(
            l.val_loader,
            "val",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            position=config.local_rank,
        )
    else:
        val_bar = l.val_loader

    for i, (sample, target) in enumerate(val_bar):
        if config.prof and i > 10:
            break
        if torch.cuda.is_available():
            sample = sample.cuda()
            target = target.cuda()
        batch_size = sample.shape[0]

        with torch.no_grad():
            output = l.model(sample)

        batch_mse = ((output - target) ** 2).mean()

        ############################
        # Collect metrics
        ###########################
        if config.local_rank == 0:
            Metrics.loss.update(batch_mse.item(), batch_size)
            if i % config.print_freq == 0:
                val_bar.set_description_str(
                    "  ".join(["\033[1;31m", f"{epoch}", str(Metrics.mse), "\033[1;0m"])
                )
    if config.local_rank == 0:
        l.summary_writer.add_scalar(f"val/mse", Metrics.mse.avg, epoch)


def save_checkpoint(epoch, start, config: argparse.Namespace, l: Learner):
    Metrics.epoch_time.update(time.time() - start)
    if not config.prof:
        with open(os.path.join(config.checkpoint_dir, "metrics.csv"), "a+") as csvfile:
            metrics_writer = csv.writer(csvfile)
            if epoch == 0:
                metrics_writer.writerow(
                    [
                        "epoch",
                        "mse.avg",
                        "loss.avg",
                        "epoch_time.val",
                        "sample_speed.avg",
                    ]
                )
            metrics_writer.writerow(
                [
                    epoch,
                    Metrics.loss.avg,
                    Metrics.mse.avg,
                    Metrics.epoch_time.val,
                    Metrics.sample_speed.avg,
                ]
            )

    Metrics.reset()
    torch.save(
        l.model.state_dict(), f"{config.checkpoint_dir}/model_epoch_{epoch:04}.pth"
    )
    torch.save(
        l.optimizer.state_dict(),
        f"{config.checkpoint_dir}/optimizer_epoch_{epoch:04}.pth",
    )


def main_loop(config: argparse.Namespace, learner: Learner):
    for epoch in range(config.start_epoch, config.epochs):
        start = time.time()
        train(epoch, config, learner)
        validate(epoch, config, learner)
        if config.local_rank == 0:
            save_checkpoint(epoch, start, config, learner)


if __name__ == "__main__":
    config = setup()
    learner = build_learner(config)
    main_loop(config, learner)
