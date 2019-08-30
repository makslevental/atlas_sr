import argparse
import os
import time
from math import log10

import pandas as pd
import torch
import torch.backends.cudnn
import torch.distributed

from data_utils.dali import SRGANMXNetPipeline
from metrics.ssim import ssim
from models.SRGAN import Generator, Discriminator, GeneratorLoss

try:
    from nvidia.dali.plugin.pytorch import (
        DALIClassificationIterator,
        DALIGenericIterator,
    )
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--train-mx-path")
parser.add_argument("--train-mx-index-path")
parser.add_argument("--val-mx-path")
parser.add_argument("--val-mx-index-path")
parser.add_argument("--checkpoint-dir")

args = parser.parse_args()
local_rank = args.local_rank
train_mx_path = args.train_mx_path
train_mx_index_path = args.train_mx_index_path
val_mx_path = args.val_mx_path
val_mx_index_path = args.val_mx_index_path
checkpoint_dir = args.checkpoint_dir

assert os.path.exists(train_mx_path)
assert os.path.exists(train_mx_index_path)
assert os.path.exists(val_mx_path)
assert os.path.exists(val_mx_index_path)
assert os.path.exists(checkpoint_dir)

fp16 = False
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

# make apex optional
if fp16 or distributed:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to run this example."
        )

if fp16:
    assert (
        torch.backends.cudnn.enabled
    ), "fp16 mode requires cudnn backend to be enabled."

upscale_factor = 2
epochs = 100
batch_size = 4
total_batch_size = world_size * batch_size
prof = False
static_loss_scale = 1.0
dynamic_loss_scale = False
# lr = 0.1
# momentum = 0.9
workers = 4
# weight_decay = 1e-4
resume = False
dali_cpu = False
print_freq = 10

if static_loss_scale != 1.0:
    if not fp16:
        print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

netG = Generator(scale_factor=upscale_factor, in_channels=1)
netD = Discriminator(in_channels=1)
generator_criterion = GeneratorLoss()

netG = netG.cuda()
netD = netD.cuda()
generator_criterion = generator_criterion.cuda()
if fp16:
    netG = network_to_half(netG)
    netD = network_to_half(netD)
    generator_criterion = network_to_half(generator_criterion)
if distributed:
    # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
    # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
    netG = DDP(netG, delay_allreduce=True)
    netD = DDP(netD, delay_allreduce=True)
    generator_criterion = DDP(generator_criterion, delay_allreduce=True)

# optimizerG = torch.optim.SGD(
#     netG.parameters(), lr, momentum=momentum, weight_decay=weight_decay
# )
# optimizerD = torch.optim.SGD(
#     netD.parameters(), lr, momentum=momentum, weight_decay=weight_decay
# )

optimizerG = torch.optim.Adam(netG.parameters())
optimizerD = torch.optim.Adam(netD.parameters())

if fp16:
    optimizerG = FP16_Optimizer(
        optimizerG,
        static_loss_scale=static_loss_scale,
        dynamic_loss_scale=dynamic_loss_scale,
    )
    optimizerD = FP16_Optimizer(
        optimizerD,
        static_loss_scale=static_loss_scale,
        dynamic_loss_scale=dynamic_loss_scale,
    )

crop_size = 224
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
train_loader = DALIGenericIterator(
    pipelines=[train_pipe],
    output_map=["lr_image", "hr_image"],
    size=int(train_pipe.epoch_size("Reader") / world_size),
    auto_reset=True,
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
)
val_pipe.build()
val_loader = DALIGenericIterator(
    pipelines=[val_pipe],
    output_map=["lr_image", "hr_image"],
    size=int(val_pipe.epoch_size("Reader") / world_size),
    auto_reset=True,
)


def train(epoch):
    netG.train()
    netD.train()

    batch_time = AverageMeter()
    d_losses = AverageMeter()
    g_losses = AverageMeter()
    end = time.time()

    for i, data in enumerate(train_loader):
        # dali for some reason outputs HWC, we need CHW
        lr_image = data[0]["lr_image"]
        hr_image = data[0]["hr_image"]

        batch_size = lr_image.shape[0]
        train_loader_len = train_loader._size // batch_size
        # adjust_learning_rate(optimizerG, epoch, i, train_loader_len)
        # adjust_learning_rate(optimizerD, epoch, i, train_loader_len)

        if prof and i > 10:
            break

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ##########################
        netD.zero_grad()
        fake_img = netG(lr_image)
        real_out = netD(hr_image).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_losses.update(d_loss.item(), batch_size)

        optimizerD.zero_grad()
        if fp16:
            optimizerD.backward(d_loss, retain_graph=True)
        else:
            d_loss.backward(retain_graph=True)
        optimizerD.step()
        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()

        g_loss = generator_criterion(
            fake_out,
            torch.cat([fake_img, fake_img, fake_img], dim=1),
            torch.cat([hr_image, hr_image, hr_image], dim=1),
        )
        g_losses.update(g_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizerG.zero_grad()
        if fp16:
            optimizerG.backward(g_loss)
        else:
            g_loss.backward()
        optimizerG.step()

        torch.cuda.synchronize(device=torch.cuda.current_device())
        # record stats
        batch_time.update(time.time() - end)
        end = time.time()
        speed = total_batch_size / batch_time.val
        avg_speed = total_batch_size / batch_time.avg
        if local_rank == 0 and i % print_freq == 0 and i > 1:
            print(
                f"Epoch: {epoch}\t"
                f"Batch: {i}/{train_loader_len}\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Speed {speed:.3f} ({avg_speed:.3f})\t"
                f"G Loss {g_losses.val:.4f} ({g_losses.avg:.4f})\t"
                f"D Loss {d_losses.val:.4f} ({d_losses.avg:.4f})\t"
            )
    return batch_time.avg


def validate():
    netG.eval()
    valing_results = {"mse": 0, "ssims": 0, "psnr": 0, "ssim": 0, "batch_sizes": 0}
    for i, data in enumerate(val_loader):
        # dali for some reason outputs HWC, we need CHW
        if prof and i > 10:
            break

        lr_image = data[0]["lr_image"]
        hr_image = data[0]["hr_image"]
        sr_image = netG(lr_image)

        batch_size = lr_image.size(0)
        valing_results["batch_sizes"] += batch_size
        batch_mse = ((sr_image - hr_image) ** 2).data.mean()
        valing_results["mse"] += batch_mse * batch_size
        batch_ssim = ssim(sr_image, hr_image).item()
        valing_results["ssims"] += batch_ssim * batch_size
        valing_results["psnr"] = 10 * log10(
            1 / (valing_results["mse"] / valing_results["batch_sizes"])
        )
        valing_results["ssim"] = valing_results["ssims"] / valing_results["batch_sizes"]

    return valing_results


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch, step, len_epoch):
#     """LR schedule that should yield 76% converged accuracy with batch size 256"""
#     factor = epoch // 30
#
#     if epoch >= 80:
#         factor = factor + 1
#
#     new_lr = lr * (0.1 ** factor)
#
#     """Warmup"""
#     if epoch < 5:
#         new_lr = new_lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)
#
#     if local_rank == 0 and step % print_freq == 0 and step > 1:
#         print("Update LR: Epoch = {}, step = {}, lr = {}".format(epoch, step, new_lr))
#
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = new_lr


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= world_size
    return rt


if __name__ == "__main__":
    total_time = AverageMeter()
    results = {"psnr": [], "ssim": [], "g_lr": [], "d_lr": []}
    for epoch in range(epochs):
        avg_train_time = train(epoch)
        total_time.update(avg_train_time)

        val_results = validate()

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
        results["g_lr"].append(optimizerG.param_groups[0]["lr"])
        results["d_lr"].append(optimizerD.param_groups[0]["lr"])
        if epoch != 0:
            data_frame = pd.DataFrame(
                data={
                    "PSNR": results["psnr"],
                    "SSIM": results["ssim"],
                    "G_LR": results["g_lr"],
                    "D_LR": results["d_lr"],
                },
            )
            data_frame.to_csv(
                os.path.join(checkpoint_dir, "metrics.csv"), index_label="Epoch"
            )

        train_loader.reset()
        val_loader.reset()
