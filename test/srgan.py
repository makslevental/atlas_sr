import glob
import os
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from PIL.Image import BICUBIC
from sewar import psnr
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose

from models.SRGAN import Generator


def test(model_pth_fp, upscale_factor, image):
    model = Generator(scale_factor=upscale_factor).eval()
    state_dict = {}
    for k, v in torch.load(model_pth_fp).items():
        state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)

    with torch.no_grad():
        out = model(image)

    return out
    # out_img = ToPILImage()(out[0].data.cpu())
    # fn, ext = basename(image_fp)
    #
    # if not os.path.exists(out_image_dir):
    #     os.makedirs(out_image_dir)
    #
    # out_img.save(
    #     os.path.join(out_image_dir, f"{fn}_{model_name}_{upscale_factor}x.{ext}")
    # )


def get_best_checkpoint(metrics_fp):
    df = pd.read_csv(metrics_fp)
    psnr_max = df["psnr.val"].idxmax()
    ssim_max = df["ssim.val"].idxmax()
    return psnr_max, ssim_max


def test_best_checkpoints(model_pth_dir, image_fp, upscale_factor):
    image = Image.open(image_fp)
    image_tensor = ToTensor()(image)[:3, :, :].unsqueeze(0)
    w, h = image.size
    r1 = Resize((h // upscale_factor, w // upscale_factor), interpolation=BICUBIC)
    r2 = Resize((h, w), interpolation=BICUBIC)
    lr_image_tensor = Compose([r1, ToTensor()])(image)
    lr_image_tensor = lr_image_tensor[:3, :, :].unsqueeze(0)
    mr_image_tensor = Compose([r1, r2, ToTensor()])(image)
    mr_image = mr_image_tensor[:3, :, :].unsqueeze(0)

    return mr_image, test(model_pth_dir, upscale_factor, lr_image_tensor), test(model_pth_dir, upscale_factor, image_tensor)


def copy_checkpoints():
    model_name = "srgan_imagenet"
    upscale_factor = 2
    for model_name in ["srgan_imagenet", "srresnet_imagenet"]:
        for upscale_factor in [2, 4, 8]:
            dir_path = f"/home/maksim/data/checkpoints/{model_name}_{upscale_factor}x"
            psnr, ssim = get_best_checkpoint(f"{dir_path}/metrics.csv")
            copyfile(
                f"{dir_path}/netG_epoch_{psnr:04}.pth",
                f"/home/maksim/dev_projects/atlas_sr/checkpoints/{model_name}_{upscale_factor}x_psnr_{psnr:04}.pth",
            )
            copyfile(
                f"{dir_path}/netG_epoch_{ssim:04}.pth",
                f"/home/maksim/dev_projects/atlas_sr/checkpoints/{model_name}_{upscale_factor}x_ssim_{ssim:04}.pth",
            )
            copyfile(
                f"{dir_path}/metrics.csv",
                f"/home/maksim/dev_projects/atlas_sr/checkpoints/{model_name}_{upscale_factor}x.csv",
            )


def save_tensor_as_img(ten, out_fp):
    out_img = ToPILImage()(ten[0].data.cpu())
    out_dir = os.path.split(out_fp)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_img.save(out_fp)


def upscale_img(img_fp, upscaled_dir, box, crop=False):
    if not os.path.exists(upscaled_dir):
        os.makedirs(upscaled_dir)
    for upscale in [2, 4, 8]:
        for model_name in ["srgan", "srresnet"]:
            hr_image = Image.open(img_fp)
            model_checkpoint_fp = glob.glob(
                f"/home/maksim/dev_projects/atlas_sr/checkpoints/{model_name}_imagenet_{upscale}x_psnr_*.pth"
            )[0]
            mr_image_tensor, sr_image_tensor, upscaled_tensor = test_best_checkpoints(
                model_checkpoint_fp, img_fp, upscale
            )

            sr_image_tensor = sr_image_tensor[:3, :, :].squeeze(0)
            mr_image_tensor = mr_image_tensor[:3, :, :].squeeze(0)
            upscaled_tensor = upscaled_tensor[:3, :, :].squeeze(0)
            hr_image_tensor = ToTensor()(hr_image)[:3, :, :]

            mr_psnr = psnr(hr_image_tensor.data.numpy(), mr_image_tensor.data.numpy())
            sr_psnr = psnr(hr_image_tensor.data.numpy(), sr_image_tensor.data.numpy())

            mr_image = ToPILImage()(mr_image_tensor.data.cpu())
            sr_image = ToPILImage()(sr_image_tensor.data.cpu())
            upscaled_image = ToPILImage()(upscaled_tensor.cpu())

            mr_image.save(upscaled_dir / f"full_bicubic_{upscale}x_{mr_psnr:.3f}.png")
            sr_image.save(
                upscaled_dir
                / f"full_{model_name}_imagenet_{upscale}x_{sr_psnr:.3f}.png"
            )
            hr_image.save(upscaled_dir / "full_hr.png")
            upscaled_image.save(upscaled_dir/f"{upscale}x.png")
            if crop:
                hr_image = hr_image.crop(box=box)
                sr_image = sr_image.crop(box=box)
                mr_image = mr_image.crop(box=box)

                mr_psnr = psnr(
                    ToTensor()(hr_image).data.numpy()[:3],
                    ToTensor()(mr_image).data.numpy(),
                )
                sr_psnr = psnr(
                    ToTensor()(hr_image).data.numpy()[:3],
                    ToTensor()(sr_image).data.numpy(),
                )

                hr_image.save(upscaled_dir / "cropped_hr.png")
                mr_image.save(
                    upscaled_dir / f"cropped_bicubic_{upscale}x_{mr_psnr:.3f}.png"
                )
                sr_image.save(
                    upscaled_dir
                    / f"cropped_{model_name}_imagenet_{upscale}x_{sr_psnr:.3f}.png"
                )


def plot_metrics_srgan(metrics_fp, title=""):
    df = pd.read_csv(metrics_fp)

    fig, ax1 = plt.subplots()
    epochs = range(len(df))

    color = "tab:red"
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("psnr", color=color)
    l1 = ax1.plot(epochs, df["psnr.val"].values, color=color, label="psnr")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("score")  # we already handled the x-label with ax1
    l2 = ax2.plot(epochs, df["g_score.avg"], color=color, label="g score")
    # ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:green"
    # ax3.set_ylabel('d_loss', color=color)  # we already handled the x-label with ax1
    l3 = ax3.plot(epochs, df["d_score.avg"], color=color, label="d score")
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc=0)
    ax3.get_yaxis().set_ticks([])
    ax3.set_title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_metrics_srresnet(metrics_fp, title=""):
    df = pd.read_csv(metrics_fp)

    fig, ax1 = plt.subplots()
    epochs = range(len(df))

    color = "tab:red"
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("psnr", color=color)
    l1 = ax1.plot(epochs, df["psnr.val"].values, color=color, label="psnr")
    ax1.tick_params(axis="y", labelcolor=color)

    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:green"
    ax3.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    l3 = ax3.plot(epochs, df["g_loss.avg"], color=color, label="g loss")
    lns = l1 + l3
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc=0)
    # ax3.get_yaxis().set_ticks([])
    ax3.tick_params(axis='y', labelcolor=color)

    ax3.set_title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == "__main__":
    box = (260, 255, 308, 277)
    img_fp = "/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01925_0002/330.tiff"
    upscaled_dir = Path("/home/maksim/data/DSIAC/upscaled/cegr01925_0002/")
    upscale_img(img_fp, upscaled_dir, box, crop=True)
