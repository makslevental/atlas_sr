import os
from math import log10
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import BICUBIC
from dsiac.cegr.arf import ARF, mad_normalization
from dsiac.cegr.cegr import CEGR
from sewar import mse, psnr
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose
from models.SRGAN import Generator
from util.util import basename
from dsiac import cegr


def test(model_pth_fp, upscale_factor, image):
    model = Generator(scale_factor=upscale_factor).eval()
    state_dict = {}
    for k, v in torch.load(model_pth_fp).items():
        state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)

    with torch.no_grad():
        out = model(image)
    # out_img = ToPILImage()(out[0].data.cpu())

    return out
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


def test_best_checkpoints(
    model_pth_dir, image_fp, out_image_dir, model_name, upscale_factor
):
    metrics_fp = os.path.join(model_pth_dir, "metrics.csv")
    assert os.path.exists(metrics_fp)
    psnr_max, ssim_max = get_best_checkpoint(metrics_fp)
    psnr_max_fp = os.path.join(model_pth_dir, f"netG_epoch_{psnr_max:04}.pth")
    ssim_max_fp = os.path.join(model_pth_dir, f"netG_epoch_{ssim_max:04}.pth")

    image = Image.open(image_fp)
    image_tensor = ToTensor()(image)[:3, :, :].unsqueeze(0)
    w, h = image.size
    # lr_image = image.resize((w // upscale_factor, h // upscale_factor), BICUBIC)
    r1 = Resize((h // upscale_factor, w // upscale_factor), interpolation=BICUBIC)
    r2 = Resize((h, w), interpolation=BICUBIC)
    lr_image_tensor = Compose([r1, ToTensor()])(image)
    lr_image_tensor = lr_image_tensor[:3, :, :].unsqueeze(0)
    mr_image_tensor = Compose([r1, r2, ToTensor()])(image)
    mr_image = mr_image_tensor[:3, :, :].unsqueeze(0)

    psnr_img = test(psnr_max_fp, upscale_factor, lr_image_tensor)
    ssim_img = test(ssim_max_fp, upscale_factor, lr_image_tensor)
    print(psnr(image_tensor.data.numpy(), psnr_img.data.numpy()))
    print(psnr(image_tensor.data.numpy(), ssim_img.data.numpy()))
    print(psnr(image_tensor.data.numpy(), mr_image.data.numpy()))


if __name__ == "__main__":
    # for upscale_factor in [2, 4, 8]:
    #     model_name = "srresnet"
    #     model_pth = f"/home/maksim/data/checkpoints/{model_name}_imagenet_{upscale_factor}x/netG_epoch_0099.pth"""
    #     test(
    #         model_pth,
    #         upscale_factor=upscale_factor,
    #         image_fp="/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/450.tiff",
    #         out_image_dir="/home/maksim/data/DSIAC/dsiac_mad_tiffs_highres/cegr01923_0011/",
    #         model_name=model_name
    #     )
    # for upscale_factor in [2, 4]:
    #     model_name = "srgan"
    #     model_pth = f"/home/maksim/data/checkpoints/{model_name}_imagenet_{upscale_factor}x/netG_epoch_0099.pth"""
    #     test(
    #         model_pth,
    #         upscale_factor=upscale_factor,
    #         image_fp="/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/450.tiff",
    #         out_image_dir="/home/maksim/data/DSIAC/dsiac_mad_tiffs_highres/cegr01923_0011/",
    #         model_name=model_name
    #     )
    #
    # upscale_factor = 8
    # model_name = "srgan"
    # test(
    #     model_pth,
    #     upscale_factor=upscale_factor,
    #     image_fp="/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/450.tiff",
    #     out_image_dir="/home/maksim/data/DSIAC/dsiac_mad_tiffs_highres/cegr01923_0011/",
    #     model_name=model_name
    # )
    #
    # get_best_checkpoint(
    #     "/home/maksim/data/checkpoints/srresnet_imagenet_2x/metrics.csv"
    # )

    # upscale_factor = 2
    # model_name = "srgan"
    # model_dir = (
    #     f"/home/maksim/data/checkpoints/{model_name}_imagenet_{upscale_factor}x/"
    # )
    # test_best_checkpoints(
    #     model_dir,
    #     image_fp="/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/450.tiff",
    #     out_image_dir="/home/maksim/data/DSIAC/dsiac_mad_tiffs_highres/cegr01923_0011/",
    #     model_name=model_name,
    #     upscale_factor=upscale_factor,
    # )

    # cegr = CEGR(
    #     "/home/maksim/data/DSIAC/cegr/arf/cegr01923_0011.arf",
    #     "/home/maksim/data/DSIAC/cegr/agt/cegr01923_0011.agt",
    #     "/home/maksim/data/DSIAC/Metric/cegr01923_0011.bbox_met"
    # )
    # print(cegr.bboxes(450))
    # arf = ARF("/home/maksim/data/DSIAC/cegr/arf/cegr01923_0011.arf")
    #
    # arf_frame = arf.get_frame_mat(450)
    # clipped_and_normed = np.clip(mad_normalization(arf_frame), -20, 20)
    # print(clipped_and_normed)
    model_name = "srgan_imagenet"
    upscale_factor = 2
    for model_name in ["srgan_imagenet", "srresnet_imagenet"]:
        for upscale_factor in [2,4,8]:
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
            copyfile(f"{dir_path}/metrics.csv", f"/home/maksim/dev_projects/atlas_sr/checkpoints/{model_name}_{upscale_factor}x.csv")

