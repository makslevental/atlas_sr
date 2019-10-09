import glob

import numpy
import torch
from PIL import Image
from PIL.Image import BICUBIC
from dsiac.cegr.arf import ARF, make_arf
from matplotlib import cm
from torchvision.transforms import Resize, ToTensor

from models.dpnn import DBPNITER
from models.edsr import EDSR
from util.util import load_model_state, show_mat

if __name__ == "__main__":
    with torch.no_grad():
        upscale_factor = 2
        dpnn = DBPNITER(
            num_channels=3,
            base_filter=64,
            feat=256,
            num_stages=3,
            scale_factor=upscale_factor,
        )
        dpnn = dpnn.to("cuda")


        # hr = Image.open("/home/maksim/dev_projects/atlas_sr/bb_test.tiff")
        # image_np = numpy.asarray(hr)
        #
        # w, h = image_np.size
        # r1 = Resize((h // upscale_factor, w // upscale_factor), interpolation=BICUBIC)
        # r2 = Resize((h, w), interpolation=BICUBIC)
        # r2 = Resize((upscale_factor * h, upscale_factor * w), interpolation=BICUBIC)
        #
        # bicubic = numpy.asarray(r2(hr))[:, :, :3] / 255
        # image_np = bicubic
        # image_tensor = ToTensor()(hr)[:3]
        #
        # sr1 = (
        #     dpnn(image_tensor.unsqueeze(0)).squeeze(0).permute((1, 2, 0)).data.numpy()
        # ) + bicubic
        #
        # sr2 = (
        #     e((image_tensor * 255).unsqueeze(0)).squeeze(0).permute((1, 2, 0)).data.numpy()
        #     / 255
        # )
        #
        # fig, axs = pyplot.subplots(
        #     nrows=2, ncols=2, constrained_layout=True, figsize=(10, 10)
        # )
        # for i in range(2):
        #     for j in range(2):
        #         axs[i, j].set_xticks([])
        #         axs[i, j].set_yticks([])
        #         axs[i, j].grid(False)
        #
        # axs[0, 0].imshow(image_np)
        # axs[0, 0].set_title("hr")
        # axs[0, 1].imshow(bicubic)
        # axs[0, 1].set_title(
        #     f"bicubic psnr {psnr(image_np[:, :, :1], bicubic[:, :, :1]):.5f}"
        # )
        # axs[1, 0].imshow(sr1)
        # axs[1, 0].set_title(f"dpnn psnr {psnr(image_np[:, :, :1], sr1[:, :, :1]):.5f}")
        # axs[1, 1].imshow(sr2)
        # axs[1, 1].set_title(f"edsr psnr {psnr(image_np[:, :, :1], sr2[:, :, :1]):.5f}")
        # pyplot.show()
        a = ARF("/home/maksim/data/DSIAC/cegr/arf/cegr01927_0001.arf")
        # for i in range(1800):
        #     print(i)
        #     a.save_frame(i, f"/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr02009_0011/{i}.tiff")
        r1 = Resize(
            (a.height // upscale_factor, a.width // upscale_factor),
            interpolation=BICUBIC,
        )
        r2 = Resize((a.height, a.width), interpolation=BICUBIC)
        r3 = Resize(
            (upscale_factor * a.height, upscale_factor * a.width), interpolation=BICUBIC
        )
        tiffs = glob.glob(
            "/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr02009_0011/*.tiff"
        )
        new_arf = make_arf(
            "/home/maksim/dev_projects/atlas_sr/cegr02009_0011_2x_all_dpnn.arf",
            upscale_factor * a.height,
            upscale_factor * a.width,
            len(tiffs),
        )

        for i, fp in enumerate(sorted(tiffs)):
            print(i)
            hr = Image.open(fp)
            image_tensor = ToTensor()(hr)[:3].to("cuda")
            bicubic = numpy.asarray(r2(hr))[:, :, :3] / 255
            sr1 = (
                      dpnn(image_tensor.unsqueeze(0)).squeeze(0).permute((1, 2, 0)).data.numpy()
                  ) + bicubic[:, :, :3]

            # sr2 = (
            #     e((image_tensor * 255).unsqueeze(0))
            #     .cpu()
            #     .squeeze(0)
            #     .permute((1, 2, 0))
            #     .data.numpy()
            # )
            new_arf[i] = sr1.mean(axis=2)
            new_arf.flush()
