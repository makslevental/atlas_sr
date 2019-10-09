import numpy as np
import torch.nn as nn
from PIL import Image
from PIL.Image import BICUBIC
from matplotlib import pyplot
from sewar import psnr
from torchvision.transforms import ToTensor, Resize

from models.common import default_conv, MeanShift, ResBlock, EDSRUpsampler
from util.util import load_model_state


class EDSR(nn.Module):
    def __init__(
        self,
        scale,
        n_resblocks,
        n_feats,
        n_colors=3,
        res_scale=1,
        rgb_range=255,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1, 1, 1),
    ):
        super(EDSR, self).__init__()
        kernel_size = 3
        self.sub_mean = MeanShift(
            sign=-1, rgb_range=rgb_range, rgb_mean=rgb_mean, rgb_std=rgb_std
        )
        self.add_mean = MeanShift(
            sign=1, rgb_range=rgb_range, rgb_mean=rgb_mean, rgb_std=rgb_std
        )

        # define head module
        m_head = [default_conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            EDSRUpsampler(scale, n_feats),
            default_conv(n_feats, n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


if __name__ == "__main__":
    upscale_factor = 2
    n_resblocks = 32
    n_feats = 256
    e = EDSR(upscale_factor, n_resblocks, n_feats, res_scale=0.1)
    load_model_state(e, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt")
    image = Image.open(
        "/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/420.tiff"
    )
    image_np = np.asarray(image)
    pyplot.imshow(np.asarray(image))
    pyplot.show()

    w, h = image.size
    r1 = Resize((h // upscale_factor, w // upscale_factor), interpolation=BICUBIC)
    image = r1(image)
    image_tensor = ToTensor()(image)[:3]
    sr = e(image_tensor.unsqueeze(0)).squeeze(0).permute((1, 2, 0)).data.numpy() / 255

    pyplot.imshow(sr)
    pyplot.show()
    print(psnr(image_np[:, :, :3], sr))
