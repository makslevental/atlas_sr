import argparse
import glob
import math
import os
from functools import reduce

import matplotlib
import numpy
import pandas
import torch
import torch.nn as nn
from dsiac.cegr.arf import mad_normalize, ARF, make_arf
from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from torch.autograd import Variable
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Lambda
from tqdm import tqdm

from data_utils.cegr import ARFDataset, torch_mad_normalize
from util.util import load_model_state, show_im, basename


class DenseBlock(torch.nn.Module):
    def __init__(
        self, input_size, output_size, bias=True, activation="relu", norm="batch"
    ):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            input_size, output_size, kernel_size, stride, padding, bias=bias
        )

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(
            input_size, output_size, kernel_size, stride, padding, bias=bias
        )

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation="prelu",
        norm="batch",
    ):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            num_filter, num_filter, kernel_size, stride, padding, bias=bias
        )
        self.conv2 = torch.nn.Conv2d(
            num_filter, num_filter, kernel_size, stride, padding, bias=bias
        )

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class UpBlock(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.up_conv2 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.up_conv3 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class UpBlockPix(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        scale=4,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale, num_filter)
        self.up_conv2 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.up_conv3 = Upsampler(scale, num_filter)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        num_stages=1,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(
            num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None
        )
        self.up_conv1 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.up_conv2 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.up_conv3 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlockPix(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        num_stages=1,
        scale=4,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(
            num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None
        )
        self.up_conv1 = Upsampler(scale, num_filter)
        self.up_conv2 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.up_conv3 = Upsampler(scale, num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.down_conv2 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.down_conv3 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DownBlockPix(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        scale=4,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        num_stages=1,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(
            num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None
        )
        self.down_conv1 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.down_conv2 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.down_conv3 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlockPix(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        num_stages=1,
        scale=4,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(
            num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None
        )
        self.down_conv1 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding, activation, norm=None
        )

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class PSBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        scale_factor,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation="prelu",
        norm="batch",
    ):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            input_size,
            output_size * scale_factor ** 2,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act="prelu", bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(
                ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None)
            )
            modules.append(torch.nn.PixelShuffle(2))
            if bn:
                modules.append(torch.nn.BatchNorm2d(n_feat))
            # modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)

        self.activation = act
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        upsample="deconv",
        activation="relu",
        norm="batch",
    ):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == "deconv":
            self.upsample = DeconvBlock(
                input_size,
                output_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=bias,
                activation=activation,
                norm=norm,
            )

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == "ps":
            self.upsample = PSBlock(
                input_size,
                output_size,
                scale_factor=scale_factor,
                bias=bias,
                activation=activation,
                norm=norm,
            )

        # 3. Resize and Convolution
        elif upsample == "rnc":
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode="nearest"),
                ConvBlock(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    activation=activation,
                    norm=norm,
                ),
            )

    def forward(self, x):
        out = self.upsample(x)
        return out


class DBPNITER(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(DBPNITER, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        self.num_stages = num_stages

        # Initial Feature Extraction
        self.feat0 = ConvBlock(
            num_channels, feat, 3, 1, 1, activation="prelu", norm=None
        )
        self.feat1 = ConvBlock(
            feat, base_filter, 1, 1, 0, activation="prelu", norm=None
        )
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        # Reconstruction
        self.output_conv = ConvBlock(
            num_stages * base_filter, num_channels, 3, 1, 1, activation=None, norm=None
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv2d") != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find("ConvTranspose2d") != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        l = self.feat1(x)

        results = []
        for i in range(self.num_stages):
            h1 = self.up1(l)
            l1 = self.down1(h1)
            h2 = self.up2(l1)

            concat_h = torch.cat((h2, h1), 1)
            l = self.down2(concat_h)

            concat_l = torch.cat((l, l1), 1)
            h = self.up3(concat_l)

            concat_h = torch.cat((h, concat_h), 1)
            l = self.down3(concat_h)

            concat_l = torch.cat((l, concat_l), 1)
            h = self.up4(concat_l)

            concat_h = torch.cat((h, concat_h), 1)
            l = self.down4(concat_h)

            concat_l = torch.cat((l, concat_l), 1)
            h = self.up5(concat_l)

            concat_h = torch.cat((h, concat_h), 1)
            l = self.down5(concat_h)

            concat_l = torch.cat((l, concat_l), 1)
            h = self.up6(concat_l)

            concat_h = torch.cat((h, concat_h), 1)
            l = self.down6(concat_h)

            concat_l = torch.cat((l, concat_l), 1)
            h = self.up7(concat_l)

            results.append(h)

        results = torch.cat(results, 1)
        x = self.output_conv(results)

        return x


def chop_forward(
    x, model, scale, shave=8, min_size=80000, n_gpus=1, self_ensemble=False
):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size) : w],
        x[:, :, (h - h_size) : h, 0:w_size],
        x[:, :, (h - h_size) : h, (w - w_size) : w],
    ]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, n_gpus):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i : (i + n_gpus)], dim=0)
            if self_ensemble:
                with torch.no_grad():
                    output_batch = x8_forward(input_batch, model)
            else:
                with torch.no_grad():
                    output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(n_gpus, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, n_gpus)
            for patch in inputlist
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))

    output[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = outputlist[1][
        :, :, 0:h_half, (w_size - w + w_half) : w_size
    ]
    output[:, :, h_half:h, 0:w_half] = outputlist[2][
        :, :, (h_size - h + h_half) : h_size, 0:w_half
    ]
    output[:, :, h_half:h, w_half:w] = outputlist[3][
        :, :, (h_size - h + h_half) : h_size, (w_size - w + w_half) : w_size
    ]

    return output


def x8_forward(img, model):
    def _transform(v, op):
        if op == "vflip":
            tfnp = v.flip(2).clone()
        elif op == "hflip":
            tfnp = v.flip(3).clone()
        elif op == "transpose":
            tfnp = v.permute((0, 1, 3, 2)).clone()

        return tfnp

    inputlist = [img]
    for tf in "vflip", "hflip", "transpose":
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], "transpose")
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], "hflip")
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], "vflip")

    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output


def linear_scale(x, vmin=None, vmax=None, rescale=None):
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    x -= vmin
    x /= vmax - vmin
    if rescale is not None:
        x *= rescale
    return x


def make_dataloader(arf_fp):
    transformed_dataset = ARFDataset(
        arf_fp,
        transform=transforms.Compose(
            [
                # Lambda(lambda x: torch_mad_normalize(x)),
                # Lambda(lambda x: torch.clamp(x, -20, 20)),
                Lambda(lambda x: linear_scale(x, vmin=0, vmax=2 ** 16 - 1)),
                Lambda(lambda x: torch.stack([x, x, x])),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        ),
    )
    _, h, w = transformed_dataset[0].size()
    dataloader = DataLoader(
        transformed_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    return dataloader, w, h


def main():
    df = pandas.read_csv("tgts.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--frame_rate", type=int, default=1)
    parser.add_argument("--upscale_factor", type=int, default=2)
    args = parser.parse_args()

    local_rank = args.local_rank
    upscale_factor = args.upscale_factor
    frame_rate = args.frame_rate

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(args.local_rank, world_size)
    with torch.cuda.device(local_rank):
        with torch.no_grad():
            model = DBPNITER(
                num_channels=3,
                base_filter=64,
                feat=256,
                num_stages=3,
                scale_factor=upscale_factor,
            )
            load_model_state(
                model,
                "/home/maksim/dev_projects/atlas_sr/checkpoints/DBPN-RES-MR64-3_2x.pth",
            )
            model = model.to("cuda")
            # print(next(model.parameters()).device)
            # for arf_fp in sorted(glob.glob("/home/maksim/data/DSIAC/cegr/arf/*.arf")):
            for scenario in tqdm(
                sorted(df["scenario"][args.local_rank :: world_size]),
                f"rank {local_rank} scenario",
            ):
                arf_fp = f"/home/maksim/data/DSIAC/cegr/arf/{scenario}.arf"
                new_arf_fp = f"/media/maksim/3125372135FE0CCE/dbpn/{basename(arf_fp)[0]}_{upscale_factor}x.arf"
                if os.path.exists(new_arf_fp):
                    continue
                print(new_arf_fp)
                dataloader, w, h = make_dataloader(arf_fp)
                n_frames = len(dataloader) // frame_rate
                if n_frames == 0:
                    continue
                print(n_frames)
                new_arf = make_arf(
                    new_arf_fp,
                    height=upscale_factor * h,
                    width=upscale_factor * w,
                    n_frames=n_frames,
                )
                for i_batch, frame in tqdm(
                    enumerate(dataloader), f"rank {local_rank} frame"
                ):
                    if i_batch % frame_rate:
                        continue
                    print(i_batch)
                    frame = frame.to("cuda")
                    bicubic = interpolate(
                        frame,
                        scale_factor=(upscale_factor, upscale_factor),
                        mode="bicubic",
                        align_corners=True,
                    )
                    out = model(frame)
                    # out = chop_forward(frame, model, upscale_factor, self_ensemble=True)
                    # out = x8_forward(frame, model)
                    # sr = denorm(out + bicubic)
                    sr = out + bicubic

                    # show_im(frame.squeeze(0).mean(dim=0).cpu().numpy())
                    # show_im(bicubic.squeeze(0).mean(dim=0).cpu().numpy())
                    # show_im(out.squeeze(0).mean(dim=0).cpu().numpy())
                    # show_im(sr.squeeze(0).mean(dim=0).cpu().numpy())
                    sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
                    # show_im(sr)
                    new_arf[i_batch // frame_rate] = (
                        (sr * 2 ** 16 - 1).astype(numpy.uint16).byteswap()
                    )
                    new_arf.flush()
                break

if __name__ == "__main__":
    main()
