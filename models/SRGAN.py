import math

import matplotlib.pyplot as plt
import torch
from PIL import Image
from apex.parallel import SyncBatchNorm
from torch import nn
from torch.nn import PixelShuffle
from torchvision.models import vgg16
from torchvision.transforms import ToTensor, Resize, Compose

from util.util import count_parameters


class Generator(nn.Module):
    def __init__(self, *, scale_factor, in_channels):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, scale_factor=2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, in_channels, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class SyncGenerator(nn.Module):
    def __init__(self, *, scale_factor, in_channels):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(SyncGenerator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.block2 = SyncResidualBlock(64)
        self.block3 = SyncResidualBlock(64)
        self.block4 = SyncResidualBlock(64)
        self.block5 = SyncResidualBlock(64)
        self.block6 = SyncResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), SyncBatchNorm(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, in_channels, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, *, in_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class SyncDiscriminator(nn.Module):
    def __init__(self, *, in_channels):
        super(SyncDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            SyncBatchNorm(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            SyncBatchNorm(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            SyncBatchNorm(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            SyncBatchNorm(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            SyncBatchNorm(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            SyncBatchNorm(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            SyncBatchNorm(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class DiscriminatorFatKernel(nn.Module):
    def __init__(self, *, in_channels):
        super(DiscriminatorFatKernel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, n_feat_maps):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feat_maps, n_feat_maps, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_feat_maps)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(n_feat_maps, n_feat_maps, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_feat_maps)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class SyncResidualBlock(nn.Module):
    def __init__(self, n_feat_maps):
        super(SyncResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feat_maps, n_feat_maps, kernel_size=3, padding=1)
        self.bn1 = SyncBatchNorm(n_feat_maps)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(n_feat_maps, n_feat_maps, kernel_size=3, padding=1)
        self.bn2 = SyncBatchNorm(n_feat_maps)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * scale_factor ** 2, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle((scale_factor))
        kernel = ICNR(self.conv.weight, upscale_factor=scale_factor)
        self.conv.weight.data.copy_(kernel)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


def icnr_mine(
    *,
    conv,
    in_channels,
    out_channels,
    kernel_size,
    r,
    initializer=nn.init.kaiming_normal_
):
    # make all of the colors of a "superpixel" the same
    subkernel = initializer(
        torch.zeros((out_channels, in_channels, kernel_size, kernel_size))
    )
    conv.weight.data.copy_(torch.zeros(conv.weight.shape))
    for n in range(0, r ** 2):
        wn_indices = [k * r ** 2 + n for k in range(0, out_channels)]
        conv.weight.data[wn_indices, :, :, :] = subkernel

    return conv


def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
    # https://arxiv.org/abs/1707.02937
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = inizializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(
            self.loss_network(out_images), self.loss_network(target_images)
        )
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return (
            image_loss
            + 0.001 * adversarial_loss
            + 0.006 * perception_loss
            + 2e-8 * tv_loss
        )


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def gray():
    img = Image.open(
        "/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00050000.JPEG"
    ).convert("L")
    img_tensor = ToTensor()(img).unsqueeze(0)
    two_x = Compose(
        [
            Resize(
                (2 * img_tensor.shape[2], 2 * img_tensor.shape[3]),
                interpolation=Image.BICUBIC,
            ),
            ToTensor(),
        ]
    )(img).unsqueeze(0)

    g = Generator(scale_factor=2, in_channels=1)
    d = Discriminator(in_channels=1)
    g_loss = GeneratorLoss()

    fake_img = g(img_tensor)
    real_out = d(img_tensor).mean()
    fake_out = d(fake_img).mean()
    loss = g_loss(
        fake_out,
        torch.cat([fake_img, fake_img, fake_img], dim=1),
        torch.cat([two_x, two_x, two_x], dim=1),
    )

    print(fake_img, fake_out, real_out, loss)


def upsample():
    upscale = 2
    channels = 1
    i = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
    upsample_block = UpsampleBLock(channels, upscale)
    print(i.shape, upsample_block(i).shape)


def count():
    m3 = Generator(scale_factor=2, in_channels=3)
    m1 = Generator(scale_factor=2, in_channels=1)
    d3 = Discriminator(in_channels=3)
    d1 = Discriminator(in_channels=1)
    dfat = DiscriminatorFatKernel(in_channels=1)
    print("3 channel generator", count_parameters(m3))
    print("1 channel generator", count_parameters(m1))
    print("3 channel discriminator", count_parameters(d3))
    print("1 channel discriminator", count_parameters(d1))
    print("4x4 kernel discriminator", count_parameters(dfat))


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


def icnr():
    torch.set_printoptions(linewidth=1000)
    ci = 1
    r = 2
    out_channels = 1
    co = out_channels * r ** 2
    kernel_size = 3
    # figure out the Wn groups
    with torch.no_grad():
        c = nn.Conv2d(
            in_channels=ci,
            out_channels=co,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        print("Wn")
        W = torch.empty(c.weight.shape)
        # for n in range(0, r ** 2):
        #     for k in range(0, co // r ** 2):
        #         j = k * r ** 2 + n
        #         W[j, :, :, :] = n
        n = 1
        for i in range(kernel_size):
            for j in range(kernel_size):
                W[:, :, i, j] = n
                n += 1
        c.weight.data.copy_(W)
        print(c.weight)
        t = torch.ones((1, 2, 4, 4))
        # tt = c(t)
        # print(tt.shape, tt)
        ps = PixelShuffle(r)
        # ttt = ps(tt)
        # print(ttt.shape)
        # print(ttt)
        c = nn.Conv2d(
            in_channels=2,
            out_channels=co,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        # print(c.weight.shape, c.weight)
        tst = icnr_init(c.weight)
        print(tst)
        c.weight.copy_(tst)
        tttt = c(t)
        # print(tttt)
        print(ps(tttt))
        c = nn.Conv2d(
            in_channels=2,
            out_channels=co,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        tttt = c(t)
        plt.matshow(ps(tttt).squeeze(0).squeeze(0))
        plt.show()
        icnr_mine(
            conv=c,
            in_channels=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            r=r,
        )
        tttt = c(t)
        print(ps(tttt))
        plt.matshow(ps(tttt).squeeze(0).squeeze(0))
        plt.show()


if __name__ == "__main__":
    # test_gray()
    # upsample()
    icnr()
