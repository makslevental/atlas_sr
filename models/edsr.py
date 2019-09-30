import math

import torch
import torch.nn as nn

from models.common import make_upsample_block


class MeanShift(nn.Conv2d):
    def __init__(
            self,
            *,
            rgb_range=255,
            rgb_mean=(0.4488, 0.4371, 0.4040),
            rgb_std=(1.0, 1.0, 1.0),
            sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, *, n_feats, kernel_size, res_scale=1):
        super(ResBlock, self).__init__()

        conv = lambda: nn.Conv2d(
            in_channels=n_feats,
            out_channels=n_feats,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
        )

        self.body = nn.Sequential(conv(), nn.ReLU(), conv())
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, scale, *, n_colors=3, n_feats=64, kernel_size=3, n_resblocks=16):
        super(EDSR, self).__init__()
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=n_colors,
                out_channels=n_feats,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
            )
        )

        m_body = [
            ResBlock(n_feats=n_feats, kernel_size=kernel_size)
            for _ in range(n_resblocks)
        ]
        m_body.append(
            nn.Conv2d(
                in_channels=n_feats,
                out_channels=n_feats,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
            )
        )
        self.body = nn.Sequential(*m_body)

        m_tail = []
        for _ in range(int(math.log2(scale))):
            conv, pixel_shuffle = make_upsample_block(n_feats)
            m_tail.extend([conv, pixel_shuffle])

        m_tail.append(
            nn.Conv2d(
                in_channels=n_feats,
                out_channels=n_colors,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
            )
        )
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
