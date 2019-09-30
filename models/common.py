import torch
from torch import nn


def icnr(tensor, *, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
    """Fills the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        inizializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = icnr(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
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


class UpsampleBlock(nn.Module):
    """2x upsample"""

    def __init__(self, *, in_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 4,  # i.e. 2**2
            kernel_size=3,
            padding=1,  # i.e. kernel_size // 2
        )
        kernel = icnr(self.conv.weight, upscale_factor=2)
        self.conv.weight.data.copy_(kernel)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


def make_upsample_block(in_channels):
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=in_channels * 4,  # i.e. 2**2
        kernel_size=3,
        padding=1,  # i.e. kernel_size // 2
    )
    kernel = icnr(conv.weight, upscale_factor=2)
    conv.weight.data.copy_(kernel)
    pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
    return conv, pixel_shuffle
