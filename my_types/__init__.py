from functools import partial
from numbers import Number
from typing import Union, NewType, Callable, Collection

from torch import Tensor, nn, optim

Rank0Tensor = NewType("OneEltTensor", Tensor)
Loss = Callable[[Tensor, Tensor], Rank0Tensor]
AnnealFunc = Callable[[Number, Number, float], Number]
ParamList = Collection[nn.Parameter]
TensorOrNumber = Union[Tensor, Number]

BatchNormTypes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
BiasTypes = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

NoWeightDecayTypes = BatchNormTypes + (nn.LayerNorm,)
AdamW = partial(optim.Adam, betas=(0.9, 0.99))
