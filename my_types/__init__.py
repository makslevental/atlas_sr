from numbers import Number
from typing import Union, NewType, Callable

from torch import Tensor

TensorOrNumber = Union[Tensor, Number]
Rank0Tensor = NewType('OneEltTensor', Tensor)
LossFunction = Callable[[Tensor, Tensor], Rank0Tensor]


