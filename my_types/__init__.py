from numbers import Number
from typing import Union, NewType

from torch import Tensor

TensorOrNumber = Union[Tensor, Number]
Rank0Tensor = NewType('OneEltTensor', Tensor)

