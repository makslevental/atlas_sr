from dataclasses import dataclass

from torch import nn, optim

from data.databunch import DataBunch
from my_types import LossFunction


@dataclass
class Learner:
    model: nn.Module
    loss_func: LossFunction
    opt: optim.Optimizer
    data: DataBunch
