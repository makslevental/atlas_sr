from typing import Callable, Any

import torch.nn.functional as F
from torch import Tensor, dist

from callbacks.classes import Callback
from train.pipeline import PipelineState
from my_types import Rank0Tensor
from util import flatten_check, is_listy, first_el
from util import num_distrib


class AverageMetric(Callback):
    val: float = 0.0
    world: int = 0
    count: int = 0
    func: Callable[[Any], float]
    name: str = ""

    def __init__(self, func):
        name = func.__name__ if hasattr(func, "__name__") else func.func.__name__
        self.func, self.name = func, name
        self.world = num_distrib()

    def on_epoch_begin(self, **kwargs):
        self.val, self.count = 0.0, 0

    def on_batch_end(self, cb_handler_state: PipelineState, **kwargs):
        last_target = cb_handler_state.last_target
        last_output = cb_handler_state.last_output
        if not is_listy(last_target):
            last_target = [last_target]
        self.count += first_el(last_target).size(0)
        val = self.func(last_output, *last_target)
        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += first_el(last_target).size(0) * val.detach().cpu()

    def on_epoch_end(self, cb_handler_state: PipelineState, **kwargs):
        print(self.val / self.count)


def mean_squared_error(pred: Tensor, targ: Tensor) -> Rank0Tensor:
    pred, targ = flatten_check(pred, targ)
    return F.mse_loss(pred, targ)
