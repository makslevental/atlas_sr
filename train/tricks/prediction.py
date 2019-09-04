# https://github.com/sanghoon/prediction_gan
import copy
from contextlib import contextmanager

import torch
from torch.optim.optimizer import Optimizer


class PredOpt(Optimizer):
    def __init__(self, params):
        super(Optimizer, self).__init__()

        self._params = list(params)
        self._prev_params = copy.deepcopy(self._params)
        self._diff_params = None

        self.step()

    def step(self):
        if self._diff_params is None:
            # Preserve parameter memory
            self._diff_params = copy.deepcopy(self._params)

        for i, _new_param in enumerate(self._params):
            # Calculate difference and store new params
            self._diff_params[i].data[:] = _new_param.data[:] - self._prev_params[i].data[:]
            self._prev_params[i].data[:] = _new_param.data[:]

    @contextmanager
    def lookahead(self, step=1.0):
        # Do nothing if lookahead stepsize is 0.0
        if step == 0.0:
            yield
            return

        for i, cur_param in enumerate(self._params):
            # Integrity check (whether we have the latest copy of parameters)
            if torch.sum(cur_param.data[:] != self._prev_params[i].data[:]) > 0:
                raise RuntimeWarning("Stored parameters differ from the current ones. Call step() after parameter updates")

            cur_param.data[:] += step * self._diff_params[i].data[:]

        yield

        # Roll-back to the original values
        for i, cur_param in enumerate(self._params):
            cur_param.data[:] = self._prev_params[i].data[:]
