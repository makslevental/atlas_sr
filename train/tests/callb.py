import unittest

import numpy as np
import torch

from train.metrics import AverageMetric, mean_squared_error
from train.pipeline import Pipeline


class MyTestCase(unittest.TestCase):
    def test_cb(self):
        avg = AverageMetric(mean_squared_error)
        c = Pipeline([avg])
        c.on_epoch_begin()
        for batch in range(10000):
            t1 = torch.Tensor([np.random.uniform(0, 1)])
            t2 = torch.Tensor([np.random.uniform(1, 2)])
            c.on_batch_begin(x=t1, y=t2)
            c.on_loss_begin(out=t1)
            c.on_batch_end(loss=t2)
        c.on_epoch_end(validation_loss=torch.Tensor([0]))


if __name__ == "__main__":
    unittest.main()
