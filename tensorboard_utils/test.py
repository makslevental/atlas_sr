import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import numpy as np

TENSORBOARD_DIR = Path(os.path.expanduser("~/data/tensorboard"))

writer = SummaryWriter(TENSORBOARD_DIR / "test")

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random())
    writer.add_scalar('Loss/test', np.random.random())
    writer.add_scalar('Accuracy/train', np.random.random())
    writer.add_scalar('Accuracy/test', np.random.random())