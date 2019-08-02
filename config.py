import os
from types import SimpleNamespace

DSIAC_DATA_DIR = os.path.expanduser("~/data/DSIAC/DSIAC/ATR Database")
YUMA_DATA_DIR = os.path.expanduser("~/dev_projects/dsiac/trn_fcst_yuma0707")
FRAMES = 1800


def num_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


DEFAULTS = SimpleNamespace(
    cpus=num_cpus(),
    cmap='viridis',
    return_fig=False,
    silent=False,
    batch_size=256
)
