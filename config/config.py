import os
from pathlib import Path

_DSIAC_DATA_DIR = Path(os.path.expanduser("~/data/DSIAC/DSIAC/ATR Database"))
DSIAC_ARFS_DIR = _DSIAC_DATA_DIR / "cegr/arf/"

DSIAC_AGTS_DIR = _DSIAC_DATA_DIR / "cegr/agt/"
DSIAC_BBOX_METS_DIR = _DSIAC_DATA_DIR / "Metric/"
DSIAC_JSONS_DIR = _DSIAC_DATA_DIR / "annotated-jsons/"

YUMA_DATA_DIR = os.path.expanduser("~/dev_projects/dsiac/trn_fcst_yuma0707")

FRAMES = 1800


def num_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
