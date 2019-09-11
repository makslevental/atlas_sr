import os
from pathlib import Path

DATA_DIR = Path(os.path.expanduser("~/data"))

_DSIAC_DATA_DIR = DATA_DIR / "DSIAC"
DSIAC_ARFS_DIR = _DSIAC_DATA_DIR / "cegr/arf/"

DSIAC_AGTS_DIR = _DSIAC_DATA_DIR / "cegr/agt/"
DSIAC_BBOX_METS_DIR = _DSIAC_DATA_DIR / "Metric/"
DSIAC_JSONS_DIR = _DSIAC_DATA_DIR / "annotated-jsons/"


FRAMES = 1800


def num_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
