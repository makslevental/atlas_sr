import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Any, List, Callable, Collection, Union

import numpy as np
from torch import Tensor, nn
from torch.nn import ModuleList

from config import YUMA_DATA_DIR, DSIAC_DATA_DIR
from my_types import NoWeightDecayTypes, ParamList, BiasTypes

old_filter = filter
filter = lambda x, y: list(old_filter(x, y))
old_zip = zip
zip = lambda x, y: list(old_zip(x, y))
old_map = map
map = lambda x, y: list(old_map(x, y))


class FilePaths(NamedTuple):
    arf_path: str
    agt_path: str
    bbox_met_path: str
    covar_json_path: str


def make_yuma_paths(fp=None, name=None):
    if name is None and fp is not None:
        name, _ext = os.path.splitext(fp)
    elif fp is not None and name is None:
        pass
    else:
        raise Exception("must supply either basename of fp")

    return FilePaths(
        f"{YUMA_DATA_DIR}/avco/arf/{name}.arf",
        f"{YUMA_DATA_DIR}/avco/agt/{name}.agt",
        f"{YUMA_DATA_DIR}/avco/metric/{name}.bbox_met",
        "",
    )


def make_dsiac_paths(fp=None, name=None):
    if name is None and fp is not None:
        name, _ext = os.path.splitext(fp)
    elif fp is not None and name is None:
        pass
    else:
        raise Exception("must supply either basename of fp")

    return FilePaths(
        f"{DSIAC_DATA_DIR}/cegr/arf/{name}.arf",
        f"{DSIAC_DATA_DIR}/cegr/agt/{name}.agt",
        f"{DSIAC_DATA_DIR}/Metric/{name}.bbox_met",
        f"{DSIAC_DATA_DIR}/annotated-jsons/{name}.json",
    )


def basename(fp):
    _, fn = os.path.split(fp)
    base_name, _ext = os.path.splitext(fn)
    return base_name


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def join_and_mkdir(*args):
    p = os.path.join(*args)
    if not os.path.exists(p):
        os.makedirs(p)
    return p


def func_args(func) -> bool:
    "Return the arguments of `func`."
    code = func.__code__
    return code.co_varnames[: code.co_argcount]


def num_distrib():
    "Return the number of processes in distributed training (if applicable)."
    return int(os.environ.get("WORLD_SIZE", 0))


def rank_distrib():
    "Return the distributed rank of this process (if applicable)."
    return int(os.environ.get("RANK", 0))


def flatten_check(out: Tensor, targ: Tensor) -> Tensor:
    "Check that `out` and `targ` have the same number of elements and flatten them."
    out, targ = out.contiguous().view(-1), targ.contiguous().view(-1)
    assert len(out) == len(
        targ
    ), f"Expected output and target to have the same number of elements but got {len(out)} and {len(targ)}."
    return out, targ


def is_listy(x: Any) -> bool:
    return isinstance(x, (tuple, list))


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))


def first_el(x: Any) -> Any:
    "Recursively get the first element of `x`."
    if is_listy(x):
        return first_el(x[0])
    if is_dict(x):
        return first_el(x[list(x.keys())[0]])
    return x


def even_multiples(start: float, stop: float, n: int) -> np.ndarray:
    "Build log-stepped array from `start` to `stop` in `n` steps."
    mult = stop / start
    step = mult ** (1 / (n - 1))
    return np.array([start * (step ** i) for i in range(n)])


def trainable_params(m: nn.Module) -> ParamList:
    "Return list of trainable params in `m`."
    return list(filter(lambda p: p.requires_grad, m.parameters()))


def uniqueify(x: Collection, sort: bool = False) -> List:
    "Return sorted unique values of `x`."
    res = list(OrderedDict.fromkeys(x).keys())
    if sort:
        res.sort()
    return res


def split_no_weight_decay_params(layer_groups: ModuleList) -> List[ParamList]:
    "Separate the parameters in `layer_groups` between `no_wd_types` and  bias (`bias_types`) from the rest."
    split_params = []
    for layer in layer_groups:
        l1, l2 = [], []
        for child in layer.children():
            if isinstance(child, NoWeightDecayTypes):
                l2 += list(trainable_params(child))
            elif isinstance(child, BiasTypes):
                bias = child.bias if hasattr(child, "bias") else None
                l1 += [p for p in trainable_params(child) if not (p is bias)]
                if bias is not None:
                    l2.append(bias)
            else:
                l1 += list(trainable_params(child))
        # Since we scan the children separately, we might get duplicates (tied weights). We need to preserve the order
        # for the optimizer load of state_dict
        l1, l2 = uniqueify(l1), uniqueify(l2)
        split_params += [l1, l2]
    return split_params


def is_pool_type(l: Callable):
    return re.search(r"Pool[123]d$", l.__class__.__name__)


def lr_range(lr: Union[float, slice], layer_groups: ModuleList) -> np.ndarray:
    "Build differential learning rates from `lr`."
    if not isinstance(lr, slice):
        return lr
    if lr.start:
        res = even_multiples(lr.start, lr.stop, len(layer_groups))
    else:
        res = [lr.stop / 10] * (len(layer_groups) - 1) + [lr.stop]
    return np.array(res)
