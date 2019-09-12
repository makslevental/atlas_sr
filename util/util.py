import io
import json
import os
import re
import sys
import zipfile
from collections import OrderedDict
from itertools import zip_longest
from pathlib import Path
from typing import NamedTuple, Any, List, Callable, Collection, Union, Optional

import numpy as np
import torch
import yaml
from git import Repo
from torch import Tensor, nn
from torch.distributed import all_reduce_multigpu, all_reduce
from torch.nn import ModuleList
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer

from models.SRGAN import Discriminator
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


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def save_model(
    file_path: Union[Path, str], model: nn.Module, opt: Optional[Optimizer] = None
):
    if rank_distrib():
        return  # don't save if slave proc
    if opt is not None:
        state = {"model": get_model(model).state_dict(), "optimizer": opt.state_dict()}
    else:
        state = get_model(model).state_dict()

    torch.save(state, f"{file_path}")


def remove_module_load(state_dict):
    """create new OrderedDict that does not contain `module.`"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    return new_state_dict


def get_model(model: nn.Module):
    "Return the model maybe wrapped inside `model`."
    return (
        model.module
        if isinstance(model, (DistributedDataParallel, nn.DataParallel))
        else model
    )


def load_model_state(model: nn.Module, fp: str):
    state = torch.load(fp, map_location='cpu')
    state = remove_module_load(state)
    model.load_state_dict(state)
    del state
    return model


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def snapshot(snapshot_dir):
    p = os.path.split(__file__)[0]
    while ".git" not in os.listdir(p):
        p = os.path.split(p)[0]

    repo = Repo(p)
    fps = sorted(
        [
            os.path.join(p, fp)
            for fp in repo.git.ls_files().split() + repo.untracked_files
        ]
    )

    zipf = zipfile.ZipFile(
        os.path.join(snapshot_dir, "snapshot.zip"), "w", zipfile.ZIP_DEFLATED
    )
    for fp in fps:
        zipf.write(fp)

    env = dict(os.environ)
    env["args"] = sys.argv
    zipf.writestr("env.txt", json.dumps(env, indent=4))

    zipf.close()


def clear_directory(dir_fp):
    for root, dirs, files in os.walk(dir_fp):
        for f in files:
            os.remove(os.path.join(root, f))


def monkey_patch_bn():
    # https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152
    # print(inspect.getsource(torch.nn.functional.batch_norm))
    def batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        if training:
            size = input.size()
            size_prods = size[0]
            for i in range(len(size) - 2):
                size_prods *= size[i + 2]
            if size_prods == 1:
                raise ValueError(
                    "Expected more than 1 value per channel when training, got input size {}".format(
                        size
                    )
                )

        return torch.batch_norm(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            False,
        )

    torch.nn.functional.batch_norm = batch_norm
    # print(inspect.getsource(torch.nn.functional.batch_norm))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def dict_to_yaml_str(j):
    s = io.StringIO()
    yaml.dump(j, s)
    s.seek(0)
    return s.read()


def adjust_learning_rate(optimizer: Optimizer, epoch, step, len_epoch, orig_lr):
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = orig_lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * (1.0 + step + epoch * len_epoch) / (5.0 * len_epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    netD = Discriminator()
    print(list(netD.parameters())[0])
    netD = load_model_state(
        netD,
        os.path.expanduser("~/data/checkpoints/srresnet_voc_2x/netD_epoch_0000.pth"),
    )
    print(list(netD.parameters())[0])
