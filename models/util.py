import gc
from collections import OrderedDict
from pathlib import Path
from typing import Union, Optional

import torch
from apex.parallel import DistributedDataParallel
from torch import nn, optim

import defaults as DEFAULTS
from util.util import rank_distrib


def save_model(
        file_path: Union[Path, str], model: nn.Module, opt: Optional[optim.Optimizer] = None
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
        new_state_dict[k[7:]] = v
    return new_state_dict


def get_model(model: nn.Module):
    "Return the model maybe wrapped inside `model`."
    return (
        model.module
        if isinstance(model, (DistributedDataParallel, nn.DataParallel))
        else model
    )


def load_model_state(
        *,
        model: nn.Module,
        file_path: Union[Path, str],
        device: torch.device = DEFAULTS.device,
        opt: Optional[optim.Optimizer] = None,
        strict: bool = True,
        remove_module: bool = False,
):
    source = f"{file_path}"
    state = torch.load(source, map_location=device)
    model_state = state["model"]
    if remove_module:
        model_state = remove_module_load(model_state)
    get_model(model).load_state_dict(model_state, strict=strict)
    if opt is not None:
        opt.load_state_dict(state["optimizer"])
    del state
    gc.collect()
