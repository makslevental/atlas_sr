from typing import Optional, Tuple, Union, List, Iterator

import numpy as np
import torch
from fastprogress import master_bar, progress_bar
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader

from callbacks.callbacks import Callback
from train.learner import Learner
from train.pipeline import Pipeline
from my_types import LossFunction
from util import first_el


def loss_batch(
    model: nn.Module,
    xb: List[Tensor],
    yb: List[Tensor],
    loss_func: LossFunction = None,
    opt: Optional[optim.Optimizer] = None,
    cb_handler: Optional[Pipeline] = None,
) -> Union[Tensor, int, float, str]:
    out = model(*xb)
    if cb_handler is not None:
        out = cb_handler.on_loss_begin(out=out)

    loss = loss_func(out, *yb)

    if opt is not None:
        loss, skip_bwd = cb_handler.on_backward_begin(loss=loss)
        if not skip_bwd:
            loss.backward()
        if not cb_handler.on_backward_end():
            opt.step()
        if not cb_handler.on_step_end():
            opt.zero_grad()

    return loss.detach().cpu()


def validate(
    model: nn.Module,
    dl: DataLoader,
    loss_func: Optional[LossFunction] = None,
    pipeline_handler: Optional[Pipeline] = None,
    pbar: Optional = None,
    average=True,
    n_batch: Optional[int] = None,
) -> Iterator[Tuple[Union[Tensor, int], ...]]:
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.eval()
    with torch.no_grad():
        val_losses, nums = [], []
        if pipeline_handler:
            pipeline_handler.set_dl(dl)
        for xb, yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if pipeline_handler:
                xb, yb = pipeline_handler.on_batch_begin(xb, yb, train=False)
            val_loss = loss_batch(model, xb, yb, loss_func, cb_handler=pipeline_handler)
            val_losses.append(val_loss)
            nums.append(first_el(yb).shape[0])
            if pipeline_handler and pipeline_handler.on_batch_end(val_losses[-1]):
                break
            if n_batch and (len(nums) >= n_batch):
                break
        nums = np.array(nums, dtype=np.float32)
        if average:
            return (
                torch.stack(val_losses).data.cpu().numpy() * nums
            ).sum() / nums.sum()
        else:
            return val_losses


def train(
    epochs: int,
    learn: Learner,
    callbacks: Optional[Callback] = None,
    metrics: Optional[Callback] = None,
) -> None:
    assert (
        len(learn.data.train_dl) != 0
    ), f"""Your training dataloader is empty, can't train a model.
        Use a smaller batch size (batch size={learn.data.train_dl.batch_size} for {len(learn.data.train_dl.dataset)} elements)."""

    pipeline_handler = Pipeline(callbacks, metrics)
    pipeline_handler.on_train_begin(epochs=epochs)
    pbar = master_bar(range(epochs))

    exception = False
    try:
        for _epoch in pbar:
            learn.model.train()
            pipeline_handler.on_epoch_begin()
            for xb, yb in progress_bar(learn.data.train_dl, parent=pbar):
                xb, yb = pipeline_handler.on_batch_begin(x=xb, y=yb)
                loss = loss_batch(
                    learn.model, xb, yb, learn.loss_func, learn.opt, pipeline_handler
                )
                if pipeline_handler.on_batch_end(loss=loss):
                    break

            if not pipeline_handler.skip_validate and not learn.data.empty_val:
                val_loss = validate(
                    learn.model,
                    learn.data.valid_dl,
                    loss_func=learn.loss_func,
                    pipeline_handler=pipeline_handler,
                    pbar=pbar,
                )
            else:
                val_loss = None
            if pipeline_handler.on_epoch_end(validation_loss=val_loss):
                break
    except Exception as e:
        exception = e
        raise
    finally:
        pipeline_handler.on_train_end(exception)
