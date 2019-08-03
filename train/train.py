from typing import Optional, List

import numpy as np
import torch
from fastprogress import master_bar, progress_bar
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader

from my_types import LossFunction
from train.learner import Learner
from train.pipeline import Pipeline, RunMode, Callback
from util import first_el


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

    exception = None
    try:
        for _epoch in pbar:
            learn.model.train()

            pipeline_handler.on_epoch_begin()

            for xb, yb in progress_bar(learn.data.train_dl, parent=pbar):
                xb, yb = pipeline_handler.on_batch_begin(x=xb, y=yb)
                loss = loss_batch(
                    model=learn.model,
                    xb=xb,
                    yb=yb,
                    loss_func=learn.loss_func,
                    opt=learn.opt,
                    pipeline_handler=pipeline_handler,
                )

                pipeline_handler.on_batch_end(loss=loss)
                if pipeline_handler.state.stop_epoch:
                    break

            if not pipeline_handler.skip_validate:
                val_loss = validate(
                    model=learn.model,
                    dl=learn.data.valid_dl,
                    loss_func=learn.loss_func,
                    pipeline_handler=pipeline_handler,
                    pbar=pbar,
                )
            else:
                val_loss = None

            pipeline_handler.on_epoch_end(validation_loss=val_loss)
            if pipeline_handler.state.stop_training:
                break

    except Exception as e:
        exception = e
        raise
    finally:
        pipeline_handler.on_train_end(exception=exception)


def loss_batch(
    *,
    model: nn.Module,
    xb: List[Tensor],
    yb: List[Tensor],
    pipeline_handler: Pipeline,
    loss_func: LossFunction = None,
    opt: Optional[optim.Optimizer] = None,
) -> Tensor:
    out = model(*xb)
    loss = loss_func(out, *yb)

    pipeline_handler.on_loss_begin(out=out)

    if opt is not None:
        pipeline_handler.on_backward_begin(loss=loss)
        if not pipeline_handler.state.skip_bwd_pass:
            loss.backward()

        pipeline_handler.on_backward_end()
        if not pipeline_handler.state.skip_step:
            opt.step()

        pipeline_handler.on_step_end()
        if not pipeline_handler.state.skip_zero_grad:
            opt.zero_grad()

    return loss.detach().cpu()


def validate(
    *,
    model: nn.Module,
    dl: DataLoader,
    pipeline_handler: Pipeline,
    loss_func: Optional[LossFunction] = None,
    pbar: Optional = None,
    n_batch: Optional[int] = None,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        val_losses, nums = [], []
        for xb, yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            pipeline_handler.on_batch_begin(x=xb, y=yb, ru_mode=RunMode.TRAIN)

            val_loss = loss_batch(
                model=model,
                xb=xb,
                yb=yb,
                loss_func=loss_func,
                pipeline_handler=pipeline_handler,
            )
            nums.append(first_el(yb).shape[0])

            pipeline_handler.on_batch_end(loss=val_loss)
            if pipeline_handler.state.stop_epoch:
                break

            if n_batch and (len(nums) >= n_batch):
                break
            val_losses.append(val_loss)
        nums = np.array(nums, dtype=np.float32)
        return (torch.stack(val_losses).data.cpu().numpy() * nums).sum() / nums.sum()
