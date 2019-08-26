from typing import Optional

import numpy as np
import torch
from fastprogress import master_bar, progress_bar
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from data.databunch import DataBunch
from my_types import Loss
from train.pipeline import Pipeline, RunMode
from util import first_el


def main_train(
        *,
        model: nn.Module,
        data: DataBunch,
        loss_func: Loss,
        optimizer: Optimizer,
        pipeline_handler: Pipeline,
        epochs: int,
) -> None:
    assert (
            len(data.train_dl) != 0
    ), f"""Your training dataloader is empty, can't train a model.
        Use a smaller batch size (batch size={data.train_dl.batch_size} for {len(data.train_dl.dataset)} elements)."""

    pipeline_handler.on_train_begin(epochs=epochs)
    pbar = master_bar(range(epochs))

    exception = None
    try:
        for _epoch in pbar:
            model.train()

            pipeline_handler.on_epoch_begin()

            for xb, yb in progress_bar(data.train_dl, parent=pbar):
                pipeline_handler.on_batch_begin(x=xb, y=yb)
                loss = loss_batch(
                    model=model,
                    xb=xb,
                    yb=yb,
                    loss_func=loss_func,
                    opt=optimizer,
                    pipeline_handler=pipeline_handler,
                )
                pipeline_handler.on_batch_end(loss=loss)
                if pipeline_handler.stop_epoch:
                    break
                print(loss)
            if not pipeline_handler.skip_validate:
                val_loss = validate(
                    model=model,
                    dl=data.valid_dl,
                    loss_func=loss_func,
                    pipeline_handler=pipeline_handler,
                    pbar=pbar,
                )
            else:
                val_loss = None

            pipeline_handler.on_epoch_end(loss=val_loss)
            if pipeline_handler.stop_training:
                break

    except Exception as e:
        exception = e
        raise
    finally:
        pipeline_handler.on_train_end(exception=exception)


def loss_batch(
        *,
        model: nn.Module,
        xb: Tensor,
        yb: Tensor,
        pipeline_handler: Pipeline,
        loss_func: Loss,
        opt: Optional[Optimizer] = None,
) -> Tensor:
    out = model(xb)
    loss = loss_func(out, yb)
    cpu_loss = loss.detach().cpu()
    pipeline_handler.on_loss_begin(out=out)

    if opt is not None:
        pipeline_handler.on_backward_begin(loss=cpu_loss)

        if not pipeline_handler.skip_bwd_pass:
            loss.backward()

        pipeline_handler.on_backward_end()

        if not pipeline_handler.skip_step:
            opt.step()
        if not pipeline_handler.skip_zero_grad:
            opt.zero_grad()

        pipeline_handler.on_step_end()

    return cpu_loss


def validate(
        *,
        model: nn.Module,
        dl: DataLoader,
        pipeline_handler: Pipeline,
        loss_func: Optional[Loss] = None,
        pbar: Optional = None,
        n_batch: Optional[int] = None,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        val_losses, nums = [], []
        for xb, yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            pipeline_handler.on_batch_begin(x=xb, y=yb, run_mode=RunMode.TRAIN)

            val_loss = loss_batch(
                model=model,
                xb=xb,
                yb=yb,
                loss_func=loss_func,
                pipeline_handler=pipeline_handler,
            )
            nums.append(first_el(yb).shape[0])

            pipeline_handler.on_batch_end(validation_loss=val_loss)
            if pipeline_handler.stop_epoch:
                break

            if n_batch and (len(nums) >= n_batch):
                break
            val_losses.append(val_loss)
        nums = np.array(nums, dtype=np.float32)
        return (torch.stack(val_losses).data.cpu().numpy() * nums).sum() / nums.sum()

# def fit_one_cycle(
#     learn: Learner,
#     cyc_len: int,
#     max_lr: Union[Floats, slice] = DEFAULTS.learning_rate,
#     moms: Tuple[float, float] = (0.95, 0.85),
#     div_factor: float = 25.0,
#     pct_start: float = 0.3,
#     final_div: float = None,
#     wd: float = None,
#     callbacks: Optional[Callback] = None,
#     tot_epochs: int = None,
#     start_epoch: int = None,
# ) -> None:
#     "Fit a model following the 1cycle policy."
#     max_lr = learn.lr_range(max_lr)
#     callbacks = list(callbacks)
#     callbacks.append(
#         OneCycleScheduler(
#             learn,
#             max_lr,
#             moms=moms,
#             div_factor=div_factor,
#             pct_start=pct_start,
#             final_div=final_div,
#             tot_epochs=tot_epochs,
#             start_epoch=start_epoch,
#         )
#     )
#     learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)


# def lr_find(
#     *,
#     model: nn.Module,
#     optimizer: optim.Optimizer,
#     loss_func: LossFunction,
#     train_dataloader: DataLoader,
#     end_lr: float,
# ):
#     lr_finder = LRFinder(model, optimizer, loss_func, device="cuda")
#     lr_finder.range_test(train_dataloader, end_lr=end_lr, num_iter=100)
#     lr_finder.plot()
