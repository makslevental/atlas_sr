from torch import nn
from torch.utils.data import DataLoader


def validate(
        model: nn.Module,
        dl: DataLoader,
        loss_func,
        cb_handler: Optional[CallbackHandler] = None,
        pbar: Optional[PBar] = None,
        average=True,
        n_batch: Optional[int] = None,
) -> Iterator[Tuple[Union[Tensor, int], ...]]:
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.eval()
    with torch.no_grad():
        val_losses, nums = [], []
        if cb_handler:
            cb_handler.set_dl(dl)
        for xb, yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler:
                xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_loss = loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler)
            val_losses.append(val_loss)
            if not is_listy(yb):
                yb = [yb]
            nums.append(first_el(yb).shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]):
                break
            if n_batch and (len(nums) >= n_batch):
                break
        nums = np.array(nums, dtype=np.float32)
        if average:
            return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:
            return val_losses


def fit(
        epochs: int,
        learn: BasicLearner,
        callbacks: Optional[CallbackList] = None,
        metrics: OptMetrics = None,
) -> None:
    "Fit the `model` on `data` and learn using `loss_func` and `opt`."
    assert (
            len(learn.data.train_dl) != 0
    ), f"""Your training dataloader is empty, can't train a model.
        Use a smaller batch size (batch size={learn.data.train_dl.batch_size} for {len(learn.data.train_dl.dataset)} elements)."""
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception = False
    try:
        for epoch in pbar:
            learn.model.train()
            cb_handler.set_dl(learn.data.train_dl)
            cb_handler.on_epoch_begin()
            for xb, yb in progress_bar(learn.data.train_dl, parent=pbar):
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = loss_batch(
                    learn.model, xb, yb, learn.loss_func, learn.opt, cb_handler
                )
                if cb_handler.on_batch_end(loss):
                    break

            if not cb_handler.skip_validate and not learn.data.empty_val:
                val_loss = validate(
                    learn.model,
                    learn.data.valid_dl,
                    loss_func=learn.loss_func,
                    cb_handler=cb_handler,
                    pbar=pbar,
                )
            else:
                val_loss = None
            if cb_handler.on_epoch_end(val_loss):
                break
    except Exception as e:
        exception = e
        raise
    finally:
        cb_handler.on_train_end(exception)
