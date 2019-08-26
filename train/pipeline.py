import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, List, Optional

from torch import Tensor


class RunMode(Enum):
    TRAIN = auto()
    VALIDATE = auto()
    TEST = auto()


@dataclass(frozen=True)
class PipelineState:
    epoch: int = 0
    iteration: int = 0
    num_batch: int = 0
    last_input: Tensor = None
    last_output: Any = None
    last_target: Any = None
    last_loss: Tensor = None
    last_validation_loss: Tensor = None
    last_metrics: Tensor = None
    stop_training: bool = False
    stop_epoch: bool = False
    skip_validate: bool = False
    skip_step: bool = False
    skip_zero_grad: bool = False
    skip_bwd_pass: bool = False
    run_mode: RunMode = RunMode.TRAIN


class Callback(ABC):
    order = 0

    def on_train_begin(self, **kwargs) -> Any:
        pass

    def on_epoch_begin(self, **kwargs) -> Any:
        pass

    def on_batch_begin(self, **kwargs) -> Any:
        pass

    def on_loss_begin(self, **kwargs) -> Any:
        pass

    def on_backward_begin(self, **kwargs) -> Any:
        pass

    def on_backward_end(self, **kwargs) -> Any:
        pass

    def on_step_end(self, **kwargs) -> Any:
        pass

    def on_batch_end(self, **kwargs) -> Any:
        pass

    def on_epoch_end(self, **kwargs) -> Any:
        pass

    def on_train_end(self, **kwargs) -> Any:
        pass

    def set_to_epoch(self, **kwargs) -> Any:
        pass


class Pipeline:
    callbacks: List[Callback]
    metrics: List[Callback]
    __state: PipelineState

    def __init__(
        self,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[List[Callback]] = None,
    ):
        if metrics is None:
            metrics = []
        if callbacks is not None:
            self.callbacks = sorted(callbacks, key=lambda c: c.order)
        else:
            self.callbacks = []
        self.metrics = metrics
        self.__state = PipelineState()

    def _call_and_update(self, cb, cb_name, **kwargs) -> None:
        if hasattr(cb, f"on_{cb_name}"):
            cb_fn = getattr(cb, f"on_{cb_name}")

            cb_kwargs = {
                kw: kwarg
                for kw, kwarg in list(kwargs.items())
                + list(self.__state.__dict__.items())
                if kw in inspect.getfullargspec(cb_fn).kwonlyargs
            }

            new = getattr(cb, f"on_{cb_name}")(**cb_kwargs)
            if new is not None:
                for k, v in new.items():
                    if k not in self.__state.__dataclass_fields__:
                        raise Exception(
                            f"{k} isn't a valid key in the state of the pipeline."
                        )
                    else:
                        self.__state = replace(self.__state, **{k: v})

    def call_cb(self, cb_name, **kwargs) -> None:
        for met in self.metrics:
            self._call_and_update(met, cb_name, **kwargs)
        for cb in self.callbacks:
            self._call_and_update(cb, cb_name, **kwargs)

    def on_train_begin(self, **kwargs) -> None:
        self.__state = replace(self.__state, epoch=0)
        self.call_cb("train_begin", **kwargs)

    def on_epoch_begin(self, **kwargs) -> None:
        self.__state = replace(self.__state, num_batch=0, stop_training=False)
        self.call_cb("epoch_begin", **kwargs)

    def on_batch_begin(
        self, *, x: Tensor, y: Tensor, run_mode: RunMode = RunMode.TRAIN, **kwargs
    ):
        self.__state = replace(
            self.__state,
            last_input=x,
            last_target=y,
            run_mode=run_mode,
            stop_epoch=False,
            skip_step=False,
            skip_zero_grad=False,
            skip_bwd_pass=False,
        )

        self.call_cb("batch_begin", **kwargs)

    def on_loss_begin(self, *, out: Tensor, **kwargs):
        self.__state = replace(self.__state, last_output=out)
        self.call_cb("loss_begin", out=out, **kwargs)

    def on_backward_begin(self, *, loss: Tensor, **kwargs):
        self.__state = replace(self.__state, last_loss=loss)
        self.call_cb("backward_begin", loss=loss, **kwargs)

    def on_backward_end(self, **kwargs):
        self.call_cb("backward_end", **kwargs)

    def on_step_end(self, **kwargs):
        self.call_cb("step_end", **kwargs)

    def on_batch_end(
        self,
        *,
        loss: Optional[Tensor] = None,
        validation_loss: Optional[Tensor] = None,
        **kwargs,
    ):
        assert (
            loss is not None or validation_loss is not None
        ), "need to provide either loss or validation loss"
        losses = {}
        if loss is not None:
            self.__state = replace(self.__state, last_loss=loss)
            losses["loss"] = loss
        if validation_loss is not None:
            self.__state = replace(self.__state, last_validation_loss=validation_loss)
            losses["validation_loss"] = validation_loss
        self.call_cb("batch_end", **losses, **kwargs)
        if self.__state.run_mode == RunMode.TRAIN:
            self.__state = replace(
                self.__state,
                iteration=self.__state.iteration + 1,
                num_batch=self.__state.num_batch + 1,
            )

    def on_epoch_end(self, *, loss: Tensor, **kwargs):
        self.call_cb("epoch_end", loss=loss, **kwargs)
        self.__state = replace(
            self.__state, epoch=self.__state.epoch + 1, last_loss=loss
        )

    def on_train_end(self, *, exception: Optional[Exception] = None, **kwargs):
        self.call_cb("train_end", exception=exception, **kwargs)

    @property
    def skip_validate(self) -> bool:
        return self.__state.skip_validate

    @skip_validate.setter
    def skip_validate(self, v):
        self.__state = replace(self.__state, skip_validate=v)

    @property
    def stop_training(self) -> bool:
        return self.__state.stop_training

    @property
    def stop_epoch(self) -> bool:
        return self.__state.stop_epoch

    @property
    def skip_step(self) -> bool:
        return self.__state.skip_step

    @property
    def skip_zero_grad(self) -> bool:
        return self.__state.skip_zero_grad

    @property
    def skip_bwd_pass(self) -> bool:
        return self.__state.skip_bwd_pass
