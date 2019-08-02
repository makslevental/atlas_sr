from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Union, List, Tuple

from torch import Tensor


class TrainOrValidate(Enum):
    TRAIN = auto()
    VALIDATE = auto()


@dataclass(frozen=True)
class CallbackHandlerState:
    epoch: int = 0
    iteration: int = 0
    num_batch: int = 0
    last_input: Tensor = None
    last_output: Any = None
    last_target: Any = None
    last_loss: Tensor = None
    last_metrics: Tensor = None
    stop_training: bool = False
    stop_epoch: bool = False
    skip_validate: bool = False
    skip_step: bool = False
    skip_zero_grad: bool = False
    skip_bwd_pass: bool = False
    train_or_validate: TrainOrValidate = TrainOrValidate.TRAIN


class Callback(ABC):
    order = 0

    @abstractmethod
    def __init__(self):
        pass

    def on_train_begin(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_epoch_begin(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_batch_begin(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_loss_begin(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_backward_begin(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_backward_end(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_step_end(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_batch_end(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_epoch_end(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def on_train_end(self, cb_handler_state: CallbackHandlerState, **kwargs):
        pass

    def set_to_epoch(self, epoch):
        pass


class CallbackHandler:
    cbs: List[Callback]
    metrics: List[Callback]
    state: CallbackHandlerState

    def __init__(self, callbacks: List[Callback], metrics: List[Callback] = None):
        if metrics is None:
            metrics = []
        self.cbs = sorted(callbacks, key=lambda c: c.order)
        self.metrics = metrics
        self.state = CallbackHandlerState()

    def _call_and_update(self, cb, cb_name, **kwargs) -> None:
        if hasattr(cb, f"on_{cb_name}"):
            new = getattr(cb, f"on_{cb_name}")(cb_handler_state=self.state, **kwargs)
            if new is not None:
                for k, v in new.items():
                    if k not in self.state.__dataclass_fields__:
                        raise Exception(
                            f"{k} isn't a valid key in the state of the callbacks."
                        )
                    else:
                        setattr(self.state, k, v)

    def __call__(self, cb_name, call_mets=True, **kwargs) -> None:
        for met in self.metrics:
            self._call_and_update(met, cb_name, **kwargs)
        for cb in self.cbs:
            self._call_and_update(cb, cb_name, **kwargs)

    def on_train_begin(self, **kwargs) -> None:
        self("train_begin", **kwargs)

    def on_epoch_begin(self, **kwargs) -> None:
        self.state = replace(self.state, num_batch=0, stop_training=False)
        self("epoch_begin", **kwargs)

    def on_batch_begin(
            self,
            *,
            x: Tensor,
            y: Tensor,
            train: TrainOrValidate = TrainOrValidate.TRAIN,
            **kwargs,
    ) -> Tuple[Any, Any]:
        self.state = replace(
            self.state,
            last_input=x,
            last_target=y,
            train_or_validate=train,
            stop_epoch=False,
            skip_step=False,
            skip_zero_grad=False,
            skip_bwd_pass=False,
        )

        self("batch_begin", **kwargs)
        return self.state.last_input, self.state.last_target

    def on_loss_begin(self, *, out: Tensor, **kwargs) -> Tensor:
        self.state = replace(self.state, last_output=out)
        self("loss_begin", **kwargs)
        return self.state.last_output

    def on_backward_begin(self, *, loss: Tensor, **kwargs) -> Tuple[Tensor, bool]:
        self.state = replace(self.state, last_loss=loss)
        self("backward_begin", **kwargs)
        return self.state.last_loss, self.state.skip_bwd_pass

    def on_backward_end(self, **kwargs) -> bool:
        self("backward_end", **kwargs)
        return self.state.skip_step

    def on_step_end(self, **kwargs) -> bool:
        self("step_end", **kwargs)
        return self.state.skip_zero_grad

    def on_batch_end(self, *, loss: Tensor, **kwargs) -> bool:
        self.state = replace(self.state, last_loss=loss)
        self("batch_end", **kwargs)
        if self.state.train_or_validate == TrainOrValidate.TRAIN:
            self.state = replace(
                self.state,
                iteration=self.state.iteration + 1,
                num_batch=self.state.num_batch + 1,
            )
        return self.state.stop_epoch

    def on_epoch_end(self, *, validation_loss: Tensor, **kwargs) -> bool:
        self("epoch_end", **kwargs)
        self.state = replace(self.state, epoch=self.state.epoch + 1, last_loss=validation_loss)
        return self.state.stop_training

    def on_train_end(self, *, exception: Union[None, Exception]) -> None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self("train_end", exception=exception)

    @property
    def skip_validate(self) -> bool:
        return self.state.skip_validate
