from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union

from torch import Tensor

from callbacks.callbacks import Callback


class TrainOrValidate(Enum):
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
    last_metrics: Tensor = None
    stop_training: bool = False
    stop_epoch: bool = False
    skip_validate: bool = False
    skip_step: bool = False
    skip_zero_grad: bool = False
    skip_bwd_pass: bool = False
    train_or_validate: TrainOrValidate = TrainOrValidate.TRAIN


class Pipeline:
    callbacks: List[Callback]
    metrics: List[Callback]
    state: PipelineState

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
        self.state = PipelineState()

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

    def call_cb(self, cb_name, call_mets=True, **kwargs) -> None:
        for met in self.metrics:
            self._call_and_update(met, cb_name, **kwargs)
        for cb in self.callbacks:
            self._call_and_update(cb, cb_name, **kwargs)

    def on_train_begin(self, **kwargs) -> None:
        self.call_cb("train_begin", **kwargs)

    def on_epoch_begin(self, **kwargs) -> None:
        self.state = replace(self.state, num_batch=0, stop_training=False)
        self.call_cb("epoch_begin", **kwargs)

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

        self.call_cb("batch_begin", **kwargs)
        return self.state.last_input, self.state.last_target

    def on_loss_begin(self, *, out: Tensor, **kwargs) -> Tensor:
        self.state = replace(self.state, last_output=out)
        self.call_cb("loss_begin", **kwargs)
        return self.state.last_output

    def on_backward_begin(self, *, loss: Tensor, **kwargs) -> Tuple[Tensor, bool]:
        self.state = replace(self.state, last_loss=loss)
        self.call_cb("backward_begin", **kwargs)
        return self.state.last_loss, self.state.skip_bwd_pass

    def on_backward_end(self, **kwargs) -> bool:
        self.call_cb("backward_end", **kwargs)
        return self.state.skip_step

    def on_step_end(self, **kwargs) -> bool:
        self.call_cb("step_end", **kwargs)
        return self.state.skip_zero_grad

    def on_batch_end(self, *, loss: Tensor, **kwargs) -> bool:
        self.state = replace(self.state, last_loss=loss)
        self.call_cb("batch_end", **kwargs)
        if self.state.train_or_validate == TrainOrValidate.TRAIN:
            self.state = replace(
                self.state,
                iteration=self.state.iteration + 1,
                num_batch=self.state.num_batch + 1,
            )
        return self.state.stop_epoch

    def on_epoch_end(self, *, validation_loss: Tensor, **kwargs) -> bool:
        self.call_cb("epoch_end", **kwargs)
        self.state = replace(
            self.state, epoch=self.state.epoch + 1, last_loss=validation_loss
        )
        return self.state.stop_training

    def on_train_end(self, *, exception: Union[None, Exception]) -> None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self.call_cb("train_end", exception=exception)

    @property
    def skip_validate(self) -> bool:
        return self.state.skip_validate
