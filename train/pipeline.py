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

    @abstractmethod
    def __init__(self):
        pass

    def on_train_begin(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_epoch_begin(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_batch_begin(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_loss_begin(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_backward_begin(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_backward_end(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_step_end(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_batch_end(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_epoch_end(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def on_train_end(self, pipeline_state: PipelineState, **kwargs) -> Any:
        pass

    def set_to_epoch(self, epoch) -> Any:
        pass


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
            new = getattr(cb, f"on_{cb_name}")(pipeline_state=self.state, **kwargs)
            if new is not None:
                for k, v in new.items():
                    if k not in self.state.__dataclass_fields__:
                        raise Exception(
                            f"{k} isn't a valid key in the state of the pipeline."
                        )
                    else:
                        self.state = replace(self.state, **{k: v})

    def call_cb(self, cb_name, **kwargs) -> None:
        for met in self.metrics:
            self._call_and_update(met, cb_name, **kwargs)
        for cb in self.callbacks:
            self._call_and_update(cb, cb_name, **kwargs)

    def on_train_begin(self, **kwargs) -> None:
        self.state = replace(self.state, epoch=0)
        self.call_cb("train_begin", **kwargs)

    def on_epoch_begin(self, **kwargs) -> None:
        self.state = replace(self.state, num_batch=0, stop_training=False)
        self.call_cb("epoch_begin", **kwargs)

    def on_batch_begin(
            self, *, x: Tensor, y: Tensor, ru_mode: RunMode = RunMode.TRAIN, **kwargs
    ):
        self.state = replace(
            self.state,
            last_input=x,
            last_target=y,
            run_mode=ru_mode,
            stop_epoch=False,
            skip_step=False,
            skip_zero_grad=False,
            skip_bwd_pass=False,
        )

        self.call_cb("batch_begin", **kwargs)

    def on_loss_begin(self, *, out: Tensor, **kwargs):
        self.state = replace(self.state, last_output=out)
        self.call_cb("loss_begin", **kwargs)

    def on_backward_begin(self, *, loss: Tensor, **kwargs):
        self.state = replace(self.state, last_loss=loss)
        self.call_cb("backward_begin", **kwargs)

    def on_backward_end(self, **kwargs):
        self.call_cb("backward_end", **kwargs)

    def on_step_end(self, **kwargs):
        self.call_cb("step_end", **kwargs)

    def on_batch_end(self, *, loss: Tensor, **kwargs):
        self.state = replace(self.state, last_loss=loss)
        self.call_cb("batch_end", **kwargs)
        if self.state.run_mode == RunMode.TRAIN:
            self.state = replace(
                self.state,
                iteration=self.state.iteration + 1,
                num_batch=self.state.num_batch + 1,
            )

    def on_epoch_end(self, *, validation_loss: Tensor, **kwargs):
        self.call_cb("epoch_end", **kwargs)
        self.state = replace(
            self.state, epoch=self.state.epoch + 1, last_loss=validation_loss
        )

    def on_train_end(self, *, exception: Optional[Exception], **kwargs):
        self.call_cb("train_end", exception=exception, **kwargs)

    @property
    def skip_validate(self) -> bool:
        return self.state.skip_validate
