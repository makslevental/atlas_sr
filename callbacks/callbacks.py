from abc import ABC, abstractmethod
from typing import Any

from train.pipeline import PipelineState


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
