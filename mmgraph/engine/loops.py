from typing import List, Union
import torch
from ..evaluation.evaluator import Evaluator
from mmengine.runner.loops import TestLoop as mmengine_TestLoop
from mmengine.runner.amp import autocast
import torch_geometric.data
from ..registry import LOOPS
from .runners.runner import _update_losses
import torch_geometric

@LOOPS.register_module()
class TestLoop(mmengine_TestLoop):
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Union[torch_geometric.data.Data, List[torch_geometric.data.Data]]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Union[torch_geometric.data.Data, List[torch_geometric.data.Data]): List when TTA
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)