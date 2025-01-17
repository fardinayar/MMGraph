# Copyright (c) OpenMMLab. All rights reserved.
from ..registry import EVALUATOR
from mmengine.evaluator import Evaluator as mmengine_Evaluator
import torch_geometric

@EVALUATOR.register_module()
class Evaluator(mmengine_Evaluator):

    def process(self,
                data_samples: torch_geometric.data.Data,
                *args, **kwargs) -> torch_geometric.data.Data:
        for metric in self.metrics:
            metric.process(data_samples=data_samples, *args, **kwargs)
