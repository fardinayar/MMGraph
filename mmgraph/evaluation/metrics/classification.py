from typing import Any, Sequence
from mmengine.evaluator import BaseMetric
import torch_geometric.data
from ...registry import METRICS
# TODO
import torch
import numpy as np
import torch_geometric 

@METRICS.register_module()
class AccuracyMetric(BaseMetric):
    """Accuracy metric for MMGraph.

    This metric calculates the accuracy of predictions for a classification task.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """

    default_prefix: str = 'accuracy'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: str = None) -> None:
        super().__init__(collect_device, prefix)

    def process(self, data_samples: torch_geometric.data.Data, *args ,**kwargs) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_samples (torch_geometric.data.Data): A torch_geometric.data.Data instance containing 'pred' and 'y' keys.
        """
        
        pred = data_samples.pred
        label = data_samples.y

        # Ensure pred and label are on the same device
        if pred.device != label.device:
            pred = pred.to(label.device)

        # If pred is a probability distribution, get the class with highest probability
        if pred.dim() > 1:
            pred = pred.argmax(dim=-1)

        # Store the predictions and labels for later computation
        self.results.append({
            'pred': pred.detach().cpu(),
            'label': label.detach().cpu()
        })

    def compute_metrics(self, results: list) -> dict:
        """Compute the accuracy metric.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed accuracy metric.
        """
        preds = torch.cat([res['pred'] for res in results])
        labels = torch.cat([res['label'] for res in results])

        correct = (preds == labels).sum().item()
        total = len(labels)
        accuracy = correct / total

        return {'accuracy': accuracy}