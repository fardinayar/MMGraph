from mmengine.model import BaseTTAModel
from ...evaluation.evaluator import Evaluator  # Add this import
from ...registry import MODELS
from typing import List, Union
import torch_geometric.data

@MODELS.register_module()
class TTAModel(BaseTTAModel):
    """Graph Test-Time Augmentation Model.

    This model handles test-time augmentation for graph data by merging predictions
    from multiple augmented versions of the input data.
    """

    def merge_preds(self, data_list: List[torch_geometric.data.Data]) -> torch_geometric.data.Data:
        """Merge predictions from multiple augmented data.

        Args:
            data_list (List[torch_geometric.data.Data]): List of augmented data.

        Returns:
            torch_geometric.data.Data: Merged data sample with combined predictions.
        """
        data = data_list[0].clone()
        merged_score = self._aggregate(data_list)
        data.pred = merged_score.round()
        return data

    def _aggregate(self, data_list: List[torch_geometric.data.Data]) -> torch_geometric.data.Data:
        """Merge predictions from a single set of augmented data.

        Args:
            data_list (List[torch_geometric.data.Data]): List of augmented data.

        Returns:
            torch_geometric.data.Data: Merged data sample with combined predictions.
        """
        return sum(data_sample.pred for data_sample in data_list) / len(data_list)
    
    def test_step(self, data_list: List[torch_geometric.data.Data]) -> torch_geometric.data.Data:
        """Get predictions from each augmented data sample and merge them.

        Args:
            data (Union[dict, list]): Enhanced data batch sampled from dataloader.

        Returns:
            torch_geometric.data.Data: Merged prediction.
        """
        predictions = []
        for data_sample in data_list:
            predictions.append(self.module.test_step(data_sample))
        data = self.merge_preds(predictions)
        return data