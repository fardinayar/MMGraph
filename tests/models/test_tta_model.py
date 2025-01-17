import unittest
import torch
from torch_geometric.data import Data
from mmgraph.models.base_models.tta_model import TTAModel
from mmgraph.registry import MODELS

class TestTTAModel(unittest.TestCase):
    def setUp(self):
        
        core = dict(
            type='GCN',
            in_channels=2,
            hidden_channels=16,
            num_classes=2,
            num_layers=2,
            layer={'type': 'GCNConv'}
        )

        model = MODELS.build(dict(
            type='GNNBaseModel',
            core=core
        ))
        
        self.model = TTAModel(module=model)  # Assuming module is not needed for merge_preds
        self.data_list = [
            Data(x=torch.tensor([[1, 2], [3, 4]], dtype=torch.float), 
                 edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), 
                 pred=torch.tensor([0.1, 0.9], dtype=torch.float)),
            Data(x=torch.tensor([[1, 2], [3, 4]], dtype=torch.float), 
                 edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), 
                 pred=torch.tensor([0.2, 0.8], dtype=torch.float)),
            Data(x=torch.tensor([[1, 2], [3, 4]], dtype=torch.float), 
                 edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), 
                 pred=torch.tensor([0.3, 0.7], dtype=torch.float))
        ]

    def test_merge_preds(self):
        merged_data = self.model.merge_preds(self.data_list)
        expected_pred = torch.tensor([0.0, 1.0], dtype=torch.float)
        self.assertTrue(torch.equal(merged_data.pred, expected_pred))

if __name__ == '__main__':
    unittest.main()