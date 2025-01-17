import unittest
import torch
from torch_geometric.data import Data
from mmgraph.datasets.transforms import NodeFeatureMasking, EdgeDrop, Compose, TTACompose

class TestTransforms(unittest.TestCase):
    def setUp(self):
        # Create a sample graph data
        self.data = Data(
            x=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float),
            y=torch.tensor([0, 1, 0], dtype=torch.long),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            train_mask=torch.tensor([True, False, True], dtype=torch.bool),
            test_mask=torch.tensor([False, True, False], dtype=torch.bool)
        )

    def test_node_feature_masking(self):
        transform = NodeFeatureMasking(p=0.5)
        transformed_data = transform(self.data)
        self.assertIsInstance(transformed_data, Data)
        self.assertEqual(transformed_data.x.shape, self.data.x.shape)

    def test_edge_drop(self):
        transform = EdgeDrop(p=0.5)
        transformed_data = transform(self.data)
        self.assertIsInstance(transformed_data, Data)
        self.assertLessEqual(transformed_data.edge_index.shape[1], self.data.edge_index.shape[1])

    def test_compose(self):
        transforms = Compose([NodeFeatureMasking(p=0.5), EdgeDrop(p=0.5)])
        transformed_data = transforms(self.data)
        self.assertIsInstance(transformed_data, Data)
        self.assertEqual(transformed_data.x.shape, self.data.x.shape)
        self.assertLessEqual(transformed_data.edge_index.shape[1], self.data.edge_index.shape[1])

    def test_tta_compose(self):
        transforms = TTACompose([NodeFeatureMasking(p=0.5), EdgeDrop(p=0.5)])
        transformed_data_list = transforms(self.data)
        self.assertIsInstance(transformed_data_list, list)
        for transformed_data in transformed_data_list:
            self.assertIsInstance(transformed_data, Data)
            self.assertEqual(transformed_data.x.shape, self.data.x.shape)
            self.assertLessEqual(transformed_data.edge_index.shape[1], self.data.edge_index.shape[1])

if __name__ == '__main__':
    unittest.main()