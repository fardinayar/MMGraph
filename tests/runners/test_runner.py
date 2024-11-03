import unittest

import torch
from mmgraph.engine.runners.runner import Runner, GCNDataLoader
from torch_geometric.datasets import Planetoid
from mmgraph.registry import MODELS
from torch_geometric.transforms import NormalizeFeatures

class TestRunner(unittest.TestCase):
    def setUp(self):
        # Load the Cora dataset
        self.dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
        self.data = self.dataset
        # Define model components
        head = dict(
            type='GCN',
            in_channels=self.dataset.num_features,
            hidden_channels=64,
            num_classes=self.dataset.num_classes,
            num_layers=2,
            layer={'type': 'GCNConv'}
        )

        self.model = MODELS.build(dict(
            type='GNNBaseModel',
            head=head
        ))

        # Create a config dictionary
        self.config = dict(
            work_dir='/tmp/test_runner',
            model=self.model,
            train_dataloader=dict(
                dataset=self.data,
            ),
            val_dataloader=dict(
                dataset=self.data,
            ),
            test_dataloader=dict(
                dataset=self.data,
            ),
            train_cfg=dict(by_epoch=True, max_epochs=10),
            test_evaluator=dict(type='AccuracyMetric'),
            test_cfg=dict(),
            val_evaluator=dict(type='AccuracyMetric'),
            val_cfg=dict(),
            optim_wrapper=dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.01)),
        )

    def test_runner_initialization(self):
        runner = Runner(**self.config)
        self.assertIsInstance(runner, Runner)

    def test_build_dataloader(self):
        # Test with GCNDataLoader
        gcn_loader = GCNDataLoader(self.data)
        result = Runner.build_dataloader(gcn_loader)
        self.assertIs(result, gcn_loader)

        # Test with dictionary
        dict_loader = dict(dataset=self.data)
        result = Runner.build_dataloader(dict_loader)
        self.assertIsInstance(result, GCNDataLoader)
        
        for batch_idx, batch in enumerate(result):
            self.assertEqual(batch.num_nodes, self.data.num_nodes)
            self.assertEqual(batch.num_edges, self.data.num_edges)
            self.assertTrue(hasattr(batch, 'train_mask'))

    def test_train(self):
        runner = Runner(**self.config)
        runner.train()
        # Check if the model has been trained
        self.assertTrue(hasattr(runner.model, 'training'))

    def test_val(self):
        runner = Runner(**self.config)
        runner.val()
        # Check if validation has been performed
        self.assertTrue(hasattr(runner, 'val_evaluator'))

    def test_test(self):
        runner = Runner(**self.config)
        runner.test()
        # Check if testing has been performed
        self.assertTrue(hasattr(runner, 'test_evaluator'))


if __name__ == '__main__':
    unittest.main()