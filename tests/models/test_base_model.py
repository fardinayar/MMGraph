import unittest
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from mmgraph.models.base_model import GNNBaseModel
from mmgraph.registry import MODELS
from mmengine.registry import OPTIM_WRAPPERS

class TestGNNBaseModel(unittest.TestCase):
    def setUp(self):
        # Load the Cora dataset
        self.dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
        self.data = self.dataset

        # Define model components
        head = dict(
            type='GCN',
            in_channels=self.dataset.num_features,
            hidden_channels=16,
            num_classes=self.dataset.num_classes,
            num_layers=2,
            layer={'type': 'GCNConv'}
        )

        self.model = MODELS.build(dict(
            type='GNNBaseModel',
            head=head
        )).cuda()
        # Initialize the model

    def test_initialization(self):
        self.assertIsInstance(self.model, GNNBaseModel)
        self.assertFalse(self.model.has_post_processer)

    def test_forward_tensor_mode(self):
        output = self.model(self.data, mode='tensor')
        self.assertIn('logits', output)
        self.assertIn('features', output)
        self.assertEqual(output.logits.shape, (self.data.num_nodes, self.dataset.num_classes))

    def test_forward_loss_mode(self):
        output = self.model(self.data, target_mask=self.data.train_mask, mode='loss')
        self.assertIn('losses', output)
        self.assertIn('loss', output.losses)
        self.assertIsInstance(output.losses['loss'], torch.Tensor)

    def test_forward_predict_mode(self):
        output = self.model(self.data, mode='predict')
        self.assertIn('pred', output)
        self.assertEqual(output.pred.shape, (self.data.num_nodes,))
        self.assertEqual(output.pred.dtype, torch.long)

    def test_val_step(self):
        output = self.model.val_step(self.data)
        self.assertIn('pred', output)
        self.assertEqual(output.pred.shape, (self.data.num_nodes,))
        self.assertEqual(output.pred.dtype, torch.long)

    def test_test_step(self):
        output = self.model.test_step(self.data)
        self.assertIn('pred', output)
        self.assertEqual(output.pred.shape, (self.data.num_nodes,))
        self.assertEqual(output.pred.dtype, torch.long)

    def test_train_step(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optim_wrapper = OPTIM_WRAPPERS.build(dict(type='OptimWrapper', optimizer=optimizer))
        
        # Store initial weights
        initial_weights = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        initial_loss = None
        for _ in range(10):  # Run 10 training steps
            log_vars = self.model.train_step(self.data, optim_wrapper)
            self.assertIsInstance(log_vars, dict)
            self.assertIn('loss', log_vars)
            
            if initial_loss is None:
                initial_loss = log_vars['loss']
        
        final_loss = log_vars['loss']
        
        # Check if the model is actually training
        self.assertIsInstance(initial_loss, torch.Tensor)
        self.assertIsInstance(final_loss, torch.Tensor)
        self.assertLess(final_loss.item(), initial_loss.item(), 
                        "Loss should decrease after training steps")

        # Check if weights have been updated
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.allclose(initial_weights[name], param),
                            f"Parameter {name} did not change during training")

        # Original assertions
        self.assertIsInstance(log_vars, dict)
        self.assertIn('loss', log_vars)