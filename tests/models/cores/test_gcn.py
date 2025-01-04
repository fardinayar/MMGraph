import unittest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn
from mmgraph.models.cores.gcn import GCN
import torch_geometric

class TestGCN(unittest.TestCase):
    def test_gcn_init_with_single_dict(self):
        in_channels = 16
        hidden_channels = 32
        num_classes = 10
        num_layers = 3
        layer_config = {'type': 'GCNConv'}
        
        model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            layer=layer_config
        )
        
        self.assertEqual(len(model.convs), num_layers)
        self.assertEqual(model.num_layers, num_layers)
        self.assertEqual(model.dropout, 0.5)
        self.assertEqual(model.activation, F.relu)
        
        # Check the first layer
        self.assertEqual(model.convs[0].in_channels, in_channels)
        self.assertEqual(model.convs[0].out_channels, hidden_channels)
        
        # Check the middle layer
        self.assertEqual(model.convs[1].in_channels, hidden_channels)
        self.assertEqual(model.convs[1].out_channels, hidden_channels)
        
        # Check the last layer
        self.assertEqual(model.convs[-1].in_channels, hidden_channels)
        self.assertEqual(model.convs[-1].out_channels, num_classes)

    def test_gcn_init_with_list_of_dicts(self):
        in_channels = 16
        hidden_channels = 32
        num_classes = 10
        num_layers = 3
        layer_config = [
            {'type': 'GCNConv'},
            {'type': 'GCNConv'},
            {'type': 'GCNConv'}
        ]
        
        model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            layer=layer_config
        )
        
        self.assertEqual(len(model.convs), num_layers)
        self.assertEqual(model.num_layers, num_layers)
        self.assertEqual(model.dropout, 0.5)
        self.assertEqual(model.activation, F.relu)
        
        # Check the first layer
        self.assertEqual(model.convs[0].in_channels, in_channels)
        self.assertEqual(model.convs[0].out_channels, hidden_channels)
        
        # Check the middle layer
        self.assertEqual(model.convs[1].in_channels, hidden_channels)
        self.assertEqual(model.convs[1].out_channels, hidden_channels)
        
        # Check the last layer
        self.assertEqual(model.convs[-1].in_channels, hidden_channels)
        self.assertEqual(model.convs[-1].out_channels, num_classes)

    def test_gcn_init_dict_first_layer_in_channels(self):
        in_channels = 16
        hidden_channels = 32
        num_classes = 10
        num_layers = 3
        layer_config = {'type': 'GCNConv'}
        
        model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            layer=layer_config
        )
        
        self.assertEqual(model.convs[0].in_channels, in_channels)
        self.assertIsInstance(model.convs[0], torch_geometric.nn.GCNConv)

    def test_gcn_init_mismatch_layer_configs(self):
        in_channels = 16
        hidden_channels = 32
        num_classes = 10
        num_layers = 3
        layer_config = [
            {'type': 'GCNConv'},
            {'type': 'GCNConv'}
        ]
        
        with self.assertRaises(AssertionError):
            GCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_classes=num_classes,
                num_layers=num_layers,
                layer=layer_config
            )

    def test_gcn_forward_modes(self):
        in_channels = 16
        hidden_channels = 32
        num_classes = 10
        num_layers = 3
        layer_config = {'type': 'GCNConv'}
        
        model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            layer=layer_config
        )
        
        # Create dummy data
        num_nodes = 100
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.randint(0, num_nodes, (2, 200))
        y = torch.randint(0, num_classes, (num_nodes,))
        target_mask = torch.randint(0, 2, (num_nodes,)).bool()
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Test tensor mode
        output_tensor = model(data, mode='tensor')
        self.assertTrue(hasattr(output_tensor, 'logits'))
        self.assertTrue(hasattr(output_tensor, 'features'))
        self.assertEqual(output_tensor.logits.shape, (num_nodes, num_classes))
        self.assertEqual(output_tensor.features.shape, (num_nodes, hidden_channels))
        
        # Test loss mode
        output_loss = model(data, target_mask=target_mask, mode='loss')
        self.assertTrue(hasattr(output_loss, 'losses'))
        self.assertIn('loss', output_loss.losses)
        self.assertIsInstance(output_loss.losses['loss'], torch.Tensor)
        self.assertEqual(output_loss.losses['loss'].dim(), 0)  # scalar loss
        
        # Test predict mode
        output_predict = model(data, mode='predict')
        self.assertTrue(hasattr(output_predict, 'pred'))
        self.assertEqual(output_predict.pred.shape, (num_nodes,))
        self.assertEqual(output_predict.pred.dtype, torch.long)

