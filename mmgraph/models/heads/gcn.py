import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ...registry import MODELS
from ...registry import LAYERS
from typing import List, Optional, Union

@MODELS.register_module()
class GCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_classes: int,
                 num_layers: int,
                 layer: Union[dict, List[dict]],
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 loss: Optional[dict] = None):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        
        if isinstance(layer, dict):
            # Use the same layer configuration for all layers
            for i in range(num_layers):
                if i == 0:
                    in_ch = in_channels
                elif i == num_layers - 1:
                    in_ch = hidden_channels
                    layer['out_channels'] = num_classes
                else:
                    in_ch = hidden_channels
                layer_config = layer.copy()
                layer_config['in_channels'] = in_ch
                layer_config['out_channels'] = layer_config.get('out_channels', hidden_channels)
                self.convs.append(LAYERS.build(layer_config))
        elif isinstance(layer, list):
            # Use different layer configurations for each layer
            assert len(layer) == num_layers, f"Expected {num_layers} layer configs, but got {len(layer)}"
            for i, layer_config in enumerate(layer):
                if i == 0:
                    layer_config['out_channels'] = layer_config.get('out_channels', hidden_channels)
                    layer_config['in_channels'] = in_channels
                elif i == num_layers - 1:
                    layer_config['in_channels'] = layer_config.get('in_channels', hidden_channels)
                    layer_config['out_channels'] = num_classes
                else:
                    layer_config['in_channels'] = layer_config.get('in_channels', hidden_channels)
                    layer_config['out_channels'] = layer_config.get('out_channels', hidden_channels)
                self.convs.append(LAYERS.build(layer_config))
        else:
            raise ValueError("layer must be either a dict or a list of dicts")

        self.activation = getattr(F, activation)
        
        self.loss = MODELS.build(loss) if loss is not None else F.cross_entropy

    def forward(self, data, target_mask=None, mode='tensor'):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        
        data.logits = x
        data.features = x

        if mode == 'loss':
            assert target_mask is not None, "Target mask is required for loss computation"
            loss = self.loss(x[target_mask], data.y[target_mask])
            data.losses = {'loss': loss}
        elif mode == 'predict':
            data.pred = x.argmax(dim=-1)

        return data