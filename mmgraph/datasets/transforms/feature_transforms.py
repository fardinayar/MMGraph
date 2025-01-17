from ...registry import TRANSFORMS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

@TRANSFORMS.register_module()
class NodeFeatureMasking(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        edge_attr = data.edge_attr
        train_mask = data.train_mask
        test_mask = data.test_mask
        edge_idx = data.edge_index

        n, d = x.shape
        
        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        x = x.clone()
        x[:, idx] = 0

        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask, edge_attr=edge_attr)
        return new_data
    
    