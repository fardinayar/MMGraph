from ...registry import TRANSFORMS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

@TRANSFORMS.register_module()
class EdgeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        train_mask = data.train_mask
        test_mask = data.test_mask
        edge_idx = data.edge_index

        edge_idx = edge_idx.permute(1, 0)
        idx = torch.empty(edge_idx.size(0)).uniform_(0, 1)
        edge_idx = edge_idx[torch.where(idx >= self.p)].permute(1, 0)
        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask)
        return new_data

    