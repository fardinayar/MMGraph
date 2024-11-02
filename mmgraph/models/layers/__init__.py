from ...registry import LAYERS
from torch_geometric.nn import GCNConv, GATv2Conv

LAYERS.register_module(module=GCNConv)
LAYERS.register_module(module=GATv2Conv)
