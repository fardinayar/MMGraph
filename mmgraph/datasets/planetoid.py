from ..registry import DATASETS
from torch_geometric.datasets import Planetoid




DATASETS.register_module('Planetoid', module=Planetoid)
