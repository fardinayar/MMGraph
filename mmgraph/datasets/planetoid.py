from ..registry import DATASETS
from torch_geometric.datasets import Planetoid as Planetoid_pyg




@DATASETS.register_module()
class Planetoid:
    def __init__(self, *args, **kwargs):
        self.data = Planetoid_pyg(*args, **kwargs)[0]

    def __getattr__(self, name):
        return getattr(self.data, name)