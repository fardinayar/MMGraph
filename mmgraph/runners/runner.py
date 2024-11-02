from mmengine.runner import Runner as mmengine_Runner
import mmengine.runner
import torch_geometric.data
from ..registry import RUNNERS
from typing import Tuple, Union, Dict, Optional
from torch.utils.data import DataLoader
import torch_geometric
from ..registry import DATASETS

class GCNDataLoader(DataLoader):
    """
    A dummy dataloader for GCN that mimics PyTorch DataLoader behavior.
    It works with single torch_geometric.data.Data objects.
    """

    def __init__(self, dataset: torch_geometric.data.Data, **kwargs):
        self.data = dataset if isinstance(dataset, torch_geometric.data.Data) else DATASETS.build(dataset)
        self.batch_size = 1
        self.num_workers = 0
        self.pin_memory = False

    def __iter__(self):
        yield self.data

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return [self.data]

    def __getattr__(self, name):
        # Implement any other DataLoader attributes/methods as needed
        return getattr(self.data, name)

@RUNNERS.register_module()
class Runner(mmengine_Runner):
    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        if isinstance(dataloader, GCNDataLoader):
            return dataloader
        elif isinstance(dataloader, Dict):
            return GCNDataLoader(**dataloader)
        
    
def _update_losses(data: torch_geometric.data.Data, losses: dict) -> Tuple[list, dict]:
    """Update and record the losses of the network.

    """
    return data, losses

import mmengine
mmengine.runner.loops._update_losses = _update_losses