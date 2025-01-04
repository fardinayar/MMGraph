from mmengine.runner import Runner as mmengine_Runner
import torch_geometric.data
from ...registry import RUNNERS
from typing import Tuple, Union, Dict, Optional
from torch.utils.data import DataLoader
import torch_geometric
from ...registry import DATASETS
from mmengine.logging import HistoryBuffer
import torch
from mmengine.utils import is_list_of
import mmengine


class GCNDataLoader(DataLoader):
    """
    A dummy dataloader for GCN that mimics PyTorch DataLoader behavior.
    It works with single torch_geometric.data.Dataset objects.
    """

    def __init__(self, dataset: torch_geometric.data.Dataset, **kwargs):
        self.data = dataset if isinstance(dataset, torch_geometric.data.Dataset) else DATASETS.build(dataset)
        self.batch_size = 1
        self.num_workers = 0
        self.pin_memory = False

    def __iter__(self):
        yield self.data[0]

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
        
    

def _update_losses(outputs: torch_geometric.data.Data, losses: dict) -> Tuple[torch_geometric.data.Data, dict]:
    """Update and record the losses of the network.

    Args:
        outputs (torch_geometric.data.Data): The outputs of the network.
        losses (dict): The losses of the network.

    Returns:
        torch_geometric.data.Data: The updated outputs of the network.
        dict: The updated losses of the network.
    """
    if 'loss' in outputs.keys():
        loss = outputs.loss  # type: ignore
    else:
        loss = dict()

    for loss_name, loss_value in loss.items():
        if loss_name not in losses:
            losses[loss_name] = HistoryBuffer()
        if isinstance(loss_value, torch.Tensor):
            losses[loss_name].update(loss_value.item())
        elif is_list_of(loss_value, torch.Tensor):
            for loss_value_i in loss_value:
                losses[loss_name].update(loss_value_i.item())
    return outputs, losses

mmengine.runner.loops._update_losses = _update_losses
