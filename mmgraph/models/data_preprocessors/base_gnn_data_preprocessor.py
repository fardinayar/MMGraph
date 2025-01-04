from mmengine.model import BaseDataPreprocessor as mmengine_BaseDataPreprocessor
from ...registry import MODELS
import torch_geometric

@MODELS.register_module()
class BaseGNNDataPreprocessor(mmengine_BaseDataPreprocessor):
    def cast_data(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        return data.to(self.device)
        

    