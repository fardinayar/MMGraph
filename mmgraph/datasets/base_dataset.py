from typing import Optional
from mmengine.dataset import Compose as mmengine_Compose
import torch_geometric

class Compose(mmengine_Compose):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __call__(self, data: torch_geometric.data.Data) -> Optional[torch_geometric.data.Data]:
        """Call function to apply transforms sequentially.

        Args:
            data (torch_geometric.data.Data): A torch_geometric.data.Data contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data
