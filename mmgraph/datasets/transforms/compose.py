from typing import List,Optional,Sequence,Callable,Union
from mmengine.dataset import Compose as mmengine_Compose
import torch_geometric
from ...registry import TRANSFORMS

@TRANSFORMS.register_module()
class Compose(mmengine_Compose):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """
    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')
    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
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

@TRANSFORMS.register_module()
class TTACompose(Compose):
    """Compose multiple transforms sequentially for TTA.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """
    def __call__(self, data: torch_geometric.data.Data) -> List[torch_geometric.data.Data]:
        """Call function to apply transforms sequentially.

        Args:
            data (torch_geometric.data.Data): A torch_geometric.data.Data contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        data_list = []
        for t in self.transforms:
            data = t(data)
            data_list.append(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data_list