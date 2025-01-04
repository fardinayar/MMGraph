from abc import ABCMeta, abstractmethod
import torch_geometric
import torch_geometric.data

class BaseTransform(metaclass=ABCMeta):
    """Abstract base class for graph data transformations.

    This class serves as a blueprint for creating custom transformations
    on graph data using the PyTorch Geometric library. Subclasses must
    implement the `transform` method to define specific transformation logic.
    """

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Applies the transformation to the given graph data.

        Args:
            data (torch_geometric.data.Data): The input graph data to be transformed.

        Returns:
            torch_geometric.data.Data: The transformed graph data.
        """
        return self.transform(data)

    @abstractmethod
    def transform(self, results: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Defines the transformation logic to be applied to the graph data.

        This method must be implemented by subclasses to specify how the
        input graph data should be transformed. If the input data includes
        a mask (e.g., an explainability mask), it should be considered in
        the transformation process if applicable. This mask can be used to
        focus the transformation on specific parts of the graph data.

        Args:
            results (torch_geometric.data.Data): The input graph data to be transformed.

        Returns:
            torch_geometric.data.Data: The transformed graph data.
        """
        pass