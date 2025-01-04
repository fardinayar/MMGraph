import torch_geometric.data
from ...registry import MODELS
from mmengine.model import BaseModel
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from enum import Enum
from typing import Optional, Union, Dict
import torch_geometric
from mmengine.optim import OptimWrapper
from ..data_preprocessors import BaseGNNDataPreprocessor

class ForwardMode(Enum):
    LOSS = 'loss'
    PREDICT = 'predict'
    TENSOR = 'tensor'


@MODELS.register_module()
class GNNBaseModel(BaseModel):
    """Base model for all GNN models classes.
    All GNN models has an optional feature extractor and a core to input class logits. 
    There also an optinal post_processer.
    """
    def __init__(self,
                 core: dict,
                 feature_extractor: Optional[dict] = None,
                 post_processer: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        """
        Initialize the GNNBaseModel.

        This constructor sets up the core components of the GNN model, including
        the core, feature extractor, post-processor, and data preprocessor.

        Args:
            core (dict): Configuration for the core GNN model.
            feature_extractor (Optional[dict], optional): Configuration for the feature extractor. 
                If None, no feature extractor is used. Defaults to None.
            post_processer (Optional[dict], optional): Configuration for the post-processor. 
                If None, no post-processor is used. Defaults to None.
            data_preprocessor (Optional[dict], optional): Configuration for the data preprocessor. 
                If None, a default BaseGNNDataPreprocessor is used. Defaults to None.
            init_cfg (Optional[dict], optional): Initialization configuration passed to the parent class. 
                Defaults to None.

        Returns:
            None
        """
        super().__init__(init_cfg)

                
        if data_preprocessor is None:
            self.data_preprocessor = MODELS.build(dict(type='BaseGNNDataPreprocessor'))


        if feature_extractor is not None:
            self.feature_extractor = MODELS.build(feature_extractor)

        self.core = MODELS.build(core)
        if post_processer is not None:
            self.post_processer = MODELS.build(post_processer)
    
    @property
    def has_post_processer(self) -> bool:
        return hasattr(self, 'post_processer')
    
    @property
    def has_feature_extractor(self) -> bool:
        return hasattr(self, 'feature_extractor')
        
        
    
    def forward(self,
                data: torch_geometric.data.Data,
                target_mask: Optional[torch.Tensor] = None,
                mode: Union[str, ForwardMode] = ForwardMode.TENSOR) -> torch_geometric.data.Data:
        """Returns losses or predictions for training, validation, testing, and
        simple inference processes.

        This method processes the input graph data through the feature extractor
        and the core, and optionally applies post-processing. The input data is
        modified in-place.

        Args:
            data (torch_geometric.data.Data): The input graph data. This object
                is modified in-place with new attributes added during processing.
            target_mask (torch.Tensor, optional): A mask tensor indicating the
                target nodes for loss computation. Required when mode is 'loss'.
            mode (Union[str, ForwardMode], optional): Specifies the forward pass mode. 
                Should be one of 'loss', 'predict', or 'tensor'. Defaults to ForwardMode.TENSOR.

        Returns:
            torch_geometric.data.Data: The processed graph data (same object as input),
            containing new or updated fields:
                - 'logits': The output logits from the core.
                - 'features': The features extracted by the feature extractor or core.
                - 'losses': A dictionary containing loss values from the core.
                - Additional fields may be added or modified by the core or post-processor.

        Raises:
            AssertionError: If mode is 'loss' and target_mask is not provided.
            ValueError: If an invalid mode is provided.

        Note:
            - The input 'data' object is modified in-place. No new Data object is created.
            - When mode is 'loss', the core computes and stores loss information in the data object.
            - For 'predict' and 'tensor' modes, the core processes without loss computation.
            - If a post-processor is defined, it is applied to the data before returning,
            potentially adding or modifying more fields.
        """
        
        if isinstance(mode, str):
            try:
                mode = ForwardMode(mode)
            except ValueError:
                raise ValueError(f"Invalid mode: {mode}. Must be one of {', '.join([m.value for m in ForwardMode])}")
        
        if self.has_feature_extractor:
            self.feature_extractor(data)
            
            
        if mode == ForwardMode.LOSS:
            assert target_mask is not None, "You must provide 'target_mask' when mode is 'loss'"
            self.core.forward(data, target_mask, mode=ForwardMode.LOSS.value)
        elif mode == ForwardMode.PREDICT:
            self.core.forward(data, mode=ForwardMode.PREDICT.value)
        else:
            self.core.forward(data, mode=ForwardMode.TENSOR.value)
            
        if self.has_post_processer:
            self.post_processer(data)
            
        return data
    

    def train_step(self, data: torch_geometric.data.Data,
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_preprocessor(data, training=True)`` to preprocess the data.
        2. Calls ``self(data, mode='loss')`` to get processed data with losses.
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(parsed_losses)`` to update model.

        Args:
            data (torch_geometric.data.Data): Graph data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        #assert isinstance(data, torch_geometric.data.Data), data
        with optim_wrapper.optim_context(self):
            if hasattr(self, 'data_preprocessor'):
                data = self.data_preprocessor(data, training=True)
            processed_data = self(data, target_mask=data.train_mask ,mode=ForwardMode.LOSS)
            
            if not hasattr(processed_data, 'losses'):
                raise AttributeError("The forward method did not add 'losses' to the data object.")
            
        parsed_losses, log_vars = self.parse_losses(processed_data.losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    
    def val_step(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` if it exists and
        ``self(data, mode='predict')`` in order. Returns the
        processed data which will be passed to evaluator.

        Args:
            data (torch_geometric.data.Data): Graph data sampled from dataset.

        Returns:
            torch_geometric.data.Data: The processed graph data containing predictions.
        """
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(data, False)
        processed_data = self(data, mode=ForwardMode.PREDICT)
        
        if not hasattr(processed_data, 'pred'):
            raise AttributeError("The forward method did not add 'pred' to the data object.")
        
        return processed_data
    
    def test_step(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        return self.val_step(data)
    
    
    
    
