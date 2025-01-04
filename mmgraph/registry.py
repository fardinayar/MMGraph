from mmengine.registry import Registry
from mmengine.registry import MODELS as mmengine_MODELS
from mmengine.registry import RUNNERS as mmengine_RUNNERS
from mmengine.registry import METRICS as mmengine_METRICS

LAYERS = Registry('layer', scope='mmgraph')
MODELS = Registry('model', scope='mmgraph', parent=mmengine_MODELS)
DATASETS = Registry('dataset', scope='mmgraph')
RUNNERS = Registry('runner', scope='mmgraph', parent=mmengine_RUNNERS)
METRICS = Registry('metric', scope='mmgraph', parent=mmengine_METRICS)
TRANSFORMS = Registry('transform', scope='mmgraph')
