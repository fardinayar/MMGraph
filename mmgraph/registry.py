from mmengine.registry import Registry
from mmengine.registry import MODELS as mmengine_MODELS
from mmengine.registry import RUNNERS as mmengine_RUNNERS
from mmengine.registry import METRICS as mmengine_METRICS

LAYERS = Registry('layers', scope='mmgraph')
MODELS = Registry('models', scope='mmgraph', parent=mmengine_MODELS)
DATASETS = Registry('datasets', scope='mmgraph')
RUNNERS = Registry('runners', scope='mmgraph', parent=mmengine_RUNNERS)
METRICS = Registry('metrics', scope='mmgraph', parent=mmengine_METRICS)
