import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mmgraph.engine.runners.runner import Runner, GCNDataLoader
from torch_geometric.datasets import Planetoid
from mmgraph.registry import MODELS, DATASETS
from torch_geometric.transforms import NormalizeFeatures
# Add parent path to import path

dataset = dict(
    type='Planetoid',
    root='/tmp/Cora',
    name='Cora',
    transform=NormalizeFeatures()
)
head = dict(
    type='GCN',
    in_channels=1433,
    hidden_channels=64,
    num_classes=7,
    num_layers=2,
    layer={'type': 'GCNConv'}
)

model = MODELS.build(dict(
    type='GNNBaseModel',
    head=head
    )
)

config = dict(
    work_dir='/tmp/test_runner',
    model=model,
    train_dataloader=dict(
        dataset=dataset,
    ),
    val_dataloader=dict(
        dataset=dataset,
    ),
    test_dataloader=dict(
        dataset=dataset,
    ),
    train_cfg=dict(by_epoch=True, max_epochs=10),
    test_evaluator=dict(type='AccuracyMetric'),
    test_cfg=dict(),
    val_evaluator=dict(type='AccuracyMetric'),
    val_cfg=dict(),
    optim_wrapper=dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.01)),
)

runner = Runner(**config)
runner.train()