default_scope = 'mmgraph'

dataset = dict(
    type='Planetoid',
    root='/tmp/Cora',
    name='Cora',
)

core = dict(
    type='GCN',
    in_channels=1433,
    hidden_channels=64,
    num_classes=7,
    num_layers=2,
    layer={'type': 'GCNConv'}
)

model = dict(
    type='GNNBaseModel',
    core=core
    )

train_dataloader=dict(
    dataset=dataset,
)
val_dataloader=dict(
    dataset=dataset,
)
test_dataloader=dict(
    dataset=dataset,
)
train_cfg=dict(by_epoch=True, max_epochs=10)
test_evaluator=dict(type='AccuracyMetric')
test_cfg=dict()
val_evaluator=dict(type='AccuracyMetric')
val_cfg=dict()
optim_wrapper=dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.01))
