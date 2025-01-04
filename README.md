# MMGraph

MMGraph is a modular and extensible framework for graph neural networks (GNNs) built on top of PyTorch Geometric and MMEngine. It provides a flexible architecture for defining, training, and evaluating GNN models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Logic](#model-logic)

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

### Training

To train a model, use the [`tools/train.py`](tools/train.py) script:

```sh
python tools/train.py configs/simple_config.py
```

## Project Structure

```
MMGraph/
├── configs/                  # Configuration files
│   └── simple_config.py      # Example configuration file
├── mmgraph/                  # Main package directory
│   ├── datasets/             # Dataset related code
│   │   ├── base_dataset.py   # Base dataset class
│   │   ├── planetoid.py      # Planetoid dataset implementation
│   │   └── transforms/       # Data transformation scripts
│   │       ├── base_transform.py        # Base transform class
│   │       ├── feature_transforms.py    # Feature transformation scripts
│   │       └── structure_transforms.py  # Structure transformation scripts
│   ├── engine/               # Training engine
│   │   └── runners/          # Training runners
│   │       └── runner.py     # Runner implementation
│   ├── evaluation/           # Evaluation metrics
│   │   └── metrics/          # Metric implementations
│   │       └── classification.py  # Classification metrics
│   ├── models/               # Model definitions
│   │   ├── base_models/      # Base model classes
│   │   │   └── gnn_base_model.py  # Base GNN model class
│   │   ├── cores/            # Core model components
│   │   │   └── gcn.py        # GCN model implementation
│   │   ├── data_preprocessors/  # Data preprocessing scripts
│   │   │   └── base_gnn_data_preprocessor.py  # Base data preprocessor
│   │   └── layers/           # Model layers
│   └── registry.py           # Registry for components
├── tests/                    # Unit tests
│   ├── datasets/             # Dataset tests
│   │   └── test_planetoid.py # Planetoid dataset tests
│   ├── models/               # Model tests
│   │   ├── cores/            # Core model tests
│   │   │   └── test_gcn.py   # GCN model tests
│   │   └── test_base_model.py  # Base model tests
│   └── runners/              # Runner tests
│       └── test_runner.py    # Runner tests
├── tools/                    # Utility scripts
│   └── train.py              # Training script
└── work_dirs/                # Working directories for experiments
    └── simple_config/        # Example working directory
```

## Configuration

The configuration file [`configs/simple_config.py`](configs/simple_config.py) defines the dataset, model, and training parameters. Here is an explanation of the key components:

1. **Dataset Configuration**:
   - Specifies the type of dataset (e.g., Planetoid) and its location.
   - Defines the dataset's root directory and name.
   - Example:
     ```python
     dataset = dict(
         type='Planetoid',
         root='data/Planetoid',
         name='Cora'
     )
     ```

2. **Model Core Configuration**:
   - Defines the core architecture of the GNN model, such as the type of GNN (e.g., GCN).
   - Specifies the input channels, hidden channels, number of classes, number of layers, and the type of layer used.
   - Example:
     ```python
     model_core = dict(
         type='GCN',
         in_channels=1433,
         hidden_channels=16,
         out_channels=7,
         num_layers=2,
         layer_type='GraphConv'
     )
     ```

3. **Model Configuration**:
   - Specifies the type of model and integrates the core configuration.
   - Example:
     ```python
     model = dict(
         type='GNNModel',
         core=model_core
     )
     ```

4. **Data Loaders**:
   - Configures the data loaders for training, validation, and testing datasets.
   - Example:
     ```python
     data_loaders = dict(
         train=dict(batch_size=64, shuffle=True),
         val=dict(batch_size=64, shuffle=False),
         test=dict(batch_size=64, shuffle=False)
     )
     ```

5. **Training Configuration**:
   - Defines training parameters such as whether to train by epoch and the maximum number of epochs.
   - Example:
     ```python
     training = dict(
         by_epoch=True,
         max_epochs=200
     )
     ```

6. **Evaluation Configuration**:
   - Specifies the metrics used for evaluating the model's performance during testing and validation.
   - Example:
     ```python
     evaluation = dict(
         metrics=['accuracy', 'f1']
     )
     ```

7. **Optimizer Configuration**:
   - Defines the optimizer type (e.g., AdamW) and its learning rate.
   - Example:
     ```python
     optimizer = dict(
         type='AdamW',
         lr=0.01
     )
     ```

## Model Logic

The logic behind the model in MMGraph is designed to be modular and flexible, allowing users to easily define and experiment with different GNN architectures. Here are the key components:

1. **Base Model**:
   - The base model class provides a common interface for all GNN models. It includes methods for initializing the model, forward propagation, and loss computation.
   - Example:
     ```python
     class GNNBaseModel(nn.Module):
         def __init__(self, core):
             super(GNNBaseModel, self).__init__()
             self.core = core

         def forward(self, x, edge_index):
             return self.core(x, edge_index)
     ```

2. **Core Model**:
   - The core model defines the main architecture of the GNN, such as the type of layers and their configurations. It is responsible for the actual computation performed by the model.
   - Example:
     ```python
     class GCN(nn.Module):
         def __init__(self, in_channels, hidden_channels, out_channels, num_layers, layer_type):
             super(GCN, self).__init__()
             self.layers = nn.ModuleList()
             self.layers.append(layer_type(in_channels, hidden_channels))
             for _ in range(num_layers - 2):
                 self.layers.append(layer_type(hidden_channels, hidden_channels))
             self.layers.append(layer_type(hidden_channels, out_channels))

         def forward(self, x, edge_index):
             for layer in self.layers:
                 x = layer(x, edge_index)
             return x
     ```
