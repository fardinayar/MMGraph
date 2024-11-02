# MMGraph

MMGraph is a high-level library designed to simplify the process of training Graph Neural Networks (GNNs). It provides a flexible and efficient framework for researchers and practitioners working with graph-structured data.

## Structure

MMGraph uses MMEngine for handling training and leverages a modular architecture for flexibility and extensibility. The project is structured as follows:

### Core Components

1. **configs**: Contains configuration files for various experiments and models.
   - `simple_config.py`: Defines basic configuration settings.

2. **mmgraph**: The main package containing the core functionality.
   - `registry.py`: Manages component registration for easy access and use.
   
   a. **datasets**: Handles data loading and preprocessing.
      - `planetoid.py`: Implementation for the Planetoid dataset.
   
   b. **evaluation**: Contains evaluation metrics and procedures.
      - `metrics/classification.py`: Defines classification-specific metrics.
   
   c. **models**: Implements various GNN models and related components.
      - `base_model.py`: Defines the base class for all models.
      - `data_preprocessors/base_gnn_data_preprocessor.py`: Base class for GNN data preprocessing.
      - `heads/gcn.py`: Implementation of the Graph Convolutional Network (GCN) head.
      - `layers/`: Contains implementations of different GNN layers.
   
   d. **runners**: Manages the training and evaluation loops.
      - `runner.py`: Implements the main training and evaluation runner.

3. **tests**: Contains unit tests for various components of the library.

### Key Features

- Modular design allowing easy extension and customization of components.
- Integration with MMEngine for efficient training management.
- Support for popular GNN models and datasets from PyG.
- Flexible configuration system for easy experiment setup.

## Getting Started

TODO

## Contributing

TODO

## License

TODO