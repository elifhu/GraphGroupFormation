# Fair and Skill-Diverse Student Group Formation: A Graph-Theoretic Approach

## About
Code for *Fair and Skill-Diverse Student Group Formation: A Graph-Theoretic Approach*, IEEE Signal Processing Magazine.

This repository contains implementations of group formation algorithms using Variable Neighborhood Search (VNS) and exact optimization methods. The code is structured to be flexible and adaptable for different datasets while maintaining a focus on educational group formation.

## Installation
Install the package requirements to your environment
```bash
pip install -r requirements.txt
```

## Contents
- `main.py`: Entry point for running experiments, using configuration files for parameter settings
- `src/`
  - `heuristic/vns.py`: Contains two main optimizers:
    - `VNSGroupOptimizer`: Implementation of Variable Neighborhood Search heuristic for group formation
    - `ExactOptimizer`: Brute force search implementation for small problem instances
  - `data/`: Data loaders and processing utilities
  - `eigenmap/`: Utils for computing the Laplacian eigenmap from provided tabular data
  - `analysis/`: Utils for computing performance statistics for algorithm runs
- `configs/`: Configuration files for different experimental settings
- `processed_data/`: Directory for storing processed dataset files
- `outputs/`: Directory for experimental outputs
- `multirun/`: Directory for parallel experiment runs

## Usage
The code below runs our experiments on simulated data.

Circle experiment: finding the optimal solution with and without balance constraint.
```bash
python main.py --multirun --config-name config_circle_sweep
```

Simulated marks data: how diversity depends on group size and number of affinities.
```bash
python main.py --multirun --config-name config_simulated_sweep
```

## Custom Datasets
To use this codebase with your own dataset, you need to:

1. Create a data loader class that implements the `.load_all()` method in `data/data_loader.py`
2. Your data loader should return:
   - `q`: Student feature vectors encoding the `skills` (eigenmap dimensions)
   - `s`: Sensitive attributes matrix
3. Configure your experiment parameters in a config file

## Contributing
For questions about the code or paper, please open an issue in this repository.
