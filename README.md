
# Learning, sleep replay, and consolidation of contextual fear memories: A neural network model

**Authors**: Lars Werne et al. (2024-2025)

**Citation**: *(Manuscript currently under review)*

## Overview

This repository contains the Python implementation of the neural network model presented in our manuscript, exploring the computational mechanisms underlying fear memory formation, consolidation, extinction, and renewal. The model integrates Bayesian Confidence Propagation Neural Networks (BCPNNs), k-Winner-Takes-All (kWTA) networks, and novel binary activation modules to simulate learning and memory processes in cortical and amygdala circuits.

### Model Structure

- **BCPNN Modules**: Represent cortical computations with soft competition dynamics.
- **kWTA Modules**: Enforce hard competition to maintain sparse activations.
- **Binary Modules**: Represent populations of neurons in amygdala regions with binary activations.
- **Threshold Cells**: Simulate specific single-unit responses using sigmoid activation.

## Repository Contents

- `model/`: Contains the neural network classes defining modules (`BCPNN`, `kWTA`, `BinaryModule`, `ThresholdCell`, etc.), and the model architecture (`AmygdalaEngrams` in file model.py).
- `utils/`: Helper scripts, including functions for generating patterns, computing similarity metrics, and managing simulation phases.
- `requirements.txt`: Lists required Python packages.

## Installation and Setup

### Prerequisites

Python version **3.8 or later** is recommended.

### Dependencies

To install dependencies, navigate to the project root directory and run:

```bash
pip install -r requirements.txt
```

## Usage

### Running Simulations

Navigate to the repository's root folder. You can run scripts directly from the command line, for example:

```bash
python run_simulation.py
```

Make sure you activate your Python environment beforehand.

### Reproducing Figures

To reproduce figures from the paper, execute the corresponding methods in `run_simulation.py`. Each method corresponds to a figure or analysis presented in our manuscript.

## Folder Structure

```
project_root/
├── model/
│   ├── __init__.py
│   ├── amy_engrams.py
│   ├── binary_module.py
│   ├── kWTA.py
│   ├── BCPNN.py
│   ├── threshold_cell.py
│   ├── feedback.py
│   └── phase.py
├── utils/
│   ├── __init__.py
│   └── util.py
├── run_simulation.py
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

Please cite our work when using or adapting this code for your own research.
