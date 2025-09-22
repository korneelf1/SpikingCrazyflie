# Training Examples

This directory contains example scripts for training different algorithms in the SpikingCrazyflie project.

## Available Examples

### 1. Behavioral Cloning (BC)
- **File**: `train_bc.py`
- **Description**: Example script for training using Behavioral Cloning
- **Usage**: `python examples/train_bc.py --slope 2 --hidden-sizes 256 128 --slope_schedule adaptive`

### 2. TD3+BC
- **File**: `train_td3bc.py`
- **Description**: Example script for training using TD3+BC algorithm
- **Usage**: `python examples/train_td3bc.py --slope 2 --hidden-sizes 256 256 --slope_schedule adaptive --curriculum`

### 3. TD3+BC Online
- **File**: `train_td3bc_online.py`
- **Description**: Example script for training using TD3+BC Online algorithm
- **Usage**: `python examples/train_td3bc_online.py --slope 2 --hidden-sizes 256 256 --slope_schedule adaptive --curriculum --bc-val 0.2 --jumpstart`

## Common Arguments

All training scripts support the following common arguments:

- `--slope`: Slope value for spiking neural network (default: 2)
- `--slope_schedule`: Schedule type for slope adaptation ('adaptive', 'interval', 'fixed')
- `--scheduling_order`: Order for scheduling (default: 3)
- `--hidden-sizes`: Hidden layer sizes (default: [256, 128])
- `--curriculum`: Enable curriculum learning
- `--bc-val`: BC coefficient value (for TD3+BC Online)
- `--bc-factor`: BC factor for decay (for TD3+BC Online)
- `--jumpstart`: Enable jumpstart phase (for TD3+BC Online)

## Requirements

Make sure you have the required dependencies installed and the data files in the correct locations:

- `buffers/l2f_buffer_1996.hdf5` - Training data buffer
- All required Python packages from `requirements.txt`
