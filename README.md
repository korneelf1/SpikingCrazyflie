# Spiking Crazyflie Gym

Welcome to the Spiking Crazyflie Gym repository! This repository contains a high-performance simulator designed for training end-to-end control of the Crazyflie 2.1 drone. The simulator provides a robust environment for developing and testing control algorithms.

In addition to the simulator, this repository includes training scripts for both artificial neural networks (ANNs) and spiking neural networks (SNNs). These scripts facilitate the training process, enabling you to develop effective control strategies for the Crazyflie 2.1 drone.

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/korneelf1/SpikingCrazyflie.git
cd SpikingCrazyflie
```

2. Create a new virtual environment:
```bash
# Using venv (Python's built-in)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR venv\Scripts\activate  # On Windows

# OR using conda
conda create -n spiking_cf python=3.11
conda activate spiking_cf
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. Install the tianshou package from github:
```bash
pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
```

5. install my l2f gym package from github:
```bash
git clone https://github.com/korneelf1/l2f_thesis.git
cd l2f_thesis
pip install -e .
```

6. Test the environment:
```bash
python test_env.py
```

If the test runs successfully, you'll see information about the environment's action and observation spaces, and a confirmation that basic operations work.

## Simulator
The simulator is based on the [Learning to Fly in Seconds](https://github.com/arplaboratory/learning-to-fly) project. The dynamics have been wrapped to follow the Gymnasium API. 
`l2f_gym.py` file contains the core implementation of the simulator, providing the necessary interfaces and dynamics to simulate the Crazyflie 2.1 drone within the Gymnasium framework. 

NOTE: as of Nov 2024, the learning to fly package supports CUDA operations, which can greatly increase the speed of the simulator!

## Training Agents
To use the training methods in this repository, you'll need to install tianshou directly from GitHub to ensure you have access to all required features:

```bash
pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
```

This installation method is required to access the high-level package features used by our logger.

### Online RL
First, the fully online RL methods can be found in the files `tianshou_l2f_<method>.py`. These are build on top of the Tianshou RL framework. 

First, SAC was used for training the networks. However, it was found that the entropy, which promotes exploration negatively influenced training performance. This is attributed to the fact that the environment is highly unstable and has random disturbances, which in itself already promotes exploration. Next, it was found that for SNN, the surrogate gradient slope needed to be much shallower than for similar supervised training rounds. This can be explained due to the fact that the steep surrogate slow more precisely updates towards the computed gradient direction, but updates fewer weights. Due to the inherent uncertainty of gradient direction in the initial training phases of RL, the steep gradient will significantly slow down training of the SNN.

### Online-Offline RL
As controllers for the Crazyflie are readily available, one wonders whether it can be used to kickstart the training of the efficient SNN. In `TD3BC_Online.py`, an implementation of TD3BC which uses Jump-Start RL is presented. One can choose any of the buffers available in this repo, or start with an empty replay buffer and an existing controller. 

During training, the agent initially uses the presented existing controller, and soflty rolls in the SNN, filling the buffer with more and more data gathered with the spiking actor. Several mechanisms are in place to avoid filling the buffer with useless data when early termination occurs.

### Offline RL
In the most basic sense, we can train an agent by copying the behavior available in the dataset. This boils down to supervised learning. In `BC.py`, the code to enable this training is presented.

When using an RL setup, with an actor-critic setup, we can leverage reward information to improve over our existing dataset. This can particularly be useful when the dataset is not expert data or when a (limited) reward curriculum is being used. You can refer to `TD3BC.py` for an implentation of such method.

### Explanation of the code
We take as example the `TD3BC_Online.py` file.

Looking at the main function, we first define the arguments we will use.

As TD3BC is an online-offline method, we need to define the controller that will be used to collect data. In this case, we use a pre-trained controller. (This can be any controller, not necessarily a spiking one.) We can also load a buffer which is used to kickstart the training, or let the controller collect data for a while before the training starts.

Then we create the environment, the buffer and the model. After initializing WandB, we initialize the spiking module and the wrapper.

The wrapper is used to convert the output of the spiking module to a continuous action.

The controller is then initialized and the TD3BC object is created.

The TD3BC object is initialized with the environment, the model, the optimizer, the buffer, the batch size, the device, the curriculum, the bc value and the bc factor.

The `run` function is then called, which will start the training. The `jumpstart` argument is used to start the training with the existing controller. This means that the (pre-trained, non-optimal) controller will be used to collect data for a while before the training starts.

# SpikingActorProb

A PyTorch implementation of Spiking Neural Networks (SNNs) for Deep Reinforcement Learning, specifically designed for continuous action spaces.

## Overview

This module provides spiking neural network implementations for reinforcement learning tasks, with a focus on continuous control problems. It extends the Tianshou RL framework with spiking neural network capabilities using the SNNTorch library.

The main components include:

- `SMLP`: A spiking multi-layer perceptron implementation
- `SlopeScheduler`: A scheduler for the surrogate gradient slope
- `SpikingNet`: A base spiking network for DRL usage
- `Actor`: A deterministic actor network for continuous action spaces
- `ActorProb`: A probabilistic actor network for continuous action spaces

## Spiking Neural Networks

Spiking Neural Networks (SNNs) are biologically inspired neural networks that mimic the behavior of biological neurons. Unlike traditional artificial neural networks, SNNs process information using discrete spikes rather than continuous values. This implementation uses the Leaky Integrate-and-Fire (LIF) neuron model from SNNTorch.

## Surrogate Gradient and Slope Scheduling

Since spiking neurons have a non-differentiable activation function (the spike), training SNNs with backpropagation requires a surrogate gradient function. This implementation uses the fast sigmoid surrogate gradient function from SNNTorch.

The slope of the surrogate gradient is a critical hyperparameter that affects the learning dynamics. A steeper slope provides a more accurate approximation of the spike function but can lead to vanishing gradients, while a shallower slope provides more gradient information but is less accurate.

### Slope Scheduling Types

The module implements three types of slope scheduling:

1. **Fixed Scheduling**
   - The slope remains constant throughout training
   - Simplest approach but may not adapt well to different training phases
   - Useful when a known, fixed slope value works well for the entire training process

2. **Interval Scheduling**
   - The slope is increased at fixed intervals (epochs)
   - Gradually increases the slope according to a predefined schedule
   - Formula: `slope = slope_init + 50 * epoch / 100`
   - Allows for a smooth transition from easier learning (shallow slope) to more accurate spike approximation (steep slope)
   - Updates occur every `update_interval` epochs after the `start_epoch`
   - **New 3rd Order Calculation**: `slope = slope_init + a*(epoch/100)³ + b*(epoch/100)² + c*(epoch/100)`
     - Provides more nuanced control over slope progression
     - Allows for initial slow increase, followed by faster middle-phase increase, and finally a gradual approach to maximum value
     - Parameters a, b, and c can be tuned to achieve desired progression curve
     - Better matches the learning dynamics of complex control tasks

3. **Adaptive Scheduling**
   - The slope is adjusted based on the agent's performance (reward)
   - Dynamically adapts the slope according to how well the agent is performing
   - Maps the normalized reward to a slope value within the range [0, max_slope]
   - Formula: `slope = reward_min + (score - reward_min)/(reward_max - reward_min) * max_slope`
   - **New 3rd Order Calculation**: `slope = min_slope + a*(norm_reward)³ + b*(norm_reward)² + c*(norm_reward)`
     - Where `norm_reward = (score - reward_min)/(reward_max - reward_min)`
     - Provides more sophisticated mapping between performance and slope
     - Allows for fine-grained control over how quickly slope increases with performance
     - Can be configured to be more conservative at lower performance levels and more aggressive at higher levels
   - Provides automatic adaptation based on learning progress
   - Particularly useful when the optimal slope may vary during different phases of learning

## Usage

The module is designed to be used with the TD3BC_Online reinforcement learning algorithm. The slope scheduling can be configured by setting the appropriate parameters in the SpikingNet constructor:

```python
# Example usage with fixed scheduling
model = SpikingNet(
    state_shape=state_dim,
    action_shape=action_dim,
    hidden_sizes=[256, 256],
    device=device,
    slope=10.0,
    schedule='fixed'
)

# Example usage with interval scheduling
model = SpikingNet(
    state_shape=state_dim,
    action_shape=action_dim,
    hidden_sizes=[256, 256],
    device=device,
    slope=10.0,
    schedule='interval',
    update_interval=100
)

# Example usage with adaptive scheduling
model = SpikingNet(
    state_shape=state_dim,
    action_shape=action_dim,
    hidden_sizes=[256, 256],
    device=device,
    slope=10.0,
    schedule='adaptive',
    reward_range=(0, 1),
    max_slope=100
)
```

The slope scheduler is updated during the model's reset method, which is called at the beginning of each learning batch in the TD3BC_Online algorithm.





