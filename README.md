# Spiking Crazyflie Gym

Welcome to the Spiking Crazyflie Gym repository! This repository contains a high-performance simulator designed for training end-to-end control of the Crazyflie 2.1 drone. The simulator provides a robust environment for developing and testing control algorithms.

In addition to the simulator, this repository includes training scripts for both artificial neural networks (ANNs) and spiking neural networks (SNNs). These scripts facilitate the training process, enabling you to develop effective control strategies for the Crazyflie 2.1 drone.

## Simulator
The simulator is based on the [Learning to Fly in Seconds](https://github.com/arplaboratory/learning-to-fly) project. The dynamics have been wrapped to follow the Gymnasium API. 
`l2f_gym.py` file contains the core implementation of the simulator, providing the necessary interfaces and dynamics to simulate the Crazyflie 2.1 drone within the Gymnasium framework. 

## Training Agents
Multiple training methods are available in this repository. 
### Online RL
First, the fully online RL methods can be found in the files `tianshou_l2f_<method>.py`. These are build on top of the Tianshou RL framework. 

First, SAC was used for training the networks. However, it was found that the entropy, which promotes exploration negatively influenced training performance. This is attributed to the fact that the environment is highly unstable and has random disturbances, which in itself already promotes exploration. Next, it was found that for SNN, the surrogate gradient slope needed to be much shallower than for similar supervised training rounds. This can be explained due to the fact that the steep surrogate slow more precisely updates towards the computed gradient direction, but updates fewer weights. Due to the inherent uncertainty of gradient direction in the initial training phases of RL, the steep gradient will significantly slow down training of the SNN.

### Online-Offline RL
As controllers for the Crazyflie are readily available, one wonders whether it can be used to kickstart the training of the efficient SNN. In `TD3BC_Online.py`, an implementation of TD3BC which uses Jump-Start RL is presented. One can choose any of the buffers available in this repo, or start with an empty replay buffer and an existing controller. 

During training, the agent initially uses the presented existing controller, and soflty rolls in the SNN, filling the buffer with more and more data gathered with the spiking actor. Several mechanisms are in place to avoid filling the buffer with useless data when early termination occurs.

### Offline RL
In the most basic sense, we can train an agent by copying the behavior available in the dataset. This boils down to supervised learning. In `BC.py`, the code to enable this training is presented.

When using an RL setup, with an actor-critic setup, we can leverage reward information to improve over our existing dataset. This can particularly be useful when the dataset is not expert data or when a (limited) reward curriculum is being used. You can refer to `TD3BC.py` for an implentation of such method.