#!/usr/bin/env python3
"""
Buffer Collector Utility

This script provides utilities to collect and save replay buffers using the l2f_actor.
These buffers can be used for offline reinforcement learning algorithms like BC or TD3BC.
"""

import os
import argparse
import datetime
import numpy as np
import torch
from tqdm import tqdm
import wandb

from tianshou.data import ReplayBuffer, Batch
from l2f_gym import Learning2Fly
from l2f_agent import ConvertedModel

class BufferCollector:
    """
    A utility class to collect and save replay buffers using the l2f_actor.
    
    This class provides methods to collect data from the environment using the l2f_actor
    and save it to a replay buffer for later use in offline reinforcement learning algorithms.
    """
    
    def __init__(
        self,
        env=None,
        controller=None,
        device="cpu",
        use_wandb=False,
        project_name="buffer_collection"
    ):
        """
        Initialize the BufferCollector.
        
        Args:
            env: The environment to collect data from. If None, a Learning2Fly environment will be created.
            controller: The controller to use for data collection. If None, the l2f_agent will be loaded.
            device: The device to use for tensor operations.
            use_wandb: Whether to log data to wandb.
            project_name: The name of the wandb project.
        """
        self.env = env if env is not None else Learning2Fly(fast_learning=False)
        self.device = device
        
        # Initialize controller
        if controller is None:
            self.controller = ConvertedModel()
            self.controller.load_state_dict(torch.load("l2f_agent.pth", map_location=device))
        else:
            self.controller = controller
            
        self.controller.to(device)
        
        # Initialize wandb if requested
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name, config={"device": device})
    
    def gather_buffer(
        self,
        size=1000,
        rollout_len=501,
        warmup=50,
        sequence_length=100,
        sequence_stride=50,
        min_valid_length=100,
        verbose=True
    ):
        """
        Gather a replay buffer using the controller.
        
        Args:
            size: The size of the buffer to collect.
            rollout_len: The length of each rollout.
            warmup: The number of warmup steps before using the controller.
            sequence_length: The length of each sequence to add to the buffer.
            sequence_stride: The stride between consecutive sequences.
            min_valid_length: The minimum length of a valid rollout.
            verbose: Whether to show progress bars and print information.
            
        Returns:
            The collected replay buffer.
        """
        buffer = ReplayBuffer(size=size)
        self.n_rollouts = int(size // (rollout_len / sequence_length))
        
        
        if verbose:
            print(f"Gathering buffer with {self.n_rollouts} rollouts...")
            rollout_iterator = tqdm(range(self.n_rollouts), desc="Gathering buffer")
        else:
            rollout_iterator = range(self.n_rollouts)
            
        total_env_steps = 0
        
        for _ in rollout_iterator:
            # Ensure we get usable sequences by discarding rollouts that crash too fast
            partial_rollout = True
            
            # Make sure to gather full rollouts
            while partial_rollout:
                # Create lists to store trajectory data
                obs_lst = []
                action_lst = []
                rewards_lst = []
                dones_lst = []
                obs_next_lst = []
                
                obs = self.env.reset()[0]
                partial_rollout = False  # Assume full rollout unless done before end of range
                t_warmup = 0  # Timestep in rollout (used after crash)
                
                for i in range(rollout_len):
                    obs_tensor = torch.tensor(obs, device=self.device)
                    
                    # Always use the controller for actions
                    action = self.controller(obs_tensor)
                    
                    # Store data - keep as tensor until final conversion
                    obs_lst.append(obs_tensor)
                    
                    # Step the environment
                    obs, rewards, dones, _, info = self.env.step(action.cpu().detach().numpy())
                    
                    obs_next_lst.append(obs)
                    action_lst.append(action.cpu().detach().numpy().reshape(4,))
                    rewards_lst.append(rewards)
                    dones_lst.append(dones)
                    
                    total_env_steps += 1
                    
                    if dones:
                        obs = self.env.reset()[0]
                        t_warmup = 0
                        
                        # If our rollout crashes before min_valid_length, we discard it
                        if i < min_valid_length:
                            partial_rollout = True
                            break
                        else:
                            # We have a valid partial rollout
                            break
                
                # If we have a valid rollout (either complete or partial but long enough)
                if not partial_rollout:
                    # Convert tensors to numpy only once for buffer storage
                    obs_np = torch.stack(obs_lst).cpu().numpy()
                    action_np = np.array(action_lst)
                    rewards_np = np.array(rewards_lst).reshape(-1, 1)
                    dones_np = np.array(dones_lst).reshape(-1, 1)
                    
                    # Stack observations, actions, rewards, and dones
                    obs_stack = np.hstack((
                        obs_np,
                        action_np,
                        rewards_np,
                        dones_np
                    ))
                    
                    # Chop up in sequence_length step sequences with stride of sequence_stride
                    for j in range(0, obs_stack.shape[0] - sequence_length, sequence_stride):
                        buffer.add(Batch({
                            'obs': obs_stack[j:j+sequence_length],
                            'act': np.array(action_lst[j+sequence_length-1]),
                            'rew': np.array(rewards_lst[j+sequence_length-1]),
                            'terminated': np.array(dones_lst[j+sequence_length-1]).reshape(-1, 1),
                            'truncated': np.array(dones_lst[j+sequence_length-1]).reshape(-1, 1)
                        }))
                    
                    # Add the last sequence if it's not already added
                    remaining = obs_stack.shape[0] % sequence_length
                    if remaining > 0 and obs_stack.shape[0] >= sequence_length:
                        buffer.add(Batch({
                            'obs': obs_stack[-sequence_length:],
                            'act': np.array(action_lst[-1]),
                            'rew': np.array(rewards_lst[-1]),
                            'terminated': np.array(dones_lst[-1]).reshape(-1, 1),
                            'truncated': np.array(dones_lst[-1]).reshape(-1, 1)
                        }))
        
        if self.use_wandb:
            wandb.log({'environment_interactions': total_env_steps})
            
        if verbose:
            print(f"Buffer collected with {len(buffer)} transitions")
            
        return buffer
    
    def save_buffer(self, buffer, filename=None):
        """
        Save the buffer to a file.
        
        Args:
            buffer: The buffer to save.
            filename: The filename to save the buffer to. If None, a timestamp will be used.
            
        Returns:
            The filename the buffer was saved to.
        """
        if filename is None:
            # timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            filename = f"buffers/l2f_buffer_{self.n_rollouts}.hdf5"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save the buffer
        buffer.save_hdf5(filename)
        
        if self.use_wandb:
            wandb.save(filename)
            
        return filename
    
    def collect_and_save(
        self,
        size=1000,
        rollout_len=501,
        warmup=50,
        sequence_length=100,
        sequence_stride=50,
        min_valid_length=100,
        filename=None,
        verbose=True
    ):
        """
        Collect and save a buffer in one operation.
        
        Args:
            size: The size of the buffer to collect.
            rollout_len: The length of each rollout.
            warmup: The number of warmup steps before using the controller.
            sequence_length: The length of each sequence to add to the buffer.
            sequence_stride: The stride between consecutive sequences.
            min_valid_length: The minimum length of a valid rollout.
            filename: The filename to save the buffer to. If None, a timestamp will be used.
            verbose: Whether to show progress bars and print information.
            
        Returns:
            The filename the buffer was saved to.
        """
        buffer = self.gather_buffer(
            size=size,
            rollout_len=rollout_len,
            warmup=warmup,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            min_valid_length=min_valid_length,
            verbose=verbose
        )
        
        return self.save_buffer(buffer, filename)


def main():
    """
    Main function to run the buffer collector from the command line.
    """
    parser = argparse.ArgumentParser(description="Collect and save replay buffers using the l2f_actor.")
    parser.add_argument("--size", type=int, default=10000, help="Size of the buffer to collect.")
    parser.add_argument("--rollout-len", type=int, default=501, help="Length of each rollout.")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup steps before using the controller.")
    parser.add_argument("--sequence-length", type=int, default=100, help="Length of each sequence to add to the buffer.")
    parser.add_argument("--sequence-stride", type=int, default=50, help="Stride between consecutive sequences.")
    parser.add_argument("--min-valid-length", type=int, default=100, help="Minimum length of a valid rollout.")
    parser.add_argument("--filename", type=str, default=None, help="Filename to save the buffer to.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for tensor operations.")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to log data to wandb.")
    parser.add_argument("--project-name", type=str, default="buffer_collection", help="Name of the wandb project.")
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU instead.")
        device = "cpu"
    
    # Create buffer collector
    collector = BufferCollector(
        device=device,
        use_wandb=args.use_wandb,
        project_name=args.project_name
    )
    
    # Collect and save buffer
    filename = collector.collect_and_save(
        size=args.size,
        rollout_len=args.rollout_len,
        warmup=args.warmup,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        min_valid_length=args.min_valid_length,
        filename=args.filename,
        verbose=True
    )
    
    print(f"Buffer saved to {filename}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 