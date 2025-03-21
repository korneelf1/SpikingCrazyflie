#!/usr/bin/env python3
"""
Example script demonstrating how to use the buffer collector.

This script shows how to collect and save a replay buffer using the l2f_actor,
and then how to load and use it with BC or TD3BC algorithms.
"""

import os
import torch
import numpy as np
from buffer_collector import BufferCollector
from tianshou.data import ReplayBuffer

def collect_buffer_example():
    """
    Example of collecting and saving a buffer.
    """
    print("Creating buffer collector...")
    
    # Determine the best device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    # Create buffer collector
    collector = BufferCollector(
        device=device,
        use_wandb=False  # Set to True if you want to log to wandb
    )
    
    # Collect a small buffer for demonstration purposes
    print("Collecting buffer...")
    buffer = collector.gather_buffer(
        size=500,  # Small size for demonstration
        rollout_len=501,
        sequence_length=100,
        sequence_stride=50,
        min_valid_length=100,
        verbose=True
    )
    
    print(f"Buffer collected with {len(buffer)} transitions")
    
    # Save the buffer
    print("Saving buffer...")
    filename = collector.save_buffer(buffer, "example_buffer.hdf5")
    
    print(f"Buffer saved to {filename}")
    
    return filename

def load_buffer_example(filename):
    """
    Example of loading a buffer.
    
    Args:
        filename: The filename of the buffer to load.
    """
    print(f"Loading buffer from {filename}...")
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    
    # Load the buffer
    buffer = ReplayBuffer.load_hdf5(filename)
    
    print(f"Buffer loaded with {len(buffer)} transitions")
    
    # Print some statistics about the buffer
    if len(buffer) > 0:
        batch = buffer[0]
        print(f"Observation shape: {batch.obs.shape}")
        print(f"Action shape: {batch.act.shape}")
        print(f"Reward: {batch.rew}")
        
        # Calculate average reward
        rewards = np.array([buffer[i].rew for i in range(len(buffer))])
        print(f"Average reward: {rewards.mean():.4f}")
        print(f"Min reward: {rewards.min():.4f}")
        print(f"Max reward: {rewards.max():.4f}")
    
    return buffer

def use_buffer_with_bc_example(buffer):
    """
    Example of how to use the buffer with BC.
    
    Args:
        buffer: The buffer to use.
    """
    print("This is where you would use the buffer with BC.")
    print("For example:")
    print("  from BC import BehavioralCloning")
    print("  bc = BehavioralCloning(buffer=buffer, ...)")
    print("  bc.learn()")

def use_buffer_with_td3bc_example(buffer):
    """
    Example of how to use the buffer with TD3BC.
    
    Args:
        buffer: The buffer to use.
    """
    print("This is where you would use the buffer with TD3BC.")
    print("For example:")
    print("  from TD3BC import TD3BC")
    print("  td3bc = TD3BC(buffer=buffer, ...)")
    print("  td3bc.learn()")

if __name__ == "__main__":
    # Collect and save a buffer
    filename = collect_buffer_example()
    
    # Load the buffer
    buffer = load_buffer_example(filename)
    
    if buffer is not None:
        # Example of using the buffer with BC
        use_buffer_with_bc_example(buffer)
        
        # Example of using the buffer with TD3BC
        use_buffer_with_td3bc_example(buffer) 