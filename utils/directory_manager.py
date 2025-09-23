"""
Directory management utilities for SpikingCrazyflie project.
Provides consistent directory structure for figures and checkpoints.
"""

import os
from pathlib import Path
from typing import Optional


def ensure_directories_exist(base_path: Optional[str] = None) -> tuple[str, str]:
    """
    Ensure that figures and checkpoint directories exist.
    
    Args:
        base_path: Base path for directories. If None, uses current working directory.
        
    Returns:
        Tuple of (figures_dir, checkpoint_dir) paths
    """
    if base_path is None:
        base_path = os.getcwd()
    
    figures_dir = os.path.join(base_path, "figures")
    checkpoint_dir = os.path.join(base_path, "checkpoint")
    
    # Create directories if they don't exist
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return figures_dir, checkpoint_dir


def get_figures_path(filename: str, base_path: Optional[str] = None) -> str:
    """
    Get the full path for a figure file in the figures directory.
    
    Args:
        filename: Name of the figure file
        base_path: Base path for directories. If None, uses current working directory.
        
    Returns:
        Full path to the figure file
    """
    figures_dir, _ = ensure_directories_exist(base_path)
    return os.path.join(figures_dir, filename)


def get_checkpoint_path(filename: str, base_path: Optional[str] = None) -> str:
    """
    Get the full path for a checkpoint file in the checkpoint directory.
    
    Args:
        filename: Name of the checkpoint file
        base_path: Base path for directories. If None, uses current working directory.
        
    Returns:
        Full path to the checkpoint file
    """
    _, checkpoint_dir = ensure_directories_exist(base_path)
    return os.path.join(checkpoint_dir, filename)


def save_figure(fig, filename: str, base_path: Optional[str] = None, 
                formats: list[str] = None, **kwargs) -> list[str]:
    """
    Save a matplotlib figure to the figures directory with multiple formats.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        base_path: Base path for directories. If None, uses current working directory.
        formats: List of formats to save (default: ['png', 'pdf', 'svg'])
        **kwargs: Additional arguments passed to savefig
        
    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ['png', 'pdf', 'svg']
    
    figures_dir, _ = ensure_directories_exist(base_path)
    saved_paths = []
    
    for fmt in formats:
        filepath = os.path.join(figures_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, format=fmt, **kwargs)
        saved_paths.append(filepath)
    
    return saved_paths


def save_checkpoint(model_state_dict: dict, filename: str, base_path: Optional[str] = None) -> str:
    """
    Save a model checkpoint to the checkpoint directory.
    
    Args:
        model_state_dict: Model state dictionary to save
        filename: Name of the checkpoint file
        base_path: Base path for directories. If None, uses current working directory.
        
    Returns:
        Path to the saved checkpoint file
    """
    import torch
    
    checkpoint_path = get_checkpoint_path(filename, base_path)
    torch.save(model_state_dict, checkpoint_path)
    return checkpoint_path


