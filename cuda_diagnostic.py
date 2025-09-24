#!/usr/bin/env python3
"""
CUDA Diagnostic Script for SpikingCrazyflie
This script helps diagnose CUDA utilization issues on cloud instances.
"""

import torch
import numpy as np
import time
import psutil
import subprocess
import os

def check_cuda_availability():
    """Check CUDA availability and device information."""
    print("=== CUDA Availability Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("CUDA not available!")
        return False
    return True

def check_nvidia_smi():
    """Check nvidia-smi output."""
    print("\n=== NVIDIA-SMI Check ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print(f"nvidia-smi failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("nvidia-smi timed out")
    except FileNotFoundError:
        print("nvidia-smi not found")

def test_gpu_computation():
    """Test basic GPU computation."""
    print("\n=== GPU Computation Test ===")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Create tensors on GPU
    print("Creating tensors on GPU...")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    # Perform computation
    print("Performing matrix multiplication...")
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Wait for GPU to finish
    end_time = time.time()
    
    print(f"Computation time: {end_time - start_time:.4f} seconds")
    print(f"Result shape: {c.shape}")
    print(f"Result device: {c.device}")
    print(f"Result dtype: {c.dtype}")

def test_model_on_gpu():
    """Test moving a simple model to GPU."""
    print("\n=== Model GPU Test ===")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping model test")
        return
    
    device = torch.device('cuda')
    
    # Create a simple model
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(18, 64),
        nn.ReLU(),
        nn.Linear(64, 4)
    )
    
    print(f"Model device before: {next(model.parameters()).device}")
    
    # Move to GPU
    model = model.to(device)
    print(f"Model device after: {next(model.parameters()).device}")
    
    # Test forward pass
    x = torch.randn(32, 18, device=device)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input device: {x.device}")
    print(f"Output device: {y.device}")
    print(f"Output shape: {y.shape}")

def monitor_gpu_usage():
    """Monitor GPU usage during computation."""
    print("\n=== GPU Usage Monitoring ===")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping monitoring")
        return
    
    device = torch.device('cuda')
    
    # Get initial memory usage
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device)
    print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
    
    # Create large tensors
    print("Creating large tensors...")
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000, device=device)
        tensors.append(tensor)
        current_memory = torch.cuda.memory_allocated(device)
        print(f"  After tensor {i+1}: {current_memory / 1024**2:.1f} MB")
    
    # Perform computation
    print("Performing computation...")
    start_time = time.time()
    result = torch.zeros(1000, 1000, device=device)
    for i, tensor in enumerate(tensors):
        result += tensor
        if i % 3 == 0:
            current_memory = torch.cuda.memory_allocated(device)
            print(f"  Computation step {i}: {current_memory / 1024**2:.1f} MB")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Computation time: {end_time - start_time:.4f} seconds")
    print(f"Final GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    # Clean up
    del tensors, result
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")

def check_system_resources():
    """Check system resources."""
    print("\n=== System Resources ===")
    print(f"CPU count: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # Check for CUDA libraries
    print("\n=== CUDA Libraries ===")
    cuda_libs = ['libcuda.so', 'libcudart.so', 'libcublas.so', 'libcurand.so']
    for lib in cuda_libs:
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if lib in result.stdout:
                print(f"✓ {lib} found")
            else:
                print(f"✗ {lib} not found")
        except:
            print(f"? {lib} check failed")

def main():
    """Run all diagnostic checks."""
    print("CUDA Diagnostic Script for SpikingCrazyflie")
    print("=" * 50)
    
    check_system_resources()
    check_cuda_availability()
    check_nvidia_smi()
    
    if torch.cuda.is_available():
        test_gpu_computation()
        test_model_on_gpu()
        monitor_gpu_usage()
    else:
        print("\n=== Recommendations ===")
        print("1. Check if NVIDIA drivers are installed: nvidia-smi")
        print("2. Check if CUDA toolkit is installed")
        print("3. Verify PyTorch CUDA version matches CUDA toolkit")
        print("4. Check if GPU is properly detected by the system")
    
    print("\n=== Diagnostic Complete ===")

if __name__ == "__main__":
    main()
