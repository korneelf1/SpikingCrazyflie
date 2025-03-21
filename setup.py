#!/usr/bin/env python3
"""
Setup script for SpikingCrazyflie.

This script installs the SpikingCrazyflie package and its dependencies.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define package requirements
requirements = [
    "numpy>=1.24.4",
    "torch>=2.3.1",
    "gymnasium>=0.28.1",
    "snntorch>=0.9.1",
    "wandb>=0.17.3",
    "numba>=0.57.1",
    "tqdm>=4.66.4",
    "matplotlib>=3.8.4",
    "h5py>=3.11.0",
    "pandas>=2.2.2",
]

# Define development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "tensorboard>=2.17.0",
]

setup(
    name="spikingcrazyflie",
    version="0.1.0",
    author="Korneel Fiers",
    author_email="",  # Add your email if desired
    description="A high-performance simulator for training end-to-end control of the Crazyflie 2.1 drone using Spiking Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/korneelf1/SpikingCrazyflie",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": dev_requirements,
    },
    dependency_links=[
        "git+https://github.com/thu-ml/tianshou.git@master#egg=tianshou",
    ],
    entry_points={
        "console_scripts": [
            "spikingcrazyflie-collect-buffer=utils.buffer_collector:main",
        ],
    },
) 