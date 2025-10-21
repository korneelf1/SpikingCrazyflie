#!/bin/bash

echo "==============================================="
echo "Setting up Python 3.11 virtual environment..."
echo "==============================================="
# Create virtual environment if it doesn't exist
if [ ! -d "venv_py311" ]; then
    echo "Creating Python 3.11 virtual environment..."
    python3.11 -m venv venv_py311
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_py311/bin/activate

# Upgrade pip in the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip

echo "✅ Virtual environment setup complete!"
echo ""

echo "==============================================="
echo "Cloning and installing tianshou (1.2.0-dev)..."
echo "==============================================="
git clone https://github.com/korneelf1/tianshou-1.2.0-dev.git || echo "Warning: tianshou repo already exists or failed to clone"
cd tianshou-1.2.0-dev 2>/dev/null || cd tianshou-1.2.0-dev || echo "Warning: could not cd into tianshou-1.2.0-dev"
echo "Installing tianshou in editable mode..."
pip install -e . || echo "Warning: pip install -e . for tianshou failed"
cd .. || echo "Warning: could not cd .. from tianshou-1.2.0-dev"
echo "✅ tianshou installation step complete!"
echo ""

echo "==============================================="
echo "Cloning and installing l2f_thesis package..."
echo "==============================================="
git clone https://github.com/korneelf1/l2f_thesis.git || echo "Warning: l2f_thesis repo already exists or failed to clone"
cd l2f_thesis 2>/dev/null || cd l2f_thesis || echo "Warning: could not cd into l2f_thesis"
git checkout last_working
echo "Initializing submodules for l2f_thesis..."
git submodule update --init --recursive external/rl-tools || echo "Warning: git submodule update failed"
echo "Installing l2f_thesis in editable mode..."
pip install -e . || echo "Warning: pip install -e . for l2f_thesis failed"
cd .. || echo "Warning: could not cd .. from l2f_thesis"
echo "✅ l2f_thesis installation step complete!"
echo ""

echo "==============================================="
echo "Installing Python requirements from requirements.txt..."
echo "==============================================="
pip install -r requirements.txt || echo "Warning: pip install -r requirements.txt failed"
echo "✅ requirements.txt installation step complete!"
echo ""

echo "==============================================="
echo "Downloading buffer data from HuggingFace..."
echo "==============================================="
hf download korneelf1/neurips --include "l2f_buffer_1996.hdf5" --local-dir buffers --repo-type dataset || echo "Warning: huggingface buffer download failed"
echo "✅ Buffer download step complete!"
echo ""

echo "==============================================="
echo "Running TD3BC_Online_SurrGrad_Experiment with virtual environment..."
echo "==============================================="
# Update the bash script to use the virtual environment's python
sed -i.bak "s|python |$PWD/venv_py311/bin/python |g" ./bash_scripts/TD3BC_Online_SurrGrad_Experiment.sh

./bash_scripts/TD3BC_Online_SurrGrad_Experiment.sh
echo ""