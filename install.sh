# #!/bin/bash

# # SpikingCrazyflie Installation Script
# # This script automates the installation of all dependencies and setup steps

# set -e  # Exit on any error

# # Colors for output
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

# # Function to print colored output
# print_status() {
#     echo -e "${BLUE}[INFO]${NC} $1"
# }

# print_success() {
#     echo -e "${GREEN}[SUCCESS]${NC} $1"
# }

# print_warning() {
#     echo -e "${YELLOW}[WARNING]${NC} $1"
# }

# print_error() {
#     echo -e "${RED}[ERROR]${NC} $1"
# }

# # Function to check if command exists
# command_exists() {
#     command -v "$1" >/dev/null 2>&1
# }

# # Function to check Python version
# check_python_version() {
#     if command_exists python3; then
#         PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
#         print_status "Found Python version: $PYTHON_VERSION"
        
#         # Check if version is >= 3.8
#         if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
#             print_success "Python version is compatible (>= 3.8)"
#             return 0
#         else
#             print_error "Python version $PYTHON_VERSION is not compatible. Please install Python 3.8 or higher."
#             return 1
#         fi
#     else
#         print_error "Python3 not found. Please install Python 3.8 or higher."
#         return 1
#     fi
# }

# # Function to create virtual environment
# create_venv() {
#     local venv_name=${1:-"venv"}
    
#     if [ -d "$venv_name" ]; then
#         print_warning "Virtual environment '$venv_name' already exists. Skipping creation."
#         return 0
#     fi
    
#     print_status "Creating virtual environment: $venv_name"
#     python3 -m venv "$venv_name"
#     print_success "Virtual environment created successfully"
# }

# # Function to activate virtual environment
# activate_venv() {
#     local venv_name=${1:-"venv"}
    
#     if [ -f "$venv_name/bin/activate" ]; then
#         print_status "Activating virtual environment: $venv_name"
#         source "$venv_name/bin/activate"
#         print_success "Virtual environment activated"
#     else
#         print_error "Virtual environment activation script not found at $venv_name/bin/activate"
#         exit 1
#     fi
# }

# # Function to upgrade pip
# upgrade_pip() {
#     print_status "Upgrading pip..."
#     pip install --upgrade pip
#     print_success "Pip upgraded successfully"
# }

# # Function to install requirements
# install_requirements() {
#     print_status "Installing requirements from requirements.txt..."
#     pip install -r requirements.txt
#     print_success "Requirements installed successfully"
# }

# # Function to install tianshou from GitHub
# install_tianshou() {
#     local tianshou_dir="tianshou-1.2.0-dev"
    
#     if [ -d "$tianshou_dir" ]; then
#         print_warning "tianshou-1.2.0-dev directory already exists. Updating..."
#         cd "$tianshou_dir"
#         git pull
#         pip install -e .
#         cd ..
#     else
#         print_status "Cloning tianshou-1.2.0-dev repository..."
#         cd ..
#     print_status "Installing tianshou from GitHub..."
#         print_status "Cloning tianshou-1.2.0-dev repository..."
#         git clone https://github.com/korneelf1/tianshou-1.2.0-dev.git
#         cd tianshou-1.2.0-dev
#         pip install -e .
#         cd ..
#     fi
#     print_success "Tianshou installed successfully"
# }

# # Function to install l2f package
# install_l2f() {
#     local l2f_dir="l2f_thesis"
    
#     if [ -d "$l2f_dir" ]; then
#         print_warning "l2f_thesis directory already exists. Updating..."
#         cd "$l2f_dir"
#         git checkout last_working
#         git pull
#         git submodule update --init --recursive external/rl-tools
#         cd external/rl-tools
#         mkdir -p build
#         cd build
#         cmake .. -DCMAKE_BUILD_TYPE=Release
#         make -j$(sysctl -n hw.ncpu) 
#         cd ..
#         python setup.py build_ext --inplace

#         # test examples/test.py
#         python examples/test.py
#     else
#         print_status "Cloning l2f_thesis repository..."
#         git clone https://github.com/korneelf1/l2f_thesis.git
#         git checkout last_working
#         git submodule update --init --recursive external/rl-tools
#         cd external/rl-tools
#         mkdir -p build
#         cd build
#         cmake .. -DCMAKE_BUILD_TYPE=Release
#         make -j$(sysctl -n hw.ncpu) 
#         cd ..
#         python setup.py build_ext --inplace

#         # test examples/test.py
#         python examples/test.py
#     fi
    

#     cd "$l2f_dir"
#     print_status "Installing l2f package..."
#     pip install -e .
#     cd ..
#     print_success "l2f package installed successfully"
# }

# # Function to download buffer data
# download_buffer() {
#     print_status "Downloading buffer data from Hugging Face..."
#     hf download korneelf1/neurips --include "l2f_buffer_1996.hdf5" --local-dir buffers --repo-type dataset
#     print_success "Buffer data downloaded successfully"
# }

# # Function to test installation
# test_installation() {
#     print_status "Testing installation..."
    
#     # Test if we can import the main modules
#     python3 -c "
# import sys
# try:
#     import torch
#     import gymnasium
#     import snntorch
#     import wandb
#     import l2f
#     print('✓ All core dependencies imported successfully')
# except ImportError as e:
#     print(f'✗ Import error: {e}')
#     sys.exit(1)
# "
    
#     # Test the gym environment if test file exists
#     if [ -f "test_gym_sim.py" ]; then
#         print_status "Running gym simulation test..."
#         python3 test_gym_sim.py
#         print_success "Gym simulation test passed"
#     else
#         print_warning "test_gym_sim.py not found, skipping gym test"
#     fi
    
#     print_success "Installation test completed successfully"
# }

# # Function to show usage
# show_usage() {
#     echo "Usage: $0 [OPTIONS]"
#     echo ""
#     echo "Options:"
#     echo "  --venv-name NAME     Name of virtual environment (default: venv)"
#     echo "  --no-venv           Skip virtual environment creation"
#     echo "  --no-test           Skip installation test"
#     echo "  --help              Show this help message"
#     echo ""
#     echo "Examples:"
#     echo "  $0                           # Install with default settings"
#     echo "  $0 --venv-name myenv        # Install with custom venv name"
#     echo "  $0 --no-venv                # Install without virtual environment"
#     echo "  $0 --no-test                # Install without running tests"
# }

# # Main installation function
# main() {
#     local venv_name="venv"
#     local create_venv_flag=true
#     local run_test_flag=true
    
#     # Parse command line arguments
#     while [[ $# -gt 0 ]]; do
#         case $1 in
#             --venv-name)
#                 venv_name="$2"
#                 shift 2
#                 ;;
#             --no-venv)
#                 create_venv_flag=false
#                 shift
#                 ;;
#             --no-test)
#                 run_test_flag=false
#                 shift
#                 ;;
#             --help)
#                 show_usage
#                 exit 0
#                 ;;
#             *)
#                 print_error "Unknown option: $1"
#                 show_usage
#                 exit 1
#                 ;;
#         esac
#     done
    
#     print_status "Starting SpikingCrazyflie installation..."
#     echo "================================================"
    
#     # Check Python version
#     if ! check_python_version; then
#         exit 1
#     fi
    
#     # Create and activate virtual environment if requested
#     if [ "$create_venv_flag" = true ]; then
#         create_venv "$venv_name"
#         activate_venv "$venv_name"
#     else
#         print_warning "Skipping virtual environment creation"
#     fi
    
#     # Upgrade pip
#     upgrade_pip
    
#     # Install requirements
#     install_requirements
    
#     # Install tianshou from GitHub
#     install_tianshou
    
#     # Install l2f package
#     install_l2f
    
#     # Download buffer data
#     download_buffer
    
#     # Test installation
#     if [ "$run_test_flag" = true ]; then
#         test_installation
#     else
#         print_warning "Skipping installation test"
#     fi
    
#     echo "================================================"
#     print_success "Installation completed successfully!"
    
#     if [ "$create_venv_flag" = true ]; then
#         echo ""
#         print_status "To activate the virtual environment in the future, run:"
#         echo "  source $venv_name/bin/activate"
#         echo ""
#         print_status "To deactivate the virtual environment, run:"
#         echo "  deactivate"
#     fi
    
#     echo ""
#     print_status "You can now start using SpikingCrazyflie!"
#     print_status "Check the README.md for usage examples and training scripts."
    
#     # Activate the virtual environment at the end if it was created
#     if [ "$create_venv_flag" = true ]; then
#         print_status "Activating virtual environment for current session..."
#         source "$venv_name/bin/activate"
#         print_success "Virtual environment is now active!"
#     fi
# }

# # Run main function with all arguments
# main "$@"

# install tianshou
# install tianshou
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
