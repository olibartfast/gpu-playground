#!/bin/bash

# Update package lists
apt-get update

# Install additional development tools that might not be in the base image
apt-get install -y \
    gdb \
    valgrind \
    htop \
    tree \
    python3-pip \
    python3-venv \
    bc

# Set up environment variables (they should already be set in the NGC image, but ensuring they're in bashrc)
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
echo 'export CUDA_HOME="/usr/local/cuda"' >> ~/.bashrc

# Verify CUDA installation
echo "CUDA installation verification:"
nvcc --version
nvidia-smi || echo "GPU not available in container (this is normal if not using --gpus flag)"

echo "Development environment setup complete!"
