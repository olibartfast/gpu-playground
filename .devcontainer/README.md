# CUDA Development Container

This devcontainer provides a complete CUDA development environment based on NVIDIA's NGC CUDA image.

## Features

- **Base Image**: NVIDIA CUDA 13.0.0 development image from NGC (`nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`)
- **GPU Support**: Configured with `--gpus=all` for GPU access
- **Pre-installed Tools in NGC Image**: 
  - CUDA Toolkit 13.0.0
  - GCC/G++ compilers
  - CMake
  - Git, wget, curl, and other utilities
- **Additional Tools Added**: GDB, Valgrind, htop, Python tools
- **VS Code Extensions**:
  - C++ Tools Extension Pack
  - CMake Tools
  - NVIDIA Nsight Visual Studio Code Edition
  - Python and Jupyter support

## Requirements

1. **Docker**: Ensure Docker is installed and running
2. **NVIDIA Container Toolkit**: Required for GPU access in containers
3. **VS Code**: With the Dev Containers extension installed

## Setup Instructions

### Installing NVIDIA Container Toolkit (if not already installed)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Using the Dev Container

1. Open this project in VS Code
2. When prompted, click "Reopen in Container" or use Command Palette: `Dev Containers: Reopen in Container`
3. The container will build and set up the development environment automatically

## Verifying GPU Access

Once inside the container, you can verify GPU access with:

```bash
nvidia-smi
nvcc --version
```

## Building CUDA Projects

The environment is pre-configured for CMake-based CUDA projects. You can build the existing projects with:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Included Sample Code

The setup script automatically downloads NVIDIA CUDA samples to `/workspace/cuda-samples` for reference and learning.

## Environment Variables

The following environment variables are automatically set:

- `CUDA_HOME=/usr/local/cuda`
- `PATH` includes `/usr/local/cuda/bin`
- `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64`

## Troubleshooting

### GPU Not Available
If you see "GPU not available" messages, ensure:
1. NVIDIA drivers are installed on the host
2. NVIDIA Container Toolkit is properly installed
3. Docker daemon has been restarted after installing the toolkit

### Build Issues
- Ensure CMake version is 3.20 or higher (automatically installed in the container)
- Check that CUDA toolkit paths are correctly set in your CMakeLists.txt
