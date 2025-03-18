#!/bin/bash
set -e  # Exit the script if any command fails

# The main stream for this setup script is following content by intel.
# https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html

# Get the current username
CURRENT_USER=$(whoami)

# ===== 1. Install Required Utilities =====
# Update system packages
sudo apt update
# Install essential development tools
sudo apt install -y \
        unzip \
        cmake \
        pkg-config \
        build-essential \
        gfortran \
        openmpi-bin \
        protobuf-compiler \
        ccache \
        musl-tools

# ===== 2. Install Bazel (System-wide) =====
wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
sudo bash bazel-5.3.0-installer-linux-x86_64.sh

# ===== 3. Install Poetry (Python package manager) =====
curl -sSL https://install.python-poetry.org | python3 -

# Set environment variables permanently
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Install packages
poetry install

# ===== 4. Install Intel OneAPI =====
# Add Intel GPG key
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# Add Intel OneAPI repository
# Other versions include this repository
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/oneAPI.list


# Obtain the repository key for Intel PyTorch GPU development tools
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor > /tmp/intel-for-pytorch-gpu-dev-keyring.gpg
sudo mv /tmp/intel-for-pytorch-gpu-dev-keyring.gpg /usr/share/keyrings
# Add repository for Intel PyTorch GPU development
echo "deb [signed-by=/usr/share/keyrings/intel-for-pytorch-gpu-dev-keyring.gpg] https://apt.repos.intel.com/intel-for-pytorch-gpu-dev all main" \
  | sudo tee /etc/apt/sources.list.d/intel-for-pytorch-gpu-dev.list

# Update package list and install Intel OneAPI core tools
sudo apt update && sudo apt -y upgrade
sudo apt -y install intel-basekit-2024.1 \
              intel-oneapi-mpi-2021.14 \
              intel-oneapi-mpi-devel-2021.14 \
              intel-oneapi-ccl-2021.14 \
              intel-oneapi-ccl-devel-2021.14 \
              intel-pti-dev

# ===== 5. User Configuration =====
# Add the current user to the render group
sudo usermod -a -G render "$CURRENT_USER"

# ===== 6. Install Intel GPU Drivers (for WSL2) =====
# Official guide: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/configure-wsl-2-for-gpu-workflows.html
# https://github.com/intel/compute-runtime/releases/tag/24.35.30872.22
sudo apt-get install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all

# Install GPU information utility
sudo apt-get install -y clinfo

# ===== 7. Install PyTorch for Intel GPU [Deprecated] =====
# Update package list and install Intel PyTorch GPU development tools
#sudo apt update
#sudo apt install -y intel-for-pytorch-gpu-dev-0.5 intel-pti-dev-0.9

# ===== 8. Clone and Configure PyTorch Source Code =====
# Clone the PyTorch source code and synchronize submodules
git clone --recursive -b v2.5.0-rc10 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive

# ===== 9. Activate Development Environment =====
# Set up environment variables for PyTorch with Intel GPU support
# https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#c-development-tips
# If notice to compile very slow, See OneDNN Warning issue.
# https://github.com/uxlfoundation/oneDNN/issues/559
# https://stackoverflow.com/questions/77921357/warning-while-using-tensorflow-tensorflow-core-util-port-cc113-onednn-custom
$(poetry env activate -C ~/devino/setup)
export USE_XPU=ON
export USE_NUMPY=0
export _GLIBCXX_USE_CXX11_ABI=1
# export USE_CUDA=ON # OFF, Not 0 And If set OFF, USE_XPU to OFF forcely.
export USE_NNPACK=0
export BUILD_TEST=0
make triton
source /opt/intel/oneapi/setvars.sh


# ===== 10. Setup Completion Message =====
echo "Setup completed successfully!"
echo "To build PyTorch for XPU, run: python setup.py build"
echo "Note: The build process can take a very long time. Please be patient."
