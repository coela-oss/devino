#!/bin/bash
set -e

# https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html
#wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
#sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
#echo "deb https://apt.repos.intel.com/openvino/2025 ubuntu24 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2025.list
#sudo apt update
#sudo apt install -y openvino

#cd /workspaces
#python3 -m venv devino
#cd devino
#source bin/activate

# Pytorch and Tensorflow optimized intel cpu
#pip install --upgrade pip
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip install intel-extension-for-pytorch oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
#pip install -r requirements.txt
#pip install intel-tensorflow

echo "Setup complete!"


# https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop-24-10
# Install the Intel graphics GPG public key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# Configure the repositories.intel.com package repository
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

# Update the package repository metadata
sudo apt update

# Install the compute-related packages
apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-gsc libze-dev intel-ocloc


# Need Ubuntu 22 kernel 6.5,
sudo apt-cache search linux-image-6.*-generic
export VERSION="6.8.0-55"
sudo apt-get install -y linux-headers-$VERSION-generic linux-image-$VERSION-generic linux-modules-extra-$VERSION-generic

sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1> $(echo $(($(awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg \
| grep -no $VERSION | sed 's/:/\n/g' | head -n 1)-2)))\"/" /etc/default/grub

sudo update-grub

# if use ubuntu 24, set this flag to work 6.8 (not work 6.11)
# https://github.com/intel/compute-runtime/issues/710#issuecomment-2002646557
export NEOReadDebugKeys=1
export OverrideGpuAddressSpace=48



wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

sudo apt update

sudo apt-get install -y libze1 intel-level-zero-gpu intel-opencl-icd clinfo
sudo apt -y install cmake pkg-config build-essential
```
import openvino as ov
core = ov.Core()
core.available_devices
>> ['CPU']
```


# Prerequired
# Utils
apt install unzip
apt -y install cmake pkg-config build-essential

# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# Bazel
wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
bash bazel-5.3.0-installer-linux-x86_64.sh --user

# Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/vscode/.local/bin:$PATH"


# sudo apt-cache search 2024
# latest version not compatitable to pytorch build step
# Step. 1
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt -y upgrade
sudo apt -y install intel-basekit-2024.1 \
              intel-oneapi-mpi-2021.14 \
              intel-oneapi-mpi-devel-2021.14 \
              intel-oneapi-ccl-2021.14 \
              intel-oneapi-ccl-devel-2021.14

# Step.2
usermod -a -G render vscode

#Step.6 Driver install
# https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/configure-wsl-2-for-gpu-workflows.html
sudo apt-get install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all

sudo apt-get install -y clinfo
# not intel-deep-learning-essentials-2025
sudo apt install intel-deep-learning-essentials 

# Tensorflow
#git clone https://github.com/intel/intel-extension-for-tensorflow.git
# https://github.com/intel/intel-extension-for-tensorflow/blob/main/examples/common_guide_running.md#prepare

# Pytorch # **Version intel-oneapi 2024.1 **
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor > /tmp/intel-for-pytorch-gpu-dev-keyring.gpg

sudo mv /tmp/intel-for-pytorch-gpu-dev-keyring.gpg /usr/share/keyrings
echo "deb [signed-by=/usr/share/keyrings/intel-for-pytorch-gpu-dev-keyring.gpg] https://apt.repos.intel.com/intel-for-pytorch-gpu-dev all main" > /tmp/intel-for-pytorch-gpu-dev.list
sudo mv /tmp/intel-for-pytorch-gpu-dev.list /etc/apt/sources.list.d
sudo apt update
sudo apt install intel-for-pytorch-gpu-dev-0.5 intel-pti-dev-0.9


git clone --recursive -b v2.5.0-rc10 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
cd ~/devino/setup

# Activate Setup Environment
# https://github.com/pytorch/pytorch/tree/v2.5.0-rc10?tab=readme-ov-file#intel-gpu-support
$(poetry env activate)
export USE_XPU=1
export _GLIBCXX_USE_CXX11_ABI=1
source /opt/intel/oneapi/setvars.sh

cd ~/pytorch/
make triton

