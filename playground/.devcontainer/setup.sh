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

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt installintel-oneapi-base-toolkit
