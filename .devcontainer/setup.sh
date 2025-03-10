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
