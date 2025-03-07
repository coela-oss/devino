#!/bin/bash
set -e

# https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html

mkdir -p neo
cd neo
sudo wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.7.11/intel-igc-core-2_2.7.11+18581_amd64.deb
sudo wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.7.11/intel-igc-opencl-2_2.7.11+18581_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-level-zero-gpu-dbgsym_1.6.32567.17_amd64.ddeb
sudo wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-level-zero-gpu_1.6.32567.17_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-opencl-icd-dbgsym_25.05.32567.17_amd64.ddeb
sudo wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-opencl-icd_25.05.32567.17_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/libigdgmm12_22.6.0_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/ww05.sum
sudo sha256sum -c ww05.sum
sudo apt-get update
sudo apt-get install -y ocl-icd-libopencl1
sudo dpkg -i *.deb

cd ..
rm -rf neo

pip install --upgrade pip
pip install --user -r requirements.txt

echo "Setup complete!"
