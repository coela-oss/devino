FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    cmake \
    gnupg \
    clinfo \
    ocl-icd-libopencl1 \
    python3-full \
    && sudo rm -rf /var/lib/apt/lists/*

# Create temporary directory
WORKDIR /workspaces/tmp

# Download Intel GPU driver and OpenCL packages
# https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-gpu.html
RUN wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.7.11/intel-igc-core-2_2.7.11+18581_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.7.11/intel-igc-opencl-2_2.7.11+18581_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-level-zero-gpu-dbgsym_1.6.32567.17_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-level-zero-gpu_1.6.32567.17_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-opencl-icd-dbgsym_25.05.32567.17_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/intel-opencl-icd_25.05.32567.17_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/libigdgmm12_22.6.0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.05.32567.17/ww05.sum

# Verify checksum
RUN sha256sum -c ww05.sum

# Install downloaded packages
RUN sudo dpkg -i *.deb

# Cleanup
WORKDIR /workspaces
RUN rm -rf /workspaces/tmp

RUN python3 -m venv devino
# && \
#    ./devino/bin/activate && \
#    pip install --upgrade pip    

WORKDIR /workspaces
