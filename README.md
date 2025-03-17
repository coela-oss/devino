# Project Overview

This repository provides an environment for leveraging OpenVINO and OneAPI to enable LLM inference on Intel devices. It includes scripts for setting up an Ubuntu-based environment, converting models to OpenVINO IR format, and deploying an OpenVINO model server.

## Repository Structure

- **playground/**: Contains sample scripts tested in an OpenVINO 2025 and Ubuntu 24 environment.
- **setup/**: Provides scripts for setting up a Pytorch-XPU environment using Ubuntu 22 and Poetry.
installation.
- **Model Directories**: Named according to Hugging Face model IDs, each containing conversion and inference scripts based on the setup environment.

## Key Features

- **OpenVINO IR Conversion**: Converts Hugging Face models to OpenVINO IR format for optimized inference.
- **OpenVINO Model Server (OVMC)**: Implements an OpenVINO model server for running converted models.
- **GPU Acceleration**: Provides performance improvements for inference using Intel GPUs.

## Getting Started

1. **Verify Ubuntu Compatibility**: Check the appropriate WSL Ubuntu version using [intel-gpu-wsl-advisor](https://github.com/coela-oss/intel-gpu-wsl-advisor).
  -  intel-gpu-wsl-advisor repo is a prerequisite repository to determine the appropriate Ubuntu version for WSL 
2. **Setup Environment**: Use the scripts in `setup/` to install dependencies and configure Pytorch-XPU.
3. **Convert Models**: Run the provided conversion scripts to transform models into OpenVINO IR format.
4. **Deploy Model Server**: Install OpenVINO GenAI's OVMC server and execute converted models. by [Makefile](./Makefile)

## References

- [OpenVINO Documentation (2025)](https://docs.openvino.ai/2025/index.html)
- [Hugging Face Optimum-Intel](https://huggingface.co/blog/deploy-with-openvino)

This repository is under active development, integrating new features for optimized inference and deployment on Intel hardware.

