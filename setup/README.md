# Setup Project

## Overview
This project provides an automated setup environment for configuring a development environment with Intel OneAPI and PyTorch for XPU. The setup script installs all necessary dependencies and tools required for running machine learning workloads on Intel GPUs.

## Requirements

- Python 3.10.16
- Poetry (for dependency management)
- Intel GPU (for XPU support)
- Ubuntu-based Linux environment (or WSL2 for Windows users)

## Installation

**Run setup script**
   ```sh
   chmod +x wsl-ubuntu24-setup.sh
   ./wsl-ubuntu24-setup.sh
   ```

**Note:** The setup process installs Intel OneAPI, Bazel, Poetry, and various GPU drivers. 

## Building PyTorch for XPU

After completing the setup(activate build environment), you can build PyTorch with XPU support using:

```sh
python setup.py build
```

If you do not want to build it:

```sh
deactivate
```

**Warning:** The build process is time-consuming and may take several hours, depending on your system's performance.

**Note:** Necessary dependencies and tools already installed.


## Dependencies
This project uses the following dependencies:

- `openvino` (>=2024.1.0, <2025.0.0)
- `transformers` (>=4.49.0, <5.0.0)
- `tensorflow` (==2.15.0)
- `intel-extension-for-tensorflow[xpu]` (>=2.15.0.2, <3.0.0.0)
- `mkl-static` (>=2024.1.0, <2024.2.0)
- `mkl-include` (>=2024.1.0, <2024.2.0)

## Poetry Configuration
The `pyproject.toml` is configured to use Intel's PyTorch XPU sources:

```toml
[[tool.poetry.source]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
priority = "explicit"

[[tool.poetry.source]]
name = "pytorch-extension-xpu"
url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
priority = "explicit"
```

## License
This project is maintained by [coela-oss](mailto:to@coela.org) and follows an open-source licensing model.

## Contact
For inquiries, please contact **coela-oss** at [to@coela.org](mailto:to@coela.org).

