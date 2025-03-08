# Exams to Use LLMs on My Local PC (Intel Iris GPU)

## Overview

This guide provides a general approach for individuals who do not own an Nvidia GPU and wish to perform inference using an integrated laptop GPU.

The goal is to enable local execution of LLMs casual without modifying existing development environments or considering the purchase of a new PC.

## Result Summary

By enabling GPU acceleration, performance improved by **5.7x**.

```
vscode ➜ /workspaces/devino (main) $ python playground/inference/roberta_text_classification.py
Function 'infer_with_transformer' executed in 9.391381 seconds
Function 'infer_with_openvino' executed in 1.628792 seconds
Transformer Model Prediction:
1) neutral 0.703
2) positive 0.2633
3) negative 0.0337

OpenVINO Model Prediction:
1) neutral 0.7033
2) positive 0.2629
3) negative 0.0338
```

## Exams Outline

* **Recognizing the GPU Driver**
  * **Windows**
    * Set up the GPU driver on the PC
  * **Docker**
    * Verify GPU driver recognition using Docker CLI
    * Check GPU driver recognition with Docker Desktop WSL Integration
* **Obtaining and Converting LLMs**
  * Download from Hugging Face
  * Export to OpenVINO (IR) format
    * https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-memory-usage.html
* **Inference**
  * Text Classification
  * Text Generation
    * https://docs.openvino.ai/2023.3/openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_ONNX.html

## How to Use This Repository

### Tools Setup
* Install [VSCode](https://code.visualstudio.com/)
* Install [Docker Desktop](https://www.docker.com/get-started/)
* Install [Git for Windows](https://git-scm.com/downloads/win)
* Install [Intel GPU Driver](?tab=readme-ov-file#pc-specifications)

### Container Build
* Clone This Repo and Open in VSCode
  * ```git clone https://github.com/coela-oss/devino.git```
  * ```code devino```
* Decide Mount Directory for LLMs
  * Modify [Devcontainer JSON](.devcontainer/devcontainer.json)
* Build Devcontainer

### Run scripts in Devcontainer

ex) Check the GPU driver is recognized

```
vscode ➜ /workspaces/devino (main) $ python playground/gpu_devices.py 
# ['CPU', 'GPU']
```

if it shows CPU only, GPU Driver invalid setup. 

## Details

### PC Specifications

It's the spec I tried, but if you have newer than 11th generation Intel CPU and GPU emdebed, it should work the same.

Detail compatibility can see [openvino/system-requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html)

* **OS:** Windows 11 Home 24H2
    * **Memory:** 16GB
    * **CPU:** [12th Gen Intel(R) Core(TM) i5-1235U 1.30 GHz](https://www.intel.com/content/www/us/en/products/sku/226261/intel-core-i51235u-processor-12m-cache-up-to-4-40-ghz/specifications.html)
        * **GPU:** Intel(R) Iris(R) Xe Graphics

### Runtime Environment

* **Docker Desktop**
    * WSL integration enabled to use the default WSL distribution
* **Docker (including Devcontainer)**
    * Debian Bookworm with Python 3.11 (Devcontainer uses the official Microsoft Python container)

### Model Sources Download

* **Hugging Face**
  * `transformers`
  * `huggingface_hub`

### Additional Tools & Keywords

* Intel GPU Driver
* OpenVINO
* optimum-intel

other keywords see [glossary doc](https://huggingface.co/docs/transformers/main/en/glossary) by huggingface.

**Note:**  
Refer to the [2025 OpenVINO documentation](https://docs.openvino.ai/2025/index.html) for the latest installation methods.

