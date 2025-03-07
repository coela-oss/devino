# Exam to Use LLMs on My Local PC (Intel Iris GPU)

## Overview

This guide provides a general approach for individuals who do not own an Nvidia GPU and wish to perform inference using an integrated laptop GPU.

The goal is to enable local execution of LLMs without modifying existing development environments or considering the purchase of a new PC.

### PC Specifications

* **OS:** Windows 11 Home 24H2
    * **Memory:** 16GB
    * **CPU:** [12th Gen Intel(R) Core(TM) i5-1235U 1.30 GHz](https://www.intel.com/content/www/us/en/products/sku/226261/intel-core-i51235u-processor-12m-cache-up-to-4-40-ghz/specifications.html)
        * **GPU:** Intel(R) Iris(R) Xe Graphics

**Note:**  
TODO: Add a link to Intel's driver compatibility table.

### Runtime Environment

* **Docker Desktop**
    * WSL integration enabled to use the default WSL distribution
* **Docker (including Devcontainer)**
    * Debian Bookworm with Python 3.11 (Devcontainer uses the official Microsoft Python container)

### Model Sources

* **Hugging Face**
  * `transformers`
  * `huggingface_hub`

### Additional Tools & Keywords

* Intel GPU Driver
* OpenVINO
* optimum-intel

**Note:**  
Refer to the [2025 OpenVINO documentation](https://docs.openvino.ai/2025/index.html) for the latest installation methods.

TODO: Add 2025 setup instructions.

## Outline

* **Recognizing the GPU Driver**
  * **Windows**
    * Set up the GPU driver on the PC
  * **Docker**
    * Verify GPU driver recognition using Docker CLI
    * Check GPU driver recognition with Docker Desktop WSL Integration
* **Obtaining and Converting LLMs**
  * Download from Hugging Face
  * Export to OpenVINO (IR) format
* **Inference**
  * Text Classification
  * Text Generation (WIP)

## Results

By enabling GPU acceleration, performance improved by **5.7x**.

```
vscode ➜ /workspaces/devino (main) $ python playground/inference/transformer_classification.py
Some weights of the model checkpoint at /workspaces/devino/mnt/models/models--cardiffnlp--twitter-roberta-base-sentiment-latest/snapshots/4ba3d4463bd152c9e4abd892b50844f30c646708 were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
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


