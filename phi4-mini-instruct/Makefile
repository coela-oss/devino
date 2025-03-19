SHELL := /bin/bash

.PHONY: help install-dependencies update-dependencies load inference-onnx inference-openvino

help:	## Show this help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install-dependencies: ## Install Project Dependencies
	@poetry install

update-dependencies: ## Update Project Dependencies
	@poetry update

include ../.env
inference-onnx: ## Inference model from HugggingFace
	# source /opt/intel/oneapi/setvars.sh --force
	HF_NOCACHE_HOME=${HF_NOCACHE_HOME} \
	MODEL_ID=microsoft/Phi-4-mini-instruct-onnx \
	poetry run python src/phi4_mini_instruct/inference_onnxruntime.py 

include ../.env
inference-openvino: ## Inference model from HugggingFace
	OV_HOME=${OV_HOME} \
	MODEL_ID=microsoft/Phi-4-mini-instruct \
	poetry run python src/phi4_mini_instruct/inference_openvino.py
