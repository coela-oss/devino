SHELL := /bin/bash

.PHONY: help install-dependencies update-dependencies load inference

MODEL_ID=microsoft/Phi-4-mini-instruct-onnx

help:	## Show this help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install-dependencies: ## Install Project Dependencies
	@poetry install

update-dependencies: ## Update Project Dependencies
	@poetry update

include ../.env
load: ## Load model from HugggingFace
	# source /opt/intel/oneapi/setvars.sh --force
	HF_HOME=${HF_HOME} \
	OV_HOME=${OV_HOME} \
	MODEL_ID=${MODEL_ID} \
	TF_ENABLE_ONEDNN_OPTS=1 \
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OVMS_LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/lib/x86_64-linux-gnu \
	poetry run python src/phi4_onnx/load.py

include ../.env
inference: ## Inference model from HugggingFace
	# source /opt/intel/oneapi/setvars.sh --force
	HF_HOME=${HF_HOME} \
	OV_HOME=${OV_HOME} \
	MODEL_ID=${MODEL_ID} \
	poetry run python src/phi4_onnx/inference.py
