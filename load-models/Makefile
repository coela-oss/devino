SHELL := /bin/bash

.PHONY: help download-phi4-mini-onnx-local convert-phi4-mini-onnx convert-phi4-mini

MODEL_ID=microsoft/Phi-4-mini-instruct


help:	## Show this help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

include ../.env
download-phi4-mini-onnx: ## Load model from HugggingFace
	HF_HOME=${HF_HOME} \
	OV_HOME=${OV_HOME} \
	huggingface-cli download \
	${MODEL_ID} \
	--include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* \
	--local-dir ${OV_HOME}/microsoft--Phi-4-mini-instruct-onnx

#convert-phi4-mini-onnx-local:
#	HF_NOCACHE_HOME=${HF_NOCACHE_HOME} \
#	poetry run optimum-cli export openvino \
#		--model ${MODEL_ID} \
#		--task "text-generation" \
#		--framework pt \
#		--weight-format int4 \
#		--cache_dir ${HF_NOCACHE_HOME}/microsoft--Phi-4-mini-instruct-onnx \
#		--trust-remote-code \
#		"out"

convert-phi4-mini:
	HF_HOME=${HF_HOME} \
	OV_HOME=${OV_HOME} \
	poetry run optimum-cli export openvino \
		--model ${MODEL_ID} \
		--task "text-generation" \
		--cache_dir ${HF_HOME}/hub \
		--trust-remote-code \
		"${OV_HOME}/microsoft--Phi-4-mini-instruct"
