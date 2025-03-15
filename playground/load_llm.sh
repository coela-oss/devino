#!/bin/bash

# Enable error handling
set -e

# Check if at least two arguments (file extension and model ID) are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <file_extension> <huggingface_model_id> [huggingface-cli_options...]"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$HF_HOME" ] || [ -z "$OV_HOME" ]; then
    echo "Error: Please set HF_HOME and OV_HOME environment variables."
    exit 1
fi

# Extract file extension, model ID, and additional options for huggingface-cli
FILE_EXTENSION=$1
MODEL_ID=$2
shift 2
CLI_OPTIONS="$@"

# Download the model using huggingface-cli
echo "Downloading model: $MODEL_ID with options: $CLI_OPTIONS"
huggingface-cli download "$MODEL_ID" $CLI_OPTIONS

# Define the cached model directory path
MODEL_CACHE_PATH="$HF_HOME/hub/models--${MODEL_ID//\//--}"

if [ ! -d "$MODEL_CACHE_PATH" ]; then
    echo "Error: Cached model directory not found: $MODEL_CACHE_PATH"
    exit 1
fi

# Define the output directory based on the model ID
OUTPUT_DIR="$OV_HOME/${MODEL_ID//\//--}"
mkdir -p "$OUTPUT_DIR"

echo "Model cache path: $MODEL_CACHE_PATH"
echo "Output directory: $OUTPUT_DIR"

# Recursively search for a file with the specified extension
MODEL_FILE=$(find "$MODEL_CACHE_PATH/snapshots" -name "*.$FILE_EXTENSION" | head -n 1)

if [ -z "$MODEL_FILE" ]; then
    echo "Error: File with extension .$FILE_EXTENSION not found in $MODEL_CACHE_PATH or its subdirectories"
    exit 1
fi

echo "Found file: $MODEL_FILE"

# Convert the model using OpenVINO
echo "Converting file with OpenVINO..."

# If killed terminal, means use the default config include memory usage defined provider
if [ "$FILE_EXTENSION" == "onnx" ]; then
    #ovc "$MODEL_FILE" --output_model "$OUTPUT_DIR/model.xml" --verbose
    MODEL_DIR=$(dirname "$MODEL_FILE")
    optimum-cli export openvino \
        --task text-generation-with-past \
        --weight-format int4 \
        --model "$MODEL_ID" \
        "$$OUTPUT_DIR"
else
    MODEL_DIR=$(dirname "$MODEL_FILE")
    optimum-cli export openvino \
        --task text-generation-with-past \
        --weight-format fp16 \
        --model "$MODEL_ID" \
        "$OUTPUT_DIR"

    #ovc "$MODEL_DIR" --output_model "$OUTPUT_DIR" --verbose
fi

echo "Conversion completed. Output saved to: $OUTPUT_DIR"
