import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from optimum.intel import OVModelForCausalLM
import torch

# https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot.ipynb
# optimum-cli export openvino --task text-generation --weight-format fp16 --model "microsoft/phi-4-mini-instruct" --cache_dir "/mnt/c/Coela/llm/models/huggingface" "/mnt/c/Coela/llm/models/openvino/microsoft--Phi-4-mini-instruct"

cache_dir = "/mnt/c/Coela/llm/models/huggingface"
os.environ["HF_HOME"] = cache_dir

model_name = "microsoft/Phi-4-mini-instruct"

SAVE_MODEL_PATH = Path("/mnt/c/Coela/llm/models/openvino/microsoft--Phi-4-mini-instruct")


def load_model():
    torch.random.manual_seed(0)
    """Load the Transformer model and tokenizer."""
    model = OVModelForCausalLM.from_pretrained(
        model_name,
        compile=True,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        config=AutoConfig.from_pretrained(model_name)
    )
    return model


def convert_and_save_model(model, save_path: Path):
    """Convert PyTorch model to OpenVINO format and save it."""
    # Disable KV caching (Fixes `past_key_values` issue)
    model.config.use_cache = False  # Fix for OpenVINO compatibility
    # https://docs.openvino.ai/2025/openvino-workflow/torch-compile.html
    # https://huggingface.co/docs/optimum/main/intel/openvino/inference#export
    model.to("gpu")
    model.save_pretrained(save_path)


if __name__ == "__main__":
    # Load model and tokenizer
    model = load_model()
    # Convert and save OpenVINO model
    convert_and_save_model(model, SAVE_MODEL_PATH)
