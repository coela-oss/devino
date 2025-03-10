from pathlib import Path
import openvino as ov
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM

MODEL_PATH = Path("/workspaces/devino/mnt/models/huggingface/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/6cee29cc49b4932e8eef091a2904ee90a8cbe46a")
SAVE_MODEL_PATH = Path("mnt/models/openvino/trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/model.xml")


def load_model(model_path: str):
    """Load the Transformer model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = OVModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer


def convert_and_save_model(model, tokenizer, save_path: Path):
    """Convert PyTorch model to OpenVINO format and save it."""
    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Tokenized Text:\n{text}")

    inputs = tokenizer(text, return_tensors="pt")
    # Disable KV caching (Fixes `past_key_values` issue)
    model.config.use_cache = False  # Fix for OpenVINO compatibility

    if not save_path.exists():
        #ov_model = ov.convert_model(
        #    model,
        #    #example_input=(inputs["input_ids"], inputs["attention_mask"]),
        #    verbose=True
        #)
        ov.save_model(model, save_path)


if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model(MODEL_PATH)
    # Convert and save OpenVINO model
    convert_and_save_model(model, tokenizer, SAVE_MODEL_PATH)
