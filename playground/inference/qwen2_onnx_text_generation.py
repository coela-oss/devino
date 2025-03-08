from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
import torch
from util import measure_time

MODEL_PATH = Path("/workspaces/devino/mnt/models/huggingface/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/6cee29cc49b4932e8eef091a2904ee90a8cbe46a")
SAVE_MODEL_PATH = Path("mnt/models/openvino/trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/model.xml")


def load_model(model_path: str):
    """Load the Transformer model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer


def load_openvino_model(save_path: Path):
    """Load and compile the OpenVINO model."""
    core = ov.Core()
    return core.compile_model(save_path, "AUTO")


def tokenize_text(tokenizer, prompt: str):
    """Tokenize input text."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Tokenized Text:\n{text}")
    return tokenizer(text, return_tensors="pt")


@measure_time
def infer_with_openvino(compiled_model, tokenized_input):
    """Perform inference using OpenVINO model."""
    # Convert Tensor to NumPy for OpenVINO
    input_tensor = tokenized_input["input_ids"]
    attention_tensor = tokenized_input["attention_mask"]

    # Run Inference
    outputs = compiled_model([input_tensor, attention_tensor], decode_strings=True)
    token_ids = torch.tensor(outputs[0]).argmax(dim=-1).tolist()
    # Decode Output Tokens into Text
    return tokenizer.batch_decode(token_ids)[0]


@measure_time
def infer_with_transformer(model, tokenized_input):
    """Perform inference using the Transformer model."""
    generate_ids = model.generate(
        tokenized_input.input_ids
    )
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model(MODEL_PATH)
    
    # Tokenize text
    prompt = "Hi"
    tokenized_input = tokenize_text(tokenizer, prompt)
        
    # Load OpenVINO model
    compiled_model = load_openvino_model(SAVE_MODEL_PATH)
    
    # Perform inference using both models

    response_transformer = infer_with_transformer(model, tokenized_input)
    response_openvino = infer_with_openvino(compiled_model, tokenized_input)
    
    # Print Response
    print("\nTransformer Model Response:")
    print(response_transformer)
    
    print("\nOpenVINO Model Response:")
    print(response_openvino)
