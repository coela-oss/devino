from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import openvino as ov
import torch
import numpy as np
from util import measure_time

MODEL_PATH = "/workspaces/devino/mnt/models/huggingface/models--cardiffnlp--twitter-roberta-base-sentiment-latest/snapshots/4ba3d4463bd152c9e4abd892b50844f30c646708"
SAVE_MODEL_PATH = Path("/workspaces/devino/mnt/models/openvino/cardiffnlp--twitter-roberta-base-sentiment-latest/model.xml")

def load_model(model_path: str):
    """Load the Transformer model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, return_dict=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torchscript=True)
    return model, tokenizer


def tokenize_text(tokenizer, text: str):
    """Tokenize input text."""
    return tokenizer(text, return_tensors="pt")


def load_openvino_model(save_path: Path):
    """Load and compile the OpenVINO model."""
    core = ov.Core()
    return core.compile_model(save_path, "AUTO")


@measure_time
def infer_with_openvino(compiled_model, input_data):
    """Perform inference using OpenVINO model."""
    scores = compiled_model(input_data.data)[0]
    scores = torch.softmax(torch.tensor(scores[0]), dim=0).detach().numpy()
    return scores


@measure_time
def infer_with_transformer(model, encoded_input):
    """Perform inference using the Transformer model."""
    output = model(**encoded_input)
    scores = torch.softmax(output[0][0], dim=0).numpy(force=True)
    return scores


def print_prediction(model, scores):
    """Print the prediction results sorted by confidence scores."""
    for i, descending_index in enumerate(scores.argsort()[::-1]):
        label = model.config.id2label[descending_index]
        score = np.round(float(scores[descending_index]), 4)
        print(f"{i+1}) {label} {score}")


if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model(MODEL_PATH)
    
    # Tokenize text
    text = "今日は晴れです！"
    encoded_input = tokenize_text(tokenizer, text)
        
    # Load OpenVINO model
    compiled_model = load_openvino_model(SAVE_MODEL_PATH)
    
    # Perform inference using both models

    scores_transformer = infer_with_transformer(model, encoded_input)
    scores_openvino = infer_with_openvino(compiled_model, encoded_input)
    
    # Print predictions
    print("Transformer Model Prediction:")
    print_prediction(model, scores_transformer)
    
    print("\nOpenVINO Model Prediction:")
    print_prediction(model, scores_openvino)
