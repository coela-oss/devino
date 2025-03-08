from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import openvino as ov

MODEL_PATH = "/workspaces/devino/mnt/models/huggingface/models--cardiffnlp--twitter-roberta-base-sentiment-latest/snapshots/4ba3d4463bd152c9e4abd892b50844f30c646708"
SAVE_MODEL_PATH = Path("/workspaces/devino/mnt/models/openvino/cardiffnlp--twitter-roberta-base-sentiment-latest/model.xml")


def load_model(model_path: str):
    """Load the Transformer model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, return_dict=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torchscript=True)
    return model, tokenizer


def convert_and_save_model(model, save_path: Path):
    """Convert PyTorch model to OpenVINO format and save it."""
    if not save_path.exists():
        ov_model = ov.convert_model(model)
        ov.save_model(ov_model, save_path)


if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model(MODEL_PATH)
        
    # Convert and save OpenVINO model
    convert_and_save_model(model, SAVE_MODEL_PATH)
