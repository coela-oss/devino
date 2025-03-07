import os
from transformers import AutoModel, AutoTokenizer

cache_dir = "./mnt/models"
os.environ["HF_HOME"] = cache_dir

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print(f"Downloading and caching model: {model_name} to {cache_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

print("Download completed!")

print(f"Cached files in {cache_dir}:")
for root, dirs, files in os.walk(cache_dir):
    for file in files:
        print(os.path.join(root, file))


