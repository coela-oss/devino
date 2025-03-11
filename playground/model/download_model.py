from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from transformers.utils.hub import default_cache_path

# export HF_HOME=/mnt/c/Coela/llm/models/huggingface
# huggingface-cli env
model_name = "microsoft/Phi-4-mini-instruct"

print(f"Downloading and caching model: {model_name} to {default_cache_path}...")

#tokenizer = AutoTokenizer.from_pretrained(
#    model_name,
#)
    
model = AutoModel.from_pretrained(model_name)

# 設定情報を取得
config = AutoConfig.from_pretrained(model_name)

# 設定の確認
print("Model Configuration:")
print(f"  Model Type: {config.model_type}")
print(f"  Hidden Size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
print(f"  Number of Attention Heads: {config.num_attention_heads if hasattr(config, 'num_attention_heads') else 'N/A'}")
print(f"  Number of Layers: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'N/A'}")
print(f"  Vocabulary Size: {config.vocab_size}")
print(f"  Max Position Embeddings: {config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 'N/A'}")
print(f"  Attention Dropout: {config.attention_probs_dropout_prob if hasattr(config, 'attention_probs_dropout_prob') else 'N/A'}")
print(f"  Hidden Dropout: {config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 'N/A'}")
print(f"  Activation Function: {config.hidden_act if hasattr(config, 'hidden_act') else 'N/A'}")

# GPUが利用可能か確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# モデルとトークナイザーをロード
#try:
 #   model = AutoModel.from_pretrained(model_name).to(device)
 #   tokenizer = AutoTokenizer.from_pretrained(model_name)
 #   print(f"\nModel {model_name} and tokenizer loaded successfully.")
#except Exception as e:
#    print(f"Error loading model: {e}")

# モデルのパラメータ数を確認
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

# モデルのデフォルト設定をチェック
print("\nModel Default Settings:")
print(f"  Torch dtype: {model.dtype}")
print(f"  Device: {model.device}")

# トークナイザーの設定を確認
print("\nTokenizer Configuration:")
print(f"  Vocab Size: {tokenizer.vocab_size}")
print(f"  Model Max Length: {tokenizer.model_max_length}")
print(f"  Padding Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"  EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"  BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

# モデルの構造を表示（オプション）
# print(model)