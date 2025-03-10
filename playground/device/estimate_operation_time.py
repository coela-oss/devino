import os
from transformers import AutoConfig

cache_dir = "./mnt/models/huggingface"
os.environ["HF_HOME"] = cache_dir

# https://www.intel.com/content/www/us/en/content-details/841556/app-metrics-for-intel-microprocessors-intel-core-processor.html

def estimate_llm_inference_time(
    num_hidden_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    max_position_embeddings: int,
    gflops: float,
    batch_size: int = 1,
    efficiency: float = 0.7,  # APPの調整係数 (0.6 ~ 0.9)
):
    """LLMの推論時間をGFLOPSを基に算出する"""
    
    expansion_rate = 4  # 通常 4
    ffn_size = hidden_size * expansion_rate

    # Self-Attention の FLOPs
    flops_attention = (2 * batch_size * max_position_embeddings * hidden_size * num_attention_heads +
                       2 * batch_size * max_position_embeddings * hidden_size**2)

    # Feed Forward Network (FFN) の FLOPs
    flops_ffn = 2 * batch_size * max_position_embeddings * hidden_size * ffn_size

    # 1 層あたりの FLOPs
    flops_per_layer = flops_attention + flops_ffn

    # 全レイヤー分
    total_flops = num_hidden_layers * flops_per_layer

    # APP (Adjusted Peak Performance) を考慮
    adjusted_gflops = gflops * efficiency

    # 推論時間 (秒)
    inference_time = total_flops / (adjusted_gflops * 1e9)  # GFLOPS の単位を合わせる

    return inference_time


def estimate_llm_loading_time(
    num_hidden_layers: int,
    hidden_size: int,
    vocab_size: int,
    dtype_size: int,  # (bytes) bfloat16=2, fp16=2, fp32=4
    storage_bandwidth: float,  # (MB/s)
    pci_bandwidth: float  # (GB/s)
):
    """LLMのロード時間を推定"""
    
    # 総パラメータ数 (簡易的な近似)
    total_parameters = (hidden_size**2 * num_hidden_layers) + (hidden_size * vocab_size)
    print(f"Total Prameters: {total_parameters}")

    # モデルのサイズ (MB)
    model_size = (total_parameters * dtype_size) / (1024**2)  # bytes → MB

    # ストレージからの読み込み時間 (秒)
    loading_time = model_size / storage_bandwidth

    # GPU 転送時間 (秒)
    gpu_transfer_time = model_size / (pci_bandwidth * 1024)  # GB/s → MB/s に変換

    # 合計ロード時間
    total_loading_time = loading_time + gpu_transfer_time

    return total_loading_time


model_id = "microsoft/phi-4"


# 量子化オプションを含む設定
config = AutoConfig.from_pretrained(model_id)
#gflops = 12500  # 例: A100 GPU (FP16)
#efficiency = 0.75  # 実測ベースの効率

gflops = 208
efficiency = 0.0624

# 仮のモデル & GPU 情報
num_hidden_layers = config.num_hidden_layers  # 例えば GPT-3 Small (6B)
hidden_size = config.hidden_size
num_attention_heads = config.num_attention_heads
max_position_embeddings = config.max_position_embeddings

inference_time = estimate_llm_inference_time(
    num_hidden_layers, hidden_size, num_attention_heads,
    max_position_embeddings, gflops, efficiency
)

print(f"推論時間: {inference_time:.6f} 秒")


# 仮のモデル & ハードウェア情報
vocab_size = config.vocab_size
dtype_size = 2  # bfloat16 (2 bytes)

storage_bandwidth = 3400  # NVMe SSD (MB/s)
pci_bandwidth = 32  # PCIe Gen4 (GB/s)

loading_time = estimate_llm_loading_time(
    num_hidden_layers, hidden_size, vocab_size,
    dtype_size, storage_bandwidth, pci_bandwidth
)

print(f"モデルのロード時間: {loading_time:.6f} 秒")
