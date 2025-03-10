import os
from optimum.intel import OVModelForCausalLM
# new imports for inference
from transformers import AutoTokenizer

cache_dir = "./mnt/models/huggingface"
os.environ["HF_HOME"] = cache_dir
SAVE_MODEL_PATH = "mnt/models/openvino/microsoft--phi-4/model.xml"
model_id = "microsoft/phi-4"
model = OVModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    export=True,
    local_files_only=True,
    config={"use_cache": False}
).to("GPU")

model.config.use_cache = False
model.save_pretrained(SAVE_MODEL_PATH)

# inference
prompt = "The weather is:"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
