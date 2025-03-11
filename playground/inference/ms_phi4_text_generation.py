from pathlib import Path
from optimum.intel import OVModelForCausalLM
from util import measure_time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openvino_genai
import openvino

from openvino_tokenizers import convert_tokenizer


cache_dir_hf = "/mnt/c/Coela/llm/models/huggingface"
cache_dir_ov = "/mnt/c/Coela/llm/models/openvino"
SAVE_MODEL_PATH =  Path("/mnt/c/Coela/llm/models/openvino/microsoft--Phi-4-mini-instruct")
model_full_path =  "/mnt/c/Coela/llm/models/openvino/microsoft--Phi-4-mini-instruct/openvino_model.xml"
model_path =  "/mnt/c/Coela/llm/models/openvino/microsoft--Phi-4-mini-instruct"
model_id = "microsoft/Phi-4-mini-instruct"

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    #"temperature": 0.0,
    #"do_sample": False,
}

hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)


@measure_time
def infer_with_openvino_gpu_cache():
    core = openvino.Core()
    core.set_property({"props.cache_dir": ".cache"})
    model = core.read_model(model=model_full_path)
    compiled_model = core.compile_model(model=model, device_name="GPU")
    text = "Replace me by any text you'd like."
    encoded_input = hf_tokenizer(text, return_tensors='pt')
    result = compiled_model({**encoded_input})
    print(result)


@measure_time
def infer_with_openvino():

    model = OVModelForCausalLM.from_pretrained(
        SAVE_MODEL_PATH,
        device="auto",
        use_cache=True,
        cache_dir=cache_dir_ov,
        local_files_only=True,
    )
        
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=hf_tokenizer,
    )
    
    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])


@measure_time
def infer_with_openvino_genai():
    pipe = openvino_genai.LLMPipeline(
        model_path,
        device="GPU",
    )
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    output = pipe.generate(messages, ov_tokenizer)
    print(output[0]['generated_text'])




if __name__ == "__main__":
    infer_with_openvino_gpu_cache()
    #infer_with_openvino()
    #infer_with_openvino_genai()

