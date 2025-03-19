import os
#import openvino_genai
import openvino
import numpy as np
import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline, AutoConfig
from openvino_tokenizers import convert_tokenizer
from openvino_tokenizers.constants import EOS_TOKEN_ID_NAME

model_id = os.environ['MODEL_ID']
ov_home = os.environ["OV_HOME"]
model_id_transformed = model_id.replace("/", "--")
ov_model_dir = f"{ov_home}/{model_id_transformed}"
ov_model_full_path = f"{ov_model_dir}/openvino_model.xml"

torch.xpu.reset_peak_memory_stats(torch.xpu.current_device())
print(torch.xpu.max_memory_allocated(0))


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
tokenizer, detokenizer = openvino.compile_model(ov_tokenizer), openvino.compile_model(ov_detokenizer)

def infer_with_openvino_gpu_cache():
    core = openvino.Core()
    #core.set_property({"props.cache_dir": ".cache"})
    model = core.read_model(model=ov_model_full_path)
    compiled_model = core.compile_model(model=model, device_name="AUTO")
    infer_request = compiled_model.create_infer_request()

    text_input = ["Quick brown fox jumped"]

    model_input = {name.any_name: output for name, output in tokenizer(text_input).items()}

    if "position_ids" in (input.any_name for input in infer_request.model_inputs):
        model_input["position_ids"] = np.arange(model_input["input_ids"].shape[1], dtype=np.int64)[np.newaxis, :]

    # No beam search, set idx to 0
    model_input["beam_idx"] = np.array([0], dtype=np.int32)
    print(model_input)

    # End of sentence token is that model signifies the end of text generation
    # Read EOS token ID from rt_info of tokenizer/detokenizer ov.Model object
    eos_token = ov_tokenizer.get_rt_info(EOS_TOKEN_ID_NAME).value

    tokens_result = np.array([[]], dtype=np.int64)
    print(tokens_result)

    # Reset KV cache inside the model before inference
    #infer_request.reset_state()
    #max_infer = 5

    output_tensor = infer_request.infer(model_input)

    # Get the most probable token
    token_indices = np.argmax(output_tensor.data, axis=-1)
    output_token = token_indices[:, -1:]

    # Concatenate previous tokens result with newly generated token
    tokens_result = np.hstack((tokens_result, output_token))
    if output_token[0, 0] == eos_token:
        print("error")
    else:
        # Prepare input for the next inference iteration
        model_input["input_ids"] = output_token
        model_input["attention_mask"] = np.hstack((model_input["attention_mask"].data, [[1]]))
        model_input["position_ids"] = np.hstack(
            (
                model_input["position_ids"].data,
                [[model_input["position_ids"].data.shape[-1]]],
            )
        )


        text_result = detokenizer(tokens_result)["string_output"]
        print(f"Prompt:\n{text_input[0]}")
        print(f"Generated:\n{text_result[0]}")



def infer_with_openvino():

    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

    ov_model = OVModelForCausalLM.from_pretrained(
        ov_model_dir,
        device='GPU',
        ov_config=ov_config,
        use_cache=False,
        config=AutoConfig.from_pretrained(ov_model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )

    tok = AutoTokenizer.from_pretrained(ov_model_dir, trust_remote_code=True)

    tokenizer_kwargs =  {"add_special_tokens": False}

    prompt = "<|user|>\nI have $20,000 in my savings account, where I receive a 4% profit per year and payments twice a year. Can you please tell me how long it will take for me to become a millionaire? Also, can you please explain the math step by step as if you were explaining it to an uneducated person?\n<|end|><|assistant|>\n"

    input_tokens = tok(prompt, return_tensors="pt", **tokenizer_kwargs)

    answer = ov_model.generate(**input_tokens, max_new_tokens=1024)

    print(tok.batch_decode(answer, skip_special_tokens=True)[0])



def infer_with_openvino_genai():
    # https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/memory_allocation_gpu_plugin.md
    # models/openvino/microsoft--Phi-4-mini-instruct/openvino_model.xml self.model.model.embed_tokens.weight

    #genai_tokenizer = openvino_genai.Tokenizer(ov_model_dir)
    #beam_idx = np.array([0], dtype=np.int32)
    #pipe = openvino_genai.LLMPipeline(ov_model_dir, genai_tokenizer, "AUTO")

    #result = pipe.generate(messages[0]["content"], beam_idx=beam_idx, max_new_tokens=1)

    #print(f"Prompt:\n{messages[0]}")
    #print(f"Generated:\n{result}")
    pass

if __name__ == "__main__":
    #infer_with_openvino_gpu_cache()
    infer_with_openvino()
    #infer_with_openvino_genai()

