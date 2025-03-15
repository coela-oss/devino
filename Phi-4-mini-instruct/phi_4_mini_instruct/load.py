import os
import torch
from pathlib import Path
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
import openvino


# https://github.com/microsoft/PhiCookBook/blob/main/md/02.Application/02.Code/Phi4/GenProjectCode/README.md
def convert_chat_model() -> Path:
    model_id = os.environ['MODEL_ID']
    ov_home = os.environ["OV_HOME"]
    model_id_transformed = model_id.replace("/", "--")
    output_dir = f"{ov_home}/{model_id_transformed}"

    # Save Tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    hf_tokenizer.save_pretrained(output_dir)
    openvino.save_model(ov_tokenizer, f"{output_dir}/openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, f"{output_dir}/openvino_detokenizer.xml")

    # load model and convert it to OpenVINO
    # XPU Competit https://github.com/huggingface/transformers/issues/31922 Pytorch >=2.6.1 But 2.6.1 not competition intel gpu extension 
    # https://github.com/openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama/main.py
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True, # For OpenVINO file structure
        trust_remote_code=True, # resolve patch for [ValueError: `rope_scaling`'s short_factor field must have length 64, got 48]
        torch_dtype=torch.bfloat16,
        device_map="AUTO",
        use_cache=False,
        load_in_8bit=True, # Abobe device map issue related. set True may clash the process by nncf compression.
        compile=True, # Can not culculate of XPU Memsize https://github.com/huggingface/transformers/issues/31922
        stateful=False, # No need any state in my case 
    )

    # https://github.com/huggingface/safetensors/pull/509
    model.to(0)

    print("Load Pretrained Model. Compile for OpenVINO backend.")

    # save converted model
    model.save_pretrained(output_dir)
    print(f"Save complete.")


if __name__ == '__main__':
    convert_chat_model()