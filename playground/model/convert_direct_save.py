import os
import click
import torch
from pathlib import Path
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig

@click.command()
@click.argument("model_id")
def convert_chat_model(model_id: str) -> Path:
    ov_home = os.environ.get("OV_HOME", "") 
    model_id_transformed = model_id.replace("/", "--")
    output_dir = f"{ov_home}/{model_id_transformed}"

    config = AutoConfig.from_pretrained(model_id)


    # load model and convert it to OpenVINO
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        torch_dtype=torch.bfloat16,
        config=config,
        local_files_only=True
    )

    # save converted model
    model.save_pretrained(output_dir)

    # export also tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True
    )
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    convert_chat_model()