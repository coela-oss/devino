import os
from pathlib import Path
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig, OVConfig, OVQuantizer
from transformers import AutoTokenizer

#TF_ENABLE_ONEDNN_OPTS=0 optimum-cli export openvino \
#    --task text-generation \
#    --weight-format int8 \
#    --model "microsoft/phi-4" \
#    --weight-format fp16  \
#    --disable-stateful \
#    --cache_dir "./mnt/models/huggingface"  \
#    "mnt/models/openvino/microsoft--phi-4"


cache_dir = "./mnt/models/huggingface"
os.environ["HF_HOME"] = cache_dir
SAVE_MODEL_PATH = "mnt/models/openvino/microsoft--phi-4"
model_id = "microsoft/phi-4"


def convert_chat_model(model_type: str, precision: str, model_dir: Path) -> Path:
    """
    Convert chat model

    Params:
        model_type: selected mode type and size
        precision: model precision ["fp16", "int8", "int4"]
        model_dir: dir to export model
    Returns:
       Path to exported model dir
    """
    output_dir = model_dir / precision

    # load model and convert it to OpenVINO
    model = OVModelForCausalLM.from_pretrained(model_type, export=True, compile=False, load_in_8bit=False)
    # change precision to FP16
    model.half()

    if precision != "fp16":
        # select quantization mode
        quant_config = OVWeightQuantizationConfig(bits=4, sym=False, ratio=0.8) if precision == "int4" else OVWeightQuantizationConfig(bits=8, sym=False)
        config = OVConfig(quantization_config=quant_config)

        suffix = "-INT4" if precision == "int4" else "-INT8"
        output_dir = output_dir.with_name(output_dir.name + suffix)

        # create a quantizer
        quantizer = OVQuantizer.from_pretrained(model, task="text-generation")
        # quantize weights and save the model to the output dir
        quantizer.quantize(save_directory=output_dir, ov_config=config)
    else:
        output_dir = output_dir.with_name(output_dir.name + "-FP16")
        # save converted model
        model.save_pretrained(output_dir)

    # export also tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.save_pretrained(output_dir)

    return Path(output_dir)


if __name__ == "__main__":
    convert_chat_model(model_id, "int4", Path(SAVE_MODEL_PATH))