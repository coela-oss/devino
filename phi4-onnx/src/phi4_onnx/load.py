import os
from pathlib import Path
import openvino

# https://github.com/microsoft/PhiCookBook/blob/main/md/02.Application/02.Code/Phi4/GenProjectCode/README.md
def convert_chat_model() -> Path:
    model_id = os.environ['MODEL_ID']
    hf_home = os.environ['HF_HOME']
    ov_home = os.environ["OV_HOME"]
    model_id_transformed = model_id.replace("/", "--")
    onnx_model_path = f"{hf_home}/hub/models--{model_id_transformed}/snapshots/9b9010e414c555d094141b5bb8da092ebe8f79fa/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/model.onnx"
    output_dir = f"{ov_home}/{model_id_transformed}"

    # Save Tokenizer
    #hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    #ov_tokenizer, ov_detokenizer = openvino.convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    #hf_tokenizer.save_pretrained(output_dir)
    #openvino.save_model(ov_tokenizer, f"{output_dir}/openvino_tokenizer.xml")
    #openvino.save_model(ov_detokenizer, f"{output_dir}/openvino_detokenizer.xml")
    print("Read Model")
    core = openvino.Core()
    compiled_model = core.compile_model(onnx_model_path, "CPU")
    print("Compile Model")

    #coverted_model = openvino.convert_model(
    #    input_model=[onnx_model_path],
    #    share_weights=False,
    #    verbose=True,
    #)

    openvino.save_model(compiled_model, output_model=f"{output_dir}/openvino_onnx_model.xml")

    print("Load Pretrained Model. Compile for OpenVINO backend.")


if __name__ == '__main__':
    convert_chat_model()