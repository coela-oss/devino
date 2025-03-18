import os
import openvino 

model_id = os.environ['MODEL_ID']
hf_home = os.environ['HF_HOME']
ov_home = os.environ["OV_HOME"]
model_id_transformed = model_id.replace("/", "--")
onnx_model_path = f"{hf_home}/hub/models--{model_id_transformed}/snapshots/9b9010e414c555d094141b5bb8da092ebe8f79fa/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/model.onnx"
output_dir = f"{ov_home}/{model_id_transformed}"

core = openvino.Core()
compiled_model = core.compile_model(onnx_model_path, "AUTO")
