import openvino as ov
core = ov.Core()

model_path = "mnt/models/models--onnx-community--Phi-3.5-mini-instruct-onnx-web/snapshots/9eb48666f3700b2799779d6c7f4d58975f9eb63c/onnx/model_q4f16.onnx"
compiled_model = core.compile_model(
    model_path,
    "AUTO"
)
infer_request = compiled_model.create_infer_request()
infer_request.start_async() 
infer_request.wait()
output = infer_request.get_output_tensor() 
output_buffer = output.data
print(output_buffer)
