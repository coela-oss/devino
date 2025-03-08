

dep-install:
	pip install -r requirements.txt

list-gpu-devices:
	python playground/gpu_devices.py

download-model:
	python playground/model/download_model.py

convert-model:
	python playground/model/convert_text_classification_model.py
