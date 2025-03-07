import openvino as ov

core = ov.Core()
assert(core.available_devices == ['CPU', 'GPU'])
