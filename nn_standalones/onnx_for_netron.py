import onnx
import onnx.utils
import onnx.version_converter

# Based on dicussion at https://github.com/lutzroeder/netron/issues/71

model_file = './results/models/forward_model.onnx'
onnx_model = onnx.load(model_file)
# onnx_model = onnx.version_converter.convert_version(onnx_model, target_version=8)
# onnx_model = onnx.utils.polish_model(onnx_model)
model_with_shapes = onnx.shape_inference.infer_shapes(onnx_model)
onnx.save(model_with_shapes, "./results/models/forward_model2.onnx")
