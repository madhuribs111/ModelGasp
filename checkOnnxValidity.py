import onnx

# Load the ONNX model
model = onnx.load('randomForest_model.onnx')

# Check the model's validity
onnx.checker.check_model(model)

print("The model is valid.")
