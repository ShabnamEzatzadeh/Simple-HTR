import keras2onnx
from tensorflow.keras.models import load_model


# convert to onnx model
prediction_model = load_model("../models/prediction-model-keras-ours-new.h5")
onnx_prediction_model = keras2onnx.convert_keras(prediction_model, 'keras-ours-prediction-onnx')
output_prediction_model_path = "../models/keras-ours-prediction-model.onnx"
# and save the model in ONNX format
keras2onnx.save_model(onnx_prediction_model, output_prediction_model_path)
