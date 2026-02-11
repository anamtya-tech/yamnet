import numpy as np
import tensorflow as tf
import os

# Use relative path from integration/yamnet_custom/ to export_out/
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'export_out/tflite/yamnet.tflite')

interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dummy mel patch [1,96,64]
dummy_patch = np.random.rand(1,96,64).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], dummy_patch)
interpreter.invoke()

preds = interpreter.get_tensor(output_details[0]['index'])
embeds = interpreter.get_tensor(output_details[1]['index'])

print("Predictions shape:", preds.shape)
print("Embeddings shape:", embeds.shape)
