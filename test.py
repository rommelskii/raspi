import onnxruntime as ort
from transformers import TrOCRProcessor
from PIL import Image
import numpy as np
import torch

# Load the processor and ONNX model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten", use_fast=True)
onnx_model_path = "trocr_large_handwritten.onnx"  # Path to the exported ONNX model
ort_session = ort.InferenceSession(onnx_model_path)

# Load and preprocess the input image
image_path = "sample.png"  # Path to your sample image
image = Image.open(image_path).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate decoder input IDs
decoder_start_token_id = processor.tokenizer.cls_token_id  # Replace with correct start token ID
decoder_input_ids = torch.tensor([[decoder_start_token_id]])

# Prepare inputs for ONNX Runtime
inputs = {
    "pixel_values": pixel_values.numpy(),
    "decoder_input_ids": decoder_input_ids.numpy()
}

# Run inference
outputs = ort_session.run(None, inputs)

# Decode the logits to text
predicted_ids = np.argmax(outputs[0], axis=-1)
decoded_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# Print the predicted text
print("Predicted text:", decoded_text)

