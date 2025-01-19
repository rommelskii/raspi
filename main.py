import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image

# Load the model and processor
model_name = "microsoft/trocr-large-handwritten"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = TrOCRProcessor.from_pretrained(model_name)

# Ensure the decoder_start_token_id is set
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id  # Set to the start token ID

# Prepare an example input
example_image_path = "sample.png"  # Replace with your image path
example_image = Image.open(example_image_path).convert("RGB")
pixel_values = processor(example_image, return_tensors="pt").pixel_values

# Generate example decoder input ids
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

# Export the model to ONNX
torch.onnx.export(
    model,                                 # The model to export
    (pixel_values, decoder_input_ids),     # Provide both encoder and decoder inputs
    "trocr_large_handwritten.onnx",        # Output ONNX file
    input_names=["pixel_values", "decoder_input_ids"],  # Input names
    output_names=["logits"],               # Output names
    dynamic_axes={                         # Allow dynamic shapes
        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        "decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14                       # Updated ONNX opset version
)


