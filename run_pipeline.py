import tensorflow as tf
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load vision model
print("Loading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

# Load reasoning model (USE EXISTING, DO NOT DOWNLOAD NEW)
model_name = "Qwen/Qwen2-7B-Instruct"

print(f"Loading reasoning model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

reasoning_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Models loaded")

# Fake camera input (simulate)
print("\nGenerating fake camera input...")
dummy_input = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
print(f"Input shape: {dummy_input.shape}")

# Run vision model
print("\nRunning vision model...")
vision_output = vision_model.signatures["serving_default"](
    tf.convert_to_tensor(dummy_input)
)

trajectory = list(vision_output.values())[0].numpy()
print(f"Vision output shape: {trajectory.shape}")

# Convert to scene text
scene_text = f"""
Predicted trajectory:
{trajectory}

Environment:
- Urban road
- Possible pedestrians

What should vehicle do?
"""

# Run reasoning model
print("\nRunning reasoning model...")
inputs = tokenizer(scene_text, return_tensors="pt").to("cuda")

outputs = reasoning_model.generate(
    **inputs,
    max_new_tokens=120,
    pad_token_id=tokenizer.eos_token_id
)

print("\n=== FINAL DECISION ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n=== PIPELINE WORKING ===")
print(f"Vision output shape: {trajectory.shape}")
print("Reasoning model: Qwen2-7B-Instruct (4-bit)")
print("Pipeline complete!")
