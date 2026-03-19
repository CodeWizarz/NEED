"""
Test script for Alpamayo-R1-10B / Qwen2-7B-Instruct

NOTE: Alpamayo-R1-10B is a VLA (Vision-Language-Action) model that requires:
- Multi-camera images (4 cameras: front-wide, front-tele, cross-left, cross-right)
- Text prompts
- Egomotion history (16 waypoints at 10Hz)

The standard transformers library doesn't yet support the alpamayo_r1 model type.
This script uses Qwen2-7B-Instruct as a working alternative for text-based reasoning.

For the full Alpamayo-R1 model, use the official inference code from:
https://github.com/NVlabs/alpamayo
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Use Qwen2-7B as a working alternative (or swap to Alpamayo when supported)
model_name = "Qwen/Qwen2-7B-Instruct"

print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Model loaded on:", next(model.parameters()).device)

prompt = """Camera scene:
- pedestrian crossing road
- traffic light is red
- ego vehicle approaching intersection

What should the vehicle do?"""

print("Running inference...")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n=== Generated Output ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
