#!/bin/bash
# Pipeline runner - Step 2: Reasoning Model Only

python3 << 'EOF'
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("=" * 50)
print("STEP 2: REASONING MODEL")
print("=" * 50)

# Load trajectory from step 1
trajectory = np.load("/tmp/trajectory.npy")
print(f"\nLoaded trajectory shape: {trajectory.shape}")

# Convert to scene text
scene_text = f"""
Predicted trajectory shape: {trajectory.shape}

Environment:
- Urban road
- Possible pedestrians
- Traffic light red

What should vehicle do?
"""

# Load reasoning model
model_name = "Qwen/Qwen2-7B-Instruct"
print(f"\nLoading reasoning model: {model_name}")

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

print("Model loaded on:", next(reasoning_model.parameters()).device)

# Run reasoning
print("\nRunning reasoning model...")
inputs = tokenizer(scene_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = reasoning_model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "=" * 50)
print("FINAL DECISION")
print("=" * 50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n" + "=" * 50)
print("PIPELINE COMPLETE!")
print("=" * 50)
EOF
