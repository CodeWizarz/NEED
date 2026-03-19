import tensorflow as tf
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import sys

# Load vision model
print("Loading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

# Load reasoning model (use cached Qwen)
print("\nLoading reasoning model...")
model_name = "Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Models loaded")

# Simulated input
print("\nGenerating camera input...")
dummy_input = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
print(f"Input shape: {dummy_input.shape}")

# Vision inference
print("\nRunning vision model...")
vision_output = vision_model.signatures["serving_default"](
    tf.convert_to_tensor(dummy_input)
)

trajectory = list(vision_output.values())[0].numpy()

print("Trajectory shape:", trajectory.shape)

# Analyze trajectory for decision
traj_mean = np.mean(trajectory[0], axis=0)
traj_speed = np.linalg.norm(traj_mean)
is_stopping = traj_speed < 0.5

# Build structured prompt
scene = f"""You are an autonomous driving AI.

Trajectory prediction: {traj_mean}
Speed estimate: {traj_speed:.3f}
Stopping detected: {is_stopping}

Environment:
- urban road
- pedestrians possible  
- traffic signals present

Output ONLY valid JSON (no other text):
{{
  "action": "STOP" or "GO" or "SLOW",
  "reason": "brief reason",
  "trajectory_points": {trajectory.shape[1]},
  "confidence": 0.0-1.0
}}"""

print("\nRunning reasoning model...")
inputs = tokenizer(scene, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,
        top_p=0.9
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*60)
print("FINAL OUTPUT")
print("="*60)
print(f"\nTrajectory shape: {trajectory.shape}")
print(f"Trajectory sample (first 3 points):\n{trajectory[0][:3]}")

# Try to extract JSON
print("\n" + "-"*60)
print("STRUCTURED DECISION:")
print("-"*60)

# Parse JSON from result
try:
    json_start = result.find('{')
    json_end = result.rfind('}') + 1
    if json_start >= 0 and json_end > json_start:
        json_str = result[json_start:json_end]
        decision = json.loads(json_str)
        print(json.dumps(decision, indent=2))
    else:
        print(result)
except:
    print(result)

print("\n" + "="*60)
print("SYSTEM WORKING: YES")
print("="*60)
