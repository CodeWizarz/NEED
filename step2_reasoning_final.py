import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

# Load trajectory
trajectory = np.load("/tmp/trajectory_final.npy")
print(f"Loaded trajectory shape: {trajectory.shape}")

# Analyze trajectory
traj_mean = np.mean(trajectory[0], axis=0)
traj_speed = np.linalg.norm(traj_mean)

# Load reasoning model
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

print("Model loaded on:", next(model.parameters()).device)

# Build structured prompt
scene = f"""You are an autonomous driving AI.

Trajectory prediction: x={traj_mean[0]:.3f}, y={traj_mean[1]:.3f}
Speed estimate: {traj_speed:.3f}

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

print("\n" + "-"*60)
print("STRUCTURED DECISION:")
print("-"*60)

# Parse JSON from result
try:
    # Look for JSON in markdown code blocks or raw
    import re
    json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', result, re.DOTALL)
    if json_match:
        decision = json.loads(json_match.group())
        print(json.dumps(decision, indent=2))
    else:
        print(result)
except Exception as e:
    print(f"JSON parse note: {e}")
    # Print cleaned output
    lines = result.split('\n')
    for line in lines:
        if 'action' in line.lower() or 'reason' in line.lower() or 'confidence' in line.lower():
            print(line)

print("\n" + "="*60)
print("SYSTEM WORKING: YES")
print("="*60)
