import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import sys
import numpy as np
import tensorflow as tf
import gc
from scipy import interpolate

print("=" * 60)
print("VISION + ALPAMAYO INTEGRATION PIPELINE")
print("=" * 60)

# Load vision model on CPU (TF uses CPU only)
print("\n[1/5] Loading vision model (CPU)...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

# Generate input for vision model
print("\n[2/5] Generating camera input...")
dummy_frames = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
print(f"Camera input shape: {dummy_frames.shape}")

# Run vision model on CPU
print("\n[3/5] Running vision model...")
vision_output = vision_model.signatures["serving_default"](
    tf.convert_to_tensor(dummy_frames)
)

trajectory = list(vision_output.values())[0].numpy()
print(f"Vision trajectory shape: {trajectory.shape}")
print(f"Trajectory sample:\n{trajectory[0][:3]}")

# Clear TF from memory
del vision_model
del vision_output
gc.collect()

# Now load Alpamayo on GPU
print("\n[4/5] Loading Alpamayo on GPU...")
sys.path.insert(0, "/tmp/alpamayo_repo/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

alpamayo_model = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager",
)
alpamayo_model = alpamayo_model.cuda()
alpamayo_model.eval()
print("Alpamayo loaded")

# Prepare Alpamayo input using vision trajectory
tokenizer = alpamayo_model.tokenizer
text = f"Road scene: vehicle trajectory x={trajectory[0,-1,0]:.2f}, y={trajectory[0,-1,1]:.2f}. What action?"
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)

input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Use vision trajectory as ego motion
B, N_TRAJ, T = 1, 1, 16

# Interpolate vision trajectory to 16 timesteps
t_vision = np.linspace(0, 1, trajectory.shape[1])
t_ego = np.linspace(0, 1, T)
ego_xyz = np.zeros((B, N_TRAJ, T, 3))
ego_xyz[0, 0, :, 0] = interpolate.interp1d(t_vision, trajectory[0, :, 0], kind='linear')(t_ego)
ego_xyz[0, 0, :, 1] = interpolate.interp1d(t_vision, trajectory[0, :, 1], kind='linear')(t_ego)
ego_xyz = torch.tensor(ego_xyz, dtype=torch.float32).cuda()

# Identity rotation
ego_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N_TRAJ, T, 1, 1).cuda()

data = {
    "tokenized_data": {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
    },
    "ego_history_xyz": ego_xyz,
    "ego_history_rot": ego_rot,
}

print(f"\nAlpamayo input - Text: {text}")
print(f"Ego XYZ shape: {ego_xyz.shape}")

# Run Alpamayo
print("\n[5/5] Running Alpamayo inference...")
with torch.autocast("cuda", dtype=torch.float16):
    pred_xyz, pred_rot, extra = alpamayo_model.sample_trajectories_from_data_with_vlm_rollout(
        data=data,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=32,
        return_extra=True,
    )

print("\n" + "=" * 60)
print("INTEGRATION COMPLETE")
print("=" * 60)

print(f"\nVision Trajectory Shape: {trajectory.shape}")
print(f"Alpamayo Prediction Shape: {pred_xyz.shape}")

print(f"\nOutput Keys: {list(extra.keys())}")
if 'answer' in extra:
    print(f"Answer: {extra['answer'][:200]}...")

print("\n" + "=" * 60)
print("FULL PIPELINE WORKING: YES")
print("=" * 60)
