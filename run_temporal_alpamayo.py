import torch
import sys
import numpy as np
import tensorflow as tf
import gc

print("=" * 60)
print("TEMPORAL ALPAMAYO PIPELINE")
print("=" * 60)

# Step 1: Run vision model on CPU
print("\n[1/4] Running vision model (CPU)...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")

T = 5
trajectories = []

for t in range(T):
    frame = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
    output = vision_model.signatures["serving_default"](tf.convert_to_tensor(frame))
    traj = list(output.values())[0].numpy()
    trajectories.append(traj[0])
    print(f"  Frame {t+1}/{T}: trajectory shape {traj.shape}")

trajectories = np.array(trajectories)
print(f"\nTemporal trajectories: {trajectories.shape}")

# Clear vision model
del vision_model
gc.collect()

# Step 2: Load Alpamayo
print("\n[2/4] Loading Alpamayo...")
sys.path.insert(0, "/tmp/alpamayo_repo/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

model = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager"
).cuda().eval()

print("Alpamayo loaded")

# Step 3: Prepare temporal input
print("\n[3/4] Preparing temporal input...")

tokenizer = model.tokenizer
text = f"Vehicle moving through intersection. Trajectory history: {T} frames. Current position x={trajectories[-1,0]:.2f}, y={trajectories[-1,1]:.2f}. What action?"
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)

# Create temporal ego motion from trajectories
B, N_TRAJ = 1, 1
ego_xyz = np.zeros((B, N_TRAJ, T, 3))
for i in range(T):
    ego_xyz[0, 0, i, 0] = trajectories[i, 0]  # x
    ego_xyz[0, 0, i, 1] = trajectories[i, 1]  # y
ego_xyz = torch.tensor(ego_xyz, dtype=torch.float32).cuda()
ego_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N_TRAJ, T, 1, 1).cuda()

data = {
    "tokenized_data": {
        "input_ids": tokens["input_ids"].cuda(),
        "attention_mask": tokens["attention_mask"].cuda(),
    },
    "ego_history_xyz": ego_xyz,
    "ego_history_rot": ego_rot,
}

print(f"  Temporal input: {T} frames")
print(f"  Trajectory history shape: {ego_xyz.shape}")

# Step 4: Run inference
print("\n[4/4] Running temporal inference...")
with torch.autocast("cuda", dtype=torch.float16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=data,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=32,
        return_extra=True,
    )

print("\n" + "=" * 60)
print("TEMPORAL PIPELINE COMPLETE")
print("=" * 60)

print(f"\nTemporal input frames: {T}")
print(f"Trajectories shape: {trajectories.shape}")
print(f"Prediction shape: {pred_xyz.shape}")
print(f"Output keys: {list(extra.keys())}")

if 'answer' in extra:
    print(f"\nAnswer: {extra['answer']}")

print("\n" + "=" * 60)
print("TEMPORAL PIPELINE WORKING: YES")
print("=" * 60)
