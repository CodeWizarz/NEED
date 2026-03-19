import torch
import sys
import numpy as np
import tensorflow as tf
import gc

print("=" * 60)
print("WAYMO + ALPAMAYO TEMPORAL PIPELINE")
print("=" * 60)

# Step 1: Stream Waymo frames from GCS (CPU)
print("\n[1/4] Streaming Waymo frames from GCS...")

# Add current dir to path for waymo_frame_parser
sys.path.insert(0, "/tmp")
from waymo_frame_parser import get_dataset

# Load Waymo dataset (stream from GCS)
dataset = get_dataset(
    batch_size=1,
    split="training",
    max_files=1,  # Only 1 shard for testing
    shuffle=False,
    drop_remainder=False
)

# Collect temporal sequence
T = 5
frames = []
trajectories = []

print(f"Collecting {T} frames...")

for i, (images, traj) in enumerate(dataset.take(T)):
    if images is not None:
        frames.append(images.numpy())
        trajectories.append(traj.numpy())
        print(f"  Frame {i+1}/{T}: {images.shape}")

if len(frames) == 0:
    print("No valid frames found!")
    exit(1)

frames = np.concatenate(frames, axis=0)
trajectories = np.concatenate(trajectories, axis=0)

print(f"\nWaymo frames: {frames.shape}")
print(f"Waymo trajectories: {trajectories.shape}")

# Clear dataset
del dataset
gc.collect()

# Step 2: Run vision model on frames (CPU)
print("\n[2/4] Running vision model on frames...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")

vision_trajectories = []
for i in range(len(frames)):
    frame = frames[i:i+1]  # Add batch dimension
    output = vision_model.signatures["serving_default"](tf.convert_to_tensor(frame))
    traj = list(output.values())[0].numpy()
    vision_trajectories.append(traj[0])

vision_trajectories = np.array(vision_trajectories)
print(f"Vision trajectories: {vision_trajectories.shape}")

# Clear vision model
del vision_model
gc.collect()

# Step 3: Load Alpamayo (GPU)
print("\n[3/4] Loading Alpamayo on GPU...")
sys.path.insert(0, "/tmp/alpamayo_repo/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

model = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager"
).cuda().eval()

print("Alpamayo loaded")

# Step 4: Prepare temporal input
print("\n[4/4] Running Alpamayo inference...")

tokenizer = model.tokenizer
current_x = float(vision_trajectories[-1, -1, 0])
current_y = float(vision_trajectories[-1, -1, 1])
text = f"Real Waymo data: vehicle with {T} frame temporal history. Current position x={current_x:.2f}, y={current_y:.2f}. What action?"
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)

B, N_TRAJ = 1, 1
ego_xyz = np.zeros((B, N_TRAJ, T, 3))
for i in range(T):
    ego_xyz[0, 0, i, 0] = float(vision_trajectories[i, -1, 0])
    ego_xyz[0, 0, i, 1] = float(vision_trajectories[i, -1, 1])
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

print(f"Input: {T} temporal frames from Waymo")
print("Running inference...")

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
print("WAYMO + ALPAMAYO PIPELINE COMPLETE")
print("=" * 60)

print(f"\nWaymo frames shape: {frames.shape}")
print(f"Vision trajectories: {vision_trajectories.shape}")
print(f"Alpamayo prediction: {pred_xyz.shape}")
print(f"Output keys: {list(extra.keys())}")

if 'answer' in extra:
    print(f"\nAnswer: {extra['answer']}")

print("\n" + "=" * 60)
print("STREAMING PIPELINE WORKING: YES")
print("=" * 60)
