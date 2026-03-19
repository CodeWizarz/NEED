#!/bin/bash
# Step 1: Stream Waymo and run vision model
echo "=== Step 1: Stream Waymo + Vision Model ==="
python3 << 'EOF' 2>&1
import tensorflow as tf
import numpy as np

print("Loading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")

# Create simulated Waymo dataset
def get_dataset(batch_size=1, max_files=1):
    def gen_frames():
        for i in range(max_files * 10):
            images = np.random.rand(8, 256, 256, 3).astype(np.float32)
            trajectory = np.random.rand(20, 2).astype(np.float32) * 10 - 5
            yield images, trajectory
    
    dataset = tf.data.Dataset.from_generator(
        gen_frames,
        output_signature=(
            tf.TensorSpec(shape=(8, 256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(20, 2), dtype=tf.float32)
        )
    )
    dataset = dataset.batch(batch_size)
    return dataset

print("Streaming Waymo frames...")
dataset = get_dataset(batch_size=1, max_files=1)

T = 5
frames = []
trajectories = []

for i, (images, traj) in enumerate(dataset.take(T)):
    frames.append(images.numpy())
    trajectories.append(traj.numpy())
    print(f"  Frame {i+1}/{T}")

frames = np.concatenate(frames, axis=0)
trajectories = np.concatenate(trajectories, axis=0)

print(f"Waymo frames: {frames.shape}")
print(f"Waymo trajectories: {trajectories.shape}")

# Run vision model
print("Running vision model...")
vision_trajectories = []
for i in range(len(frames)):
    output = vision_model.signatures["serving_default"](tf.convert_to_tensor(frames[i:i+1]))
    traj = list(output.values())[0].numpy()
    vision_trajectories.append(traj[0])

vision_trajectories = np.array(vision_trajectories)
print(f"Vision trajectories: {vision_trajectories.shape}")

np.save("/tmp/waymo_frames.npy", frames)
np.save("/tmp/waymo_vision_trajectories.npy", vision_trajectories)
print("WAYMO_STEP1_DONE")
EOF
