import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf

print("=" * 60)
print("STEP 1: Vision Model (CPU)")
print("=" * 60)

# Load vision model on CPU
print("\nLoading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

# Simulate incoming frames and run vision model
T = 5
trajectories = []

print(f"\nRunning vision model for {T} frames...")
for i in range(T):
    frame = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
    out = vision_model.signatures["serving_default"](tf.convert_to_tensor(frame))
    traj = list(out.values())[0].numpy()
    trajectories.append(traj[0])
    print(f"  Frame {i+1}/{T}: trajectory shape {traj.shape}")

trajectories = np.array(trajectories)
np.save("/tmp/sim_trajectories.npy", trajectories)
print(f"\nTrajectories saved: {trajectories.shape}")
print("STEP1_DONE")
