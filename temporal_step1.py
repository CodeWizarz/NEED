import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import numpy as np

print("[1/2] Running vision model (CPU)...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
T = 5
trajectories = []

for t in range(T):
    frame = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
    output = vision_model.signatures["serving_default"](tf.convert_to_tensor(frame))
    traj = list(output.values())[0].numpy()
    trajectories.append(traj[0])
    print(f"  Frame {t+1}/{T}")

trajectories = np.array(trajectories)
np.save("/tmp/temporal_trajectories.npy", trajectories)
print(f"TEMPORAL_VISION_DONE:{trajectories.shape}")
