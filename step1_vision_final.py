import tensorflow as tf
import numpy as np
import gc

# Load vision model
print("Loading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

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

# Analyze trajectory
traj_mean = np.mean(trajectory[0], axis=0)
traj_speed = np.linalg.norm(traj_mean)

# Clear vision model to free memory
del vision_model
del vision_output
gc.collect()

# Save trajectory
np.save("/tmp/trajectory_final.npy", trajectory)
print(f"\nTrajectory saved. Mean: {traj_mean}, Speed: {traj_speed:.3f}")
print("Step 1 complete - Vision model done")
