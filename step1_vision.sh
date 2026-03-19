#!/bin/bash
# Pipeline runner - Step 1: Vision Model Only

cd /home/vickr

python3 << 'EOF'
import tensorflow as tf
import numpy as np

print("=" * 50)
print("STEP 1: VISION MODEL")
print("=" * 50)

# Load vision model
print("\nLoading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

# Fake camera input (simulate)
print("\nGenerating fake camera input...")
dummy_input = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)
print(f"Input shape: {dummy_input.shape}")

# Run vision model
print("\nRunning vision model...")
vision_output = vision_model.signatures["serving_default"](
    tf.convert_to_tensor(dummy_input)
)

trajectory = list(vision_output.values())[0].numpy()
print(f"Vision output shape: {trajectory.shape}")
print(f"Vision output sample: {trajectory[0][:3]}")

# Save trajectory for next step
np.save("/tmp/trajectory.npy", trajectory)
print("\nTrajectory saved to /tmp/trajectory.npy")

print("\n" + "=" * 50)
print("VISION MODEL: SUCCESS")
print("=" * 50)
EOF
