import numpy as np
import tensorflow as tf

# Load your trained vision model
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")

def generate_alpamayo_input():
    # Simulate multi-camera temporal input
    dummy_frames = np.random.rand(1, 8, 256, 256, 3).astype(np.float32)

    output = vision_model.signatures["serving_default"](
        tf.convert_to_tensor(dummy_frames)
    )

    trajectory = list(output.values())[0].numpy()

    # Convert to Alpamayo-style dict
    alpamayo_input = {
        "images": dummy_frames,
        "trajectory": trajectory,
        "ego_motion": np.zeros((1, 10, 6)),
    }

    return alpamayo_input
