import tensorflow as tf
import numpy as np


def get_dataset(batch_size=1, split="training", max_files=1, shuffle=False, drop_remainder=False):
    """
    Simulate Waymo dataset with realistic camera frames.
    Use this for testing when GCS access is not available.
    """
    
    print(f"Creating simulated Waymo dataset ({batch_size} batch, {max_files} files)")
    
    # Generate synthetic camera frames (simulating 8 cameras)
    def gen_frames():
        for i in range(max_files * 10):  # 10 frames per file
            # 8 camera views (front, front-left, front-right, side-left, side-right, rear-left, rear-right, rear)
            images = np.random.rand(8, 256, 256, 3).astype(np.float32)
            # Simulated trajectory
            trajectory = np.random.rand(20, 2).astype(np.float32) * 10 - 5
            yield images, trajectory
    
    dataset = tf.data.Dataset.from_generator(
        gen_frames,
        output_signature=(
            tf.TensorSpec(shape=(8, 256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(20, 2), dtype=tf.float32)
        )
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
