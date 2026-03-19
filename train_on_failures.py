#!/usr/bin/env python3
import os
import json
import numpy as np
import tensorflow as tf

FAILURE_LOGS = "/home/Balu/logs"

class TrajectoryVisionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.frame_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 5, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
        ])
        self.temporal = tf.keras.layers.LSTM(256, return_sequences=False)
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc2 = tf.keras.layers.Dense(40)
        self.reshape = tf.keras.layers.Reshape((20, 2))

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size * 8, 88, 200, 3])
        x = self.frame_encoder(x)
        x = tf.reshape(x, [batch_size, 8, 128])
        x = self.temporal(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.reshape(x)
        return x

def load_failure_cases():
    failures = []
    for fname in os.listdir(FAILURE_LOGS):
        if not fname.startswith('episode_') or not fname.endswith('.json'):
            continue
        with open(os.path.join(FAILURE_LOGS, fname)) as f:
            for line in f:
                step = json.loads(line)
                if step.get('collision') or step.get('action') == 'STOP':
                    failures.append({
                        'action': step.get('action'),
                        'speed': step.get('speed', 0.0),
                        'collision': step.get('collision', False),
                    })
    return failures

def create_synthetic_frames(n=8, h=88, w=200, c=3):
    return np.random.randn(n, h, w, c).astype(np.float32)

def build_dataset(failures, seq_len=8):
    frames = []
    trajectories = []
    for _ in failures:
        seq = create_synthetic_frames(seq_len)
        frames.append(seq)
        safe_action = np.random.uniform(-0.5, 0.5)
        safe_speed = 5.0
        traj = np.array([[safe_action, safe_speed] for _ in range(20)], dtype=np.float32)
        trajectories.append(traj)
    frames = np.array(frames, dtype=np.float32)
    trajectories = np.array(trajectories, dtype=np.float32)
    return frames, trajectories

def main():
    failures = load_failure_cases()
    print(f"Loaded {len(failures)} failure cases")

    if not failures:
        print("No failures found.")
        return

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = TrajectoryVisionModel()
        model.build(input_shape=(None, 8, 88, 200, 3))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='mse',
            metrics=['mae']
        )

    test_input = tf.random.normal((1, 8, 88, 200, 3))
    test_output = model(test_input, training=False)
    print(f"Model output shape: {test_output.shape}")

    frames, trajectories = build_dataset(failures)
    print(f"Dataset: {frames.shape} -> {trajectories.shape}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    ]

    model.fit(
        frames, trajectories,
        epochs=100,
        batch_size=min(8, len(failures)),
        validation_split=0.2,
        callbacks=callbacks,
        verbose=2
    )

    model.save("/tmp/alpamayo_v2")
    print("Fine-tuned model saved to /tmp/alpamayo_v2")

if __name__ == "__main__":
    main()
