#!/usr/bin/env python3
import os, sys, time, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

EPISODE_DIR = "/home/Balu/logs/episodes/episode_0000"
MODEL_DIR = "/tmp/alpamayo_v2"
METRICS_CSV = "/home/Balu/logs/metrics_history.csv"

def load_episode_data():
    df = pd.read_csv(f"{EPISODE_DIR}/controller.csv")
    mdf = pd.read_parquet(f"{EPISODE_DIR}/metrics.parquet")

    collision_any = mdf[mdf["name"] == "collision_any"]
    is_collision = (collision_any["values"].values > 0).astype(int)

    steps = []
    for i in range(min(len(df), 40)):
        row = df.iloc[i]
        steps.append({
            "step": i,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "vx": float(row["vx"]),
            "vy": float(row["vy"]),
            "wz": float(row["wz"]),
            "steering": float(row["u_steering_angle"]),
            "throttle": float(row["u_longitudinal_actuation"]),
            "accel": float(row["acceleration"]),
            "ref_x0": float(row["x_ref_0"]),
            "ref_y0": float(row["y_ref_0"]),
            "collision": is_collision[i] if i < len(is_collision) else 0,
        })

    progress_vals = mdf[mdf["name"] == "progress"]["values"].values
    progress_rel_vals = mdf[mdf["name"] == "progress_rel"]["values"].values
    dist_vals = mdf[mdf["name"] == "dist_to_gt_trajectory"]["values"].values
    plan_dev_vals = mdf[mdf["name"] == "plan_deviation"]["values"].values

    return steps, progress_vals, progress_rel_vals, dist_vals, plan_dev_vals

def build_training_data(steps):
    print("Building training dataset...")

    safe_steps = [s for s in steps if s["collision"] == 0]
    collision_steps = [s for s in steps if s["collision"] == 1]

    print(f"  Safe steps: {len(safe_steps)}")
    print(f"  Collision steps: {len(collision_steps)}")

    seq_len = 8
    frames = []
    targets = []

    def make_safe_trajectory(step, n=20):
        steer = max(-0.3, min(0.3, step["steering"] * 0.5))
        speed = max(2.0, min(8.0, step["vx"] * 0.6))
        return np.array([[steer, speed] for _ in range(n)], dtype=np.float32)

    def make_collision_trajectory(step, n=20):
        steer = max(-0.5, min(0.5, step["steering"] * 0.8))
        speed = max(0.5, step["vx"] * 0.3)
        return np.array([[steer, speed] for _ in range(n)], dtype=np.float32)

    for step in safe_steps:
        frame_seq = np.random.randn(seq_len, 88, 200, 3).astype(np.float32)
        frame_seq = (frame_seq - frame_seq.mean()) / (frame_seq.std() + 1e-8)
        frames.append(frame_seq)
        targets.append(make_safe_trajectory(step))

    for step in collision_steps:
        frame_seq = np.random.randn(seq_len, 88, 200, 3).astype(np.float32)
        frame_seq = (frame_seq - frame_seq.mean()) / (frame_seq.std() + 1e-8)
        frames.append(frame_seq)
        targets.append(make_collision_trajectory(step))

    for step in safe_steps[-5:]:
        frame_seq = np.random.randn(seq_len, 88, 200, 3).astype(np.float32)
        frame_seq = (frame_seq - frame_seq.mean()) / (frame_seq.std() + 1e-8)
        frames.append(frame_seq)
        targets.append(make_safe_trajectory(step))

    frames = np.array(frames, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    print(f"  Dataset: {frames.shape} -> {targets.shape}")
    return frames, targets

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 88, 200, 3)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 5, activation="relu")),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3, activation="relu")),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, 3, activation="relu")),
        tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),
        tf.keras.layers.LSTM(256, return_sequences=False),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(40),
        tf.keras.layers.Reshape((20, 2)),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model

def main():
    print("=" * 60)
    print("ALPASIM CONTINUOUS TRAINING - Iteration 0")
    print("=" * 60)

    steps, progress, progress_rel, dist, plan_dev = load_episode_data()
    print(f"\nEpisode 0 Summary:")
    print(f"  Total steps: {len(steps)}")
    print(f"  Collisions: {sum(s['collision'] for s in steps)}")
    print(f"  Progress: {progress[-1]:.1%}" if len(progress) else "  Progress: N/A")
    print(f"  Avg dist to GT: {dist.mean():.3f}m" if len(dist) else "  Dist: N/A")

    frames, targets = build_training_data(steps)

    print("\n--- Training vision model ---")
    model = create_model()
    model.summary()

    test = tf.random.normal((2, 8, 88, 200, 3))
    out = model(test)
    print(f"Output: {out.shape} (expected: (2, 20, 2))")

    history = model.fit(
        frames, targets,
        epochs=100,
        batch_size=min(8, len(frames)),
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2
    )

    print(f"\nFinal train loss: {history.history['loss'][-1]:.6f}")
    print(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Epochs: {len(history.history['loss'])}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/model.keras")
    print(f"\nModel saved to {MODEL_DIR}/model.keras")

    os.makedirs("/home/Balu/logs/episodes", exist_ok=True)
    metrics = {
        "iteration": 0,
        "episode": 0,
        "total_steps": len(steps),
        "collisions": sum(s["collision"] for s in steps),
        "collision_rate": sum(s["collision"] for s in steps) / len(steps),
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "epochs_trained": len(history.history["loss"]),
    }
    with open("/home/Balu/logs/training_metrics.json", "w") as f:
        json.dump({k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for k, v in metrics.items()}, f, indent=2)
    print(f"Metrics saved: {metrics}")

    return metrics

if __name__ == "__main__":
    main()
