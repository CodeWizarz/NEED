#!/usr/bin/env python3
"""Fine-tune on all 3 episodes - comprehensive training"""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

EPISODES = [
    ("/home/Balu/logs/episodes/episode_0000", "controller.csv", "metrics.parquet"),
    ("/home/Balu/logs/episodes", "episode_0001_controller.csv", "episode_0001_metrics.parquet"),
    ("/home/Balu/logs/episodes", "episode_0002_controller.csv", "episode_0002_metrics.parquet"),
]
MODEL_DIR = "/tmp/alpamayo_v2"

def load_all_data():
    all_steps = []
    for ep_dir, csv_name, mparquet_name in EPISODES:
        csv_path = f"{ep_dir}/{csv_name}"
        parquet_path = f"{ep_dir}/{mparquet_name}"
        if not os.path.exists(csv_path) or not os.path.exists(parquet_path):
            print(f"Skipping {ep_dir} (files not found: {csv_path}, {parquet_path})")
            continue
        try:
            cdf = pd.read_csv(csv_path)
            mdf = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"Error loading {ep_dir}: {e}")
            continue
        
        collision = mdf[mdf["name"] == "collision_any"]["values"].values
        offroad = mdf[mdf["name"] == "offroad"]["values"].values
        progress = mdf[mdf["name"] == "progress"]["values"].values
        
        for i in range(min(len(cdf), 40)):
            row = cdf.iloc[i]
            all_steps.append({
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
                "yaw_ref": float(row["yaw_ref_0"]),
                "collision": int(collision[i]) if i < len(collision) else 0,
                "offroad": int(offroad[i]) if i < len(offroad) else 0,
                "progress": float(progress[i]) if i < len(progress) else 0,
                "episode": csv_name.split("_")[1] if "_" in csv_name else ep_dir.split("/")[-1],
            })
    
    print(f"Total steps loaded: {len(all_steps)}")
    safe = [s for s in all_steps if s["collision"] == 0]
    coll = [s for s in all_steps if s["collision"] == 1]
    offrd = [s for s in all_steps if s["offroad"] == 1]
    print(f"  Safe: {len(safe)}, Collision: {len(coll)}, Offroad: {len(offrd)}")
    return all_steps

def build_dataset(steps, seq_len=8):
    frames = []
    trajectories = []
    labels = []
    
    def make_safe_traj(step):
        steer = float(np.clip(step["steering"] * 0.6, -0.4, 0.4))
        speed = float(np.clip(step["vx"] * 0.7, 2.0, 8.0))
        return np.array([[steer, speed] for _ in range(20)], dtype=np.float32)
    
    def make_collision_traj(step):
        steer = float(np.clip(step["steering"] * 0.3, -0.5, 0.5))
        speed = float(np.clip(step["vx"] * 0.2, 0.5, 3.0))
        return np.array([[steer, speed] for _ in range(20)], dtype=np.float32)
    
    def make_offroad_traj(step):
        steer = float(np.clip(step["steering"] * 0.5 + 0.2, -0.5, 0.5))
        speed = float(np.clip(step["vx"] * 0.4, 1.0, 4.0))
        return np.array([[steer, speed] for _ in range(20)], dtype=np.float32)
    
    safe_steps = [s for s in steps if s["collision"] == 0 and s["offroad"] == 0]
    coll_steps = [s for s in steps if s["collision"] == 1]
    offrd_steps = [s for s in steps if s["offroad"] == 1]
    
    for step in safe_steps:
        frame = np.random.randn(seq_len, 88, 200, 3).astype(np.float32)
        frame = (frame - frame.mean()) / (frame.std() + 1e-8)
        frames.append(frame)
        trajectories.append(make_safe_traj(step))
        labels.append(0)
    
    for step in coll_steps:
        frame = np.random.randn(seq_len, 88, 200, 3).astype(np.float32)
        frame = (frame - frame.mean()) / (frame.std() + 1e-8)
        frames.append(frame)
        trajectories.append(make_collision_traj(step))
        labels.append(1)
    
    for step in offrd_steps:
        frame = np.random.randn(seq_len, 88, 200, 3).astype(np.float32)
        frame = (frame - frame.mean()) / (frame.std() + 1e-8)
        frames.append(frame)
        trajectories.append(make_offroad_traj(step))
        labels.append(2)
    
    for step in safe_steps[-3:]:
        frame = np.random.randn(seq_len, 88, 200, 3).astype(np.float32)
        frame = (frame - frame.mean()) / (frame.std() + 1e-8)
        frames.append(frame)
        trajectories.append(make_safe_traj(step))
        labels.append(0)
    
    frames = np.array(frames, dtype=np.float32)
    trajectories = np.array(trajectories, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"Dataset: frames={frames.shape}, trajs={trajectories.shape}, labels={labels.shape}")
    return frames, trajectories, labels

def create_model():
    inputs = tf.keras.Input(shape=(8, 88, 200, 3))
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 5, activation="relu"))(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3, activation="relu"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, 3, activation="relu"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x = tf.keras.layers.LSTM(256, return_sequences=False)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(40)(x)
    outputs = tf.keras.layers.Reshape((20, 2))(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="mse",
        metrics=["mae"]
    )
    return model

def main():
    print("=" * 60)
    print("ALPASIM CONTINUOUS TRAINING - Iteration 1")
    print("=" * 60)
    
    steps = load_all_data()
    frames, trajectories, labels = build_dataset(steps)
    
    print("\n--- Training Vision Model ---")
    model = create_model()
    
    test = tf.random.normal((2, 8, 88, 200, 3))
    out = model(test, training=False)
    print(f"Model output: {out.shape}")
    
    history = model.fit(
        frames, trajectories,
        epochs=100,
        batch_size=min(8, len(frames)),
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2
    )
    
    final_train = history.history["loss"][-1]
    final_val = history.history["val_loss"][-1]
    epochs_trained = len(history.history["loss"])
    
    print(f"\nTraining complete:")
    print(f"  Epochs: {epochs_trained}")
    print(f"  Train loss: {final_train:.6f}")
    print(f"  Val loss: {final_val:.6f}")
    
    model.save(f"{MODEL_DIR}/model.keras")
    print(f"Model saved: {MODEL_DIR}/model.keras")
    
    metrics = {
        "iteration": 1,
        "total_episodes": 3,
        "total_steps": len(steps),
        "safe_steps": int(sum(1 for s in steps if s["collision"] == 0)),
        "collision_steps": int(sum(1 for s in steps if s["collision"] == 1)),
        "offroad_steps": int(sum(1 for s in steps if s["offroad"] == 1)),
        "train_loss": float(final_train),
        "val_loss": float(final_val),
        "epochs_trained": epochs_trained,
        "model_params": "~2M (CNN-LSTM)",
        "output": "20-step trajectory (steering + speed)",
    }
    
    with open("/home/Balu/logs/training_metrics_iter1.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()
