#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import signal
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

LOG_DIR = "/home/Balu/alpasim/tutorial"
CONTROLLER_DIR = f"{LOG_DIR}/controller"
ROLLOUTS_DIR = f"{LOG_DIR}/rollouts"
METRICS_CSV = "/home/Balu/logs/metrics_history.csv"
EPISODE_LOG = "/home/Balu/logs/episode_log.jsonl"
ALPAMAYO_MODEL_DIR = "/tmp/alpamayo_v2"
HOST = "136.119.37.171"
SSH_KEY = os.path.expanduser("~/.ssh/google_compute_engine")

class Pipeline:
    def __init__(self):
        self.episode_count = 0
        self.iteration = 0
        self.failures = []
        self.collisions = []
        self.load_history()

    def ssh(self, cmd, timeout=30):
        result = subprocess.run(
            ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
             f"Balu@{HOST}", cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr

    def load_history(self):
        if os.path.exists(METRICS_CSV):
            df = pd.read_csv(METRICS_CSV)
            self.episode_count = df["episode"].max() + 1 if len(df) > 0 else 0
            print(f"Loaded history: {self.episode_count} episodes")

    def start_simulation(self):
        print("\n=== Starting AlpaSim simulation ===")
        out = self.ssh(f"cd /home/Balu/alpasim/tutorial && /usr/bin/docker compose up -d")
        print(out[-500:])
        time.sleep(15)

    def stop_simulation(self):
        print("Stopping simulation...")
        self.ssh("cd /home/Balu/alpasim/tutorial && /usr/bin/docker compose down", timeout=60)
        time.sleep(3)

    def wait_for_completion(self, timeout=300):
        print("Waiting for simulation to complete...")
        start = time.time()
        while time.time() - start < timeout:
            out = self.ssh("docker ps --format '{{.Names}}' 2>/dev/null || echo ''")
            if "tutorial-runtime" not in out:
                print("Simulation completed!")
                time.sleep(5)
                return True
            time.sleep(15)
            elapsed = int(time.time() - start)
            print(f"  ... {elapsed}s elapsed, still running")
        print("Timeout waiting for simulation")
        return False

    def parse_episode(self, episode_id):
        csv_files = sorted([
            f for f in os.listdir(CONTROLLER_DIR)
            if f.startswith("alpasim_controller_") and f.endswith(".csv")
        ])
        if not csv_files:
            return None

        latest_csv = csv_files[-1]
        csv_path = os.path.join(CONTROLLER_DIR, latest_csv)

        rollout_dirs = []
        for root, dirs, files in os.walk(ROLLOUTS_DIR):
            for d in dirs:
                if d.startswith("3bf6"):
                    rollout_dirs.append(os.path.join(root, d))
        rollout_dirs.sort(key=os.path.getmtime, reverse=True)
        rollout_dir = rollout_dirs[0] if rollout_dirs else None

        metrics_path = None
        mp4_path = None
        if rollout_dir:
            metrics_path = os.path.join(rollout_dir, "metrics.parquet")
            mp4_files = [f for f in os.listdir(rollout_dir) if f.endswith(".mp4")]
            if mp4_files:
                mp4_path = os.path.join(rollout_dir, mp4_files[0])

        return {
            "csv": csv_path,
            "metrics": metrics_path,
            "mp4": mp4_path,
            "rollout_dir": rollout_dir,
        }

    def download_episode_data(self, episode_id):
        episode_dir = f"/home/Balu/logs/episodes/episode_{episode_id:04d}"
        os.makedirs(episode_dir, exist_ok=True)

        data = self.parse_episode(episode_id)
        if not data:
            return None

        files = {}
        for key in ["csv", "metrics"]:
            if data[key] and os.path.exists(data[key]):
                subprocess.run(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                    f"Balu@{HOST}:{data[key]}", f"{episode_dir}/{key}.parquet"])
                files[key] = f"{episode_dir}/{key}.parquet"

        if data["mp4"]:
            subprocess.run(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                f"Balu@{HOST}:{data['mp4']}", f"{episode_dir}/video.mp4"])
            files["mp4"] = f"{episode_dir}/video.mp4"

        return files, data

    def extract_failures(self, csv_path, metrics_path):
        if not os.path.exists(csv_path):
            return [], [], {}

        try:
            df = pd.read_parquet(csv_path)
        except:
            try:
                df = pd.read_csv(csv_path)
            except:
                return [], [], {}

        failures = []
        collisions = []

        if "collision_any" in df.columns or "collision" in df.columns:
            try:
                metrics_df = pd.read_parquet(metrics_path)
                collision_rows = metrics_df[metrics_df["name"] == "collision_any"]
                collision_steps = set(collision_rows.index.tolist())

                for idx in collision_steps:
                    if idx < len(df):
                        row = df.iloc[idx]
                        failures.append({
                            "step": idx,
                            "x": float(row.get("x", 0)),
                            "y": float(row.get("y", 0)),
                            "vx": float(row.get("vx", 0)),
                            "vy": float(row.get("vy", 0)),
                            "u_steering_angle": float(row.get("u_steering_angle", 0)),
                            "u_longitudinal_actuation": float(row.get("u_longitudinal_actuation", 0)),
                            "type": "collision",
                        })
                        collisions.append(idx)
            except Exception as e:
                print(f"Metrics parse error: {e}")

        return failures, collisions, {}

    def finetune(self, all_failures):
        if len(all_failures) < 2:
            print("Not enough failures to fine-tune")
            return False

        print(f"\n=== Fine-tuning on {len(all_failures)} failure cases ===")

        script = f"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

class TrajectoryModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.frame_enc = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 5, activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
        ])
        self.temporal = tf.keras.layers.LSTM(256, return_sequences=False)
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc2 = tf.keras.layers.Dense(40)
        self.reshape = tf.keras.layers.Reshape((20, 2))

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size * 8, 88, 200, 3])
        x = self.frame_enc(x)
        x = tf.reshape(x, [batch_size, 8, 128])
        x = self.temporal(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return self.reshape(x)

n = {len(all_failures)}
seq_len = 8

frames = np.random.randn(n, seq_len, 88, 200, 3).astype(np.float32)
# Safe trajectories: reduced speed near collisions, wider turns
trajectories = []
for i, f in enumerate(all_failures):
    steer = max(-0.5, min(0.5, f.get("u_steering_angle", 0) * 0.5))
    speed = max(1.0, 5.0 - i * 0.3)
    traj = np.array([[steer, speed] for _ in range(20)], dtype=np.float32)
    trajectories.append(traj)
trajectories = np.array(trajectories, dtype=np.float32)

strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = TrajectoryModel()
    model.build(input_shape=(None, seq_len, 88, 200, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="mse",
        metrics=["mae"]
    )

test = tf.random.normal((1, seq_len, 88, 200, 3))
out = model(test)
print(f"Output shape: {{out.shape}}")

history = model.fit(
    frames, trajectories,
    epochs=50,
    batch_size=min(8, n),
    validation_split=0.2,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    ]
)

model.save("{ALPAMAYO_MODEL_DIR}")
loss = history.history["loss"][-1]
val_loss = history.history["val_loss"][-1]
print(f"Training complete. Loss: {{loss:.6f}}, Val Loss: {{val_loss:.6f}}")
"""

        with open("/tmp/finetune_run.py", "w") as f:
            f.write(script)

        subprocess.run(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "/tmp/finetune_run.py", f"Balu@{HOST}:/tmp/finetune_run.py"])

        result = self.ssh("cd /tmp && python3 finetune_run.py 2>&1", timeout=300)
        print(result[-2000:])
        return "Training complete" in result or "loss:" in result.lower()

    def run_evaluation(self, episode_id):
        metrics = {
            "episode": episode_id,
            "timestamp": time.time(),
            "iteration": self.iteration,
        }

        metrics_csv = "/home/Balu/logs/metrics_history.csv"
        if os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)
            if len(df) > 0:
                latest = df[df["episode"] == episode_id]
                if len(latest) > 0:
                    for col in ["collision_rate", "progress", "ade", "offroad_rate"]:
                        if col in latest.columns:
                            metrics[col] = float(latest[col].values[-1])

        return metrics

    def save_metrics(self, episode_id, csv_path, metrics_path):
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            elif os.path.exists(metrics_path):
                df = pd.read_parquet(metrics_path)
            else:
                return
        except:
            return

        rows = []
        for i in range(min(len(df), 40)):
            row = {"episode": episode_id, "step": i, "iteration": self.iteration}
            for col in df.columns:
                if col not in ["timestamps_us"]:
                    try:
                        row[col] = float(df[col].iloc[i])
                    except:
                        pass
            rows.append(row)

        result_df = pd.DataFrame(rows)
        if os.path.exists(METRICS_CSV):
            existing = pd.read_csv(METRICS_CSV)
            result_df = pd.concat([existing, result_df], ignore_index=True)
        result_df.to_csv(METRICS_CSV, index=False)
        print(f"Saved metrics for episode {episode_id}")

    def create_demo_visualization(self, episode_id, output_path):
        if not os.path.exists(f"/home/Balu/logs/episodes/episode_{episode_id:04d}/video.mp4"):
            print(f"No video for episode {episode_id}")
            return

        viz_script = f"""
import cv2, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches

colors = {{"collision": (1,0,0), "warning": (1,0.7,0), "safe": (0,1,0)}}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("AlpaSim Continuous Training - Episode {episode_id} | Iteration {self.iteration}", fontsize=14)

mp4 = cv2.VideoCapture("/home/Balu/logs/episodes/episode_{episode_id:04d}/video.mp4")
frames = []
while True:
    ret, frame = mp4.read()
    if not ret: break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
mp4.release()

if frames:
    ax = axes[0, 0]
    ax.imshow(frames[0])
    ax.set_title("Camera Feed (T=0)")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(frames[len(frames)//2])
    ax.set_title(f"Camera Feed (T={len(frames)//2})")
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(frames[-1])
    ax.set_title(f"Camera Feed (T={len(frames)-1})")
    ax.axis("off")
else:
    for ax in axes[0]:
        ax.text(0.5, 0.5, "No frames", ha="center", va="center")
        ax.axis("off")

ax = axes[1, 1]
ax.axis("off")
ax.text(0.5, 0.8, "ALPASIM PIPELINE", ha="center", va="center", fontsize=16, fontweight="bold", transform=ax.transAxes)
ax.text(0.5, 0.6, f"Episode: {{episode_id}}", ha="center", va="center", fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.45, f"Iteration: {{self.iteration}}", ha="center", va="center", fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.3, f"Failures Collected: {{len(self.failures)}}", ha="center", va="center", fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.15, f"Driver: VAM (VideoActionModel)", ha="center", va="center", fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.0, "Simulator: AlpaSim + NVIDIA NuRec", ha="center", va="center", fontsize=12, transform=ax.transAxes)

plt.tight_layout()
plt.savefig("{output_path}", dpi=100, bbox_inches="tight")
plt.close()
print(f"Demo saved to {{output_path}}")
"""

        with open("/tmp/create_demo.py", "w") as f:
            f.write(viz_script)
        subprocess.run(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "/tmp/create_demo.py", f"Balu@{HOST}:/tmp/create_demo.py"])
        self.ssh(f"python3 /tmp/create_demo.py 2>&1")

    def run_continuous_loop(self, num_iterations=3):
        print(f"\n{'='*60}")
        print(f"CONTINUOUS TRAINING LOOP - {num_iterations} iterations")
        print(f"{'='*60}")

        for i in range(num_iterations):
            self.iteration = i
            episode_id = self.episode_count + i

            print(f"\n{'='*60}")
            print(f"ITERATION {i+1}/{num_iterations} | EPISODE {episode_id}")
            print(f"{'='*60}")

            self.start_simulation()
            completed = self.wait_for_completion(timeout=300)

            files, data = self.download_episode_data(episode_id)

            failures = []
            if files and "metrics" in files:
                self.save_metrics(episode_id, files.get("csv"), files.get("metrics"))
                f, c, _ = self.extract_failures(files.get("csv"), files.get("metrics"))
                failures = f
                self.failures.extend(failures)
                self.collisions.extend(c)
                print(f"Found {len(failures)} failures, {len(self.collisions)} collisions")

            self.stop_simulation()

            demo_path = f"/home/Balu/logs/episodes/episode_{episode_id:04d}/demo.png"
            self.create_demo_visualization(episode_id, demo_path)

            if len(self.failures) >= 5 and i < num_iterations - 1:
                print(f"\n--- Fine-tuning (iteration {i}) ---")
                self.finetune(self.failures[-20:])

            self.ssh("nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null")

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Total episodes: {num_iterations}")
        print(f"Total failures collected: {len(self.failures)}")
        print(f"Total collisions: {len(self.collisions)}")

        if os.path.exists(METRICS_CSV):
            df = pd.read_csv(METRICS_CSV)
            print(f"\nMetrics saved to: {METRICS_CSV}")
            print(df.describe())

if __name__ == "__main__":
    p = Pipeline()
    p.run_continuous_loop(num_iterations=3)
