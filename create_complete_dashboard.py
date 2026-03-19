#!/usr/bin/env python3
"""Final comprehensive dashboard"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

OUT = "/Users/Balu/Documents/NEED"

def load_ep(name):
    ep_dir = f"{OUT}/episodes/{name}"
    cdf = pd.read_csv(f"{ep_dir}/controller.csv")
    mdf = pd.read_parquet(f"{ep_dir}/metrics.parquet")
    mp4_files = [f for f in os.listdir(ep_dir) if f.endswith(".mp4")]
    frames = []
    if mp4_files:
        import cv2
        cap = cv2.VideoCapture(os.path.join(ep_dir, mp4_files[0]))
        while True:
            ret, f = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
    
    def get(name):
        rows = mdf[mdf["name"] == name]
        return rows["values"].values if len(rows) > 0 else np.zeros(40)
    
    return cdf, frames, get("collision_any"), get("progress"), get("dist_to_gt_trajectory")

plt.style.use("dark_background")
fig = plt.figure(figsize=(28, 16))
gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.25)

fig.patch.set_facecolor("#0a0a1a")
fig.suptitle("NVIDIA-Style Autonomous Driving Continuous Training Pipeline\n"
             "AlpaSim 0.42 + VAM Driver + Alpamayo Vision Model | NVIDIA L4 GPU | 3 Episodes | 120 Steps",
             fontsize=17, fontweight="bold", y=0.97, color="white")

ep_colors = ["#e74c3c", "#3498db", "#2ecc71"]
ep_names = ["episode_0000", "episode_0001", "episode_0002"]

for i, (name, color) in enumerate(zip(ep_names, ep_colors)):
    cdf, frames, collision, progress, dist = load_ep(name)
    
    ax = fig.add_subplot(gs[0, i])
    if frames:
        ax.imshow(frames[0])
    ax.set_title(f"{name}\nFrame 0 (Start)", fontsize=10, color="white")
    ax.axis("off")
    
    ax = fig.add_subplot(gs[0, i+3])
    if frames:
        mid = len(frames) // 2
        ax.imshow(frames[mid])
    ax.set_title(f"Frame {mid} (Mid)", fontsize=10, color="white")
    ax.axis("off")

ax = fig.add_subplot(gs[1, :2])
for name, color in zip(ep_names, ep_colors):
    cdf, frames, collision, progress, dist = load_ep(name)
    ax.plot(progress, "-o", markersize=3, color=color, linewidth=2, label=name)
ax.set_title("Route Progress Over Time", fontsize=11, fontweight="bold", color="white")
ax.set_xlabel("Step", color="white")
ax.set_ylabel("Progress (%)", color="white")
ax.legend()
ax.grid(alpha=0.3, color="gray")
ax.tick_params(colors="white")

ax = fig.add_subplot(gs[1, 2:4])
width = 0.25
x = np.arange(3)
coll_rates = []
for name in ep_names:
    cdf, frames, collision, progress, dist = load_ep(name)
    coll_rates.append(sum(1 for c in collision if c > 0) / 40 * 100)
bars = ax.bar(x, coll_rates, width, color=ep_colors, alpha=0.8, edgecolor="white")
for bar, rate in zip(bars, coll_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{rate:.0f}%", ha="center", fontsize=12, fontweight="bold", color="white")
ax.set_title("Collision Rate by Episode", fontsize=11, fontweight="bold", color="white")
ax.set_ylabel("Collision %", color="white")
ax.set_xticks(x)
ax.set_xticklabels(["Ep0\n(Baseline)", "Ep1\n(Best)", "Ep2\n(Retry)"], color="white")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, axis="y", color="gray")
ax.tick_params(colors="white")

ax = fig.add_subplot(gs[1, 4:])
for name, color in zip(ep_names, ep_colors):
    cdf, frames, collision, progress, dist = load_ep(name)
    ax.plot(dist, "-o", markersize=3, color=color, linewidth=2, label=name, alpha=0.8)
ax.set_title("Distance to Ground Truth Trajectory", fontsize=11, fontweight="bold", color="white")
ax.set_xlabel("Step", color="white")
ax.set_ylabel("Distance (m)", color="white")
ax.legend()
ax.grid(alpha=0.3, color="gray")
ax.tick_params(colors="white")

for i, (name, color) in enumerate(zip(ep_names, ep_colors)):
    ax = fig.add_subplot(gs[2, i])
    cdf, frames, collision, progress, dist = load_ep(name)
    x_pos = cdf["x"].values[:40]
    y_pos = cdf["y"].values[:40]
    safe_x = [x_pos[j] for j in range(len(x_pos)-1) if collision[j] == 0]
    safe_y = [y_pos[j] for j in range(len(y_pos)-1) if collision[j] == 0]
    coll_x = [x_pos[j] for j in range(len(x_pos)-1) if collision[j] > 0]
    coll_y = [y_pos[j] for j in range(len(y_pos)-1) if collision[j] > 0]
    ax.plot(safe_x, safe_y, color="#2ecc71", linewidth=3, label="Safe", alpha=0.7)
    ax.plot(coll_x, coll_y, color="#e74c3c", linewidth=3, label="Collision", alpha=0.7)
    ax.scatter(x_pos[0], y_pos[0], color="#3498db", s=150, zorder=10, marker=">")
    ax.scatter(x_pos[-1], y_pos[-1], color="#9b59b6", s=150, zorder=10, marker="s")
    n_coll = sum(1 for c in collision if c > 0)
    ax.set_title(f"{name}\n{n_coll} collisions, {40-n_coll} safe", fontsize=10, fontweight="bold", color="white")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.grid(alpha=0.3, color="gray")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")

ax = fig.add_subplot(gs[2, 3:])
ax.axis("off")

info = [
    ("PIPELINE ARCHITECTURE", "=" * 40, "#3498db"),
    ("", "", ""),
    ("  SIMULATION LAYER", "", "#3498db"),
    ("  AlpaSim 0.42 + NVIDIA NuRec", "", "white"),
    ("  Scene: clipgt-a309e228", "", "white"),
    ("  Episodes: 3 (120 total steps)", "", "white"),
    ("  Camera: 900x1000 @ 2Hz", "", "white"),
    ("", "", ""),
    ("  POLICY LAYER (VAM Driver)", "", "#3498db"),
    ("  Vision Encoder: 318M params", "", "white"),
    ("  Action Expert: 38M params", "", "white"),
    ("  Output: 20-step trajectory", "", "white"),
    ("", "", ""),
    ("  VISION MODEL (Alpamayo)", "", "#3498db"),
    ("  Architecture: CNN-LSTM", "", "white"),
    ("  Training: 123 samples (3 eps)", "", "white"),
    ("  Loss: 4.184 train / 2.216 val", "", "white"),
    ("  Epochs: 23 | Batch: 8", "", "white"),
    ("", "", ""),
    ("  RESULTS", "", "#3498db"),
    ("  Total collisions: 72/120 (60%)", "", "white"),
    ("  Best episode: ep_0001 (40%)", "", "white"),
    ("  Worst episode: 70% collision", "", "white"),
    ("", "", ""),
    ("  NEXT STEPS", "", "#3498db"),
    ("  Replace VAM with Alpamayo-R1-10B planner", "", "white"),
    ("  Download more PhysicalAI scenes", "", "white"),
    ("  Continuous loop: collect -> train -> eval", "", "white"),
]

y_pos = 0.97
for line, _, color in info:
    if line == "":
        y_pos -= 0.025
        continue
    is_header = "LAYER" in line or "ARCHITECTURE" in line or "RESULTS" in line or "NEXT" in line
    fs = 10 if is_header else 9
    fw = "bold" if is_header else "normal"
    ax.text(0.02, y_pos, line, transform=ax.transAxes, fontsize=fs,
            fontweight=fw, color=color, va="top")
    y_pos -= 0.035

plt.savefig(f"{OUT}/pipeline_complete.png", dpi=120, bbox_inches="tight",
            facecolor="#0a0a1a", edgecolor="none")
plt.close()

size = os.path.getsize(f"{OUT}/pipeline_complete.png") / 1e3
print(f"Dashboard saved: {OUT}/pipeline_complete.png ({size:.0f}KB)")

for f in sorted(os.listdir(OUT)):
    if f.endswith(".mp4") or f.endswith(".png"):
        size = os.path.getsize(f"{OUT}/{f}") / 1e6
        print(f"  {f}: {size:.1f}MB")
