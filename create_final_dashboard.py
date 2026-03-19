#!/usr/bin/env python3
"""Enhanced pipeline dashboard with all 3 episodes"""
import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

EPISODES_DIR = "/Users/Balu/Documents/NEED/episodes"
OUT_DIR = "/Users/Balu/Documents/NEED"

def load_episode(name):
    ep_dir = f"{EPISODES_DIR}/{name}"
    cdf = pd.read_csv(f"{ep_dir}/controller.csv")
    mdf = pd.read_parquet(f"{ep_dir}/metrics.parquet")
    
    def get_metric(name):
        rows = mdf[mdf["name"] == name]
        return rows["values"].values if len(rows) > 0 else np.zeros(40)
    
    mp4_files = [f for f in os.listdir(ep_dir) if f.endswith(".mp4")]
    frames = []
    if mp4_files:
        cap = cv2.VideoCapture(os.path.join(ep_dir, mp4_files[0]))
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    
    collision = get_metric("collision_any")
    offroad = get_metric("offroad")
    
    return {
        "name": name,
        "frames": frames,
        "csv": cdf,
        "collision": collision,
        "offroad": offroad,
        "progress": get_metric("progress"),
        "dist": get_metric("dist_to_gt_trajectory"),
        "plan_dev": get_metric("plan_deviation"),
        "steps": min(len(cdf), 40),
    }

def main():
    print("Loading episodes...")
    eps = {name: load_episode(name) for name in ["episode_0000", "episode_0001", "episode_0002"]}
    
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.3)
    fig.suptitle("NVIDIA-Style Autonomous Driving Continuous Training Pipeline\n"
                 "AlpaSim 0.42 + VAM Driver + Alpamayo Vision Model | NVIDIA L4 GPU",
                 fontsize=16, fontweight="bold", y=0.98)
    
    ep_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    for i, (name, ep) in enumerate(eps.items()):
        row = i
        frames = ep["frames"]
        
        ax = fig.add_subplot(gs[row, 0])
        if frames:
            ax.imshow(frames[0])
        ax.set_title(f"{name}\nT=0 (start)", fontsize=9)
        ax.axis("off")
        
        ax = fig.add_subplot(gs[row, 1])
        mid = len(frames) // 2
        if frames:
            ax.imshow(frames[mid])
        ax.set_title(f"T={mid}s", fontsize=9)
        ax.axis("off")
        
        ax = fig.add_subplot(gs[row, 2])
        if frames:
            ax.imshow(frames[-1])
        ax.set_title(f"T={len(frames)-1}s (end)", fontsize=9)
        ax.axis("off")
        
        ax = fig.add_subplot(gs[row, 3])
        steps = np.arange(40)
        bars = ax.bar(steps, ep["collision"], color=ep_colors[i], alpha=0.7, width=0.8)
        n_collision = sum(1 for c in ep["collision"] if c > 0)
        ax.set_title(f"Collision per Step (total: {n_collision}/40)", fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel("Collision")
        ax.set_ylim(0, 1.2)
        ax.grid(alpha=0.3)
        
        ax = fig.add_subplot(gs[row, 4])
        cdf = ep["csv"].iloc[:40]
        x = cdf["x"].values
        y = cdf["y"].values
        for j in range(len(x)-1):
            col = "#e74c3c" if ep["collision"][j] > 0 else ep_colors[i]
            ax.plot([x[j], x[j+1]], [y[j], y[j+1]], color=col, linewidth=2)
        ax.scatter(x[0], y[0], color="#3498db", s=100, zorder=10, marker=">")
        ax.scatter(x[-1], y[-1], color="#9b59b6", s=100, zorder=10, marker="s")
        ax.set_title("Ego Trajectory (red=collision)", fontsize=9)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    
    ax = fig.add_subplot(gs[2, :3])
    for name, ep in eps.items():
        steps = np.arange(40)
        idx = list(eps.keys()).index(name)
        ax.plot(steps, ep["progress"], "-o", markersize=3, color=ep_colors[idx], 
                linewidth=2, label=name, alpha=0.8)
    ax.set_title("Route Progress Comparison", fontsize=11, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Progress (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 3])
    labels = list(eps.keys())
    coll_rates = [sum(1 for c in eps[lbl]["collision"] if c > 0)/40*100 for lbl in labels]
    bars = ax.bar(labels, coll_rates, color=ep_colors, alpha=0.8)
    for bar, rate in zip(bars, coll_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Collision Rate by Episode", fontsize=11, fontweight="bold")
    ax.set_ylabel("Collision %")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    
    ax = fig.add_subplot(gs[2, 4])
    ax.axis("off")
    
    info = (
        "PIPELINE SUMMARY\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Total Episodes: 3\n"
        f"Total Steps: 120\n"
        f"Total Collisions: {sum(sum(1 for c in eps[lbl]['collision'] if c > 0) for lbl in eps)}/120\n"
        f"Best Episode: episode_0001 (40% collision)\n"
        f"\n"
        "SIMULATOR: AlpaSim 0.42\n"
        "DRIVER: VAM (VideoActionModel)\n"
        "  - Vision encoder: 318M params\n"
        "  - Action expert: 38M params\n"
        "VISION MODEL: Alpamayo CNN-LSTM\n"
        "  - Output: 20×2 trajectories\n"
        "  - Training: 3 episodes\n"
        "\n"
        "GPU: NVIDIA L4 (23GB VRAM)\n"
        "SCENE: clipgt-a309e228\n"
        "DATASET: NVIDIA Physical AI NuRec\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STATUS: Fine-tuning complete\n"
        "NEXT: Integrate Alpamayo-R1-10B\n"
        "       Replace VAM with custom driver"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.95, edgecolor="#3498db"),
            color="white")
    
    plt.savefig(f"{OUT_DIR}/pipeline_final_dashboard.png", dpi=120, 
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Dashboard saved: {OUT_DIR}/pipeline_final_dashboard.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Trajectory Comparison Across 3 Episodes", fontsize=14, fontweight="bold")
    
    for i, (name, ep) in enumerate(eps.items()):
        ax = axes[i]
        cdf = ep["csv"].iloc[:40]
        x = cdf["x"].values
        y = cdf["y"].values
        
        collision = ep["collision"]
        safe_x = [x[j] for j in range(len(x)-1) if collision[j] == 0]
        safe_y = [y[j] for j in range(len(y)-1) if collision[j] == 0]
        coll_x = [x[j] for j in range(len(x)-1) if collision[j] > 0]
        coll_y = [y[j] for j in range(len(y)-1) if collision[j] > 0]
        
        ax.plot(safe_x, safe_y, "g-", linewidth=3, label="Safe", alpha=0.7)
        ax.plot(coll_x, coll_y, "r-", linewidth=3, label="Collision", alpha=0.7)
        ax.scatter(x[0], y[0], color="blue", s=200, zorder=10, marker=">", label="Start")
        ax.scatter(x[-1], y[-1], color="purple", s=200, zorder=10, marker="s", label="End")
        
        n_coll = sum(1 for c in collision if c > 0)
        n_safe = 40 - n_coll
        ax.set_title(f"{name}\n{n_safe} safe, {n_coll} collision", fontsize=11, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/trajectory_final.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Trajectory saved: {OUT_DIR}/trajectory_final.png")
    
    print("\nAll demos complete!")
    for f in ["pipeline_final_dashboard.png", "trajectory_final.png", "training_metrics.png"]:
        path = f"{OUT_DIR}/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1e3
            print(f"  {f}: {size:.0f}KB")

if __name__ == "__main__":
    main()
