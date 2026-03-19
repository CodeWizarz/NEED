#!/usr/bin/env python3
import os, subprocess, pandas as pd

SSH_KEY = os.path.expanduser("~/.ssh/google_compute_engine")
HOST = "136.119.37.171"

def ssh(cmd):
    result = subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
         f"Balu@{HOST}", cmd],
        capture_output=True, text=True, timeout=60
    )
    return result.stdout + result.stderr

episodes = [
    ("3bf6c85c-2386-11f1-8c57-37b1dc1643c0", "episode_0000"),
    ("dafbf1aa-238d-11f1-adb9-8b3a2e5967da", "episode_0001"),
]

os.makedirs("/Users/Balu/Documents/NEED/episodes", exist_ok=True)

for uuid, ep_name in episodes:
    ep_dir = f"/Users/Balu/Documents/NEED/episodes/{ep_name}"
    os.makedirs(ep_dir, exist_ok=True)
    os.makedirs(ep_dir, exist_ok=True)
    
    base = f"/home/Balu/alpasim/tutorial/rollouts/clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb/{uuid}"
    ctrl = f"/home/Balu/alpasim/tutorial/controller/alpasim_controller_{uuid}.csv"
    
    subprocess.run(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
        f"Balu@{HOST}:{base}/metrics.parquet", f"{ep_dir}/metrics.parquet"], check=False)
    subprocess.run(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
        f"Balu@{HOST}:{base}/*.mp4", f"{ep_dir}/"], check=False)
    subprocess.run(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
        f"Balu@{HOST}:{ctrl}", f"{ep_dir}/controller.csv"], check=False)
    
    print(f"\n=== {ep_name} ===")
    
    if os.path.exists(f"{ep_dir}/metrics.parquet"):
        mdf = pd.read_parquet(f"{ep_dir}/metrics.parquet")
        for metric in ["collision_any", "offroad", "progress", "dist_to_gt_trajectory", "plan_deviation"]:
            rows = mdf[mdf["name"] == metric]
            vals = rows["values"].values
            nonzero = sum(1 for v in vals if v > 0)
            print(f"  {metric}: mean={vals.mean():.3f}, max={vals.max():.3f}, nonzero={nonzero}/{len(vals)}")
    
    if os.path.exists(f"{ep_dir}/controller.csv"):
        cdf = pd.read_csv(f"{ep_dir}/controller.csv")
        print(f"  Steps: {len(cdf)}")
        print(f"  Speed: {cdf['vx'].min():.1f} - {cdf['vx'].max():.1f} m/s")
        print(f"  Traveled: {cdf['x'].max() - cdf['x'].min():.1f}m")
    
    mp4_files = [f for f in os.listdir(ep_dir) if f.endswith(".mp4")]
    for mp4 in mp4_files:
        size = os.path.getsize(f"{ep_dir}/{mp4}") / 1e6
        print(f"  Video: {mp4} ({size:.1f} MB)")

print("\n=== All episodes downloaded ===")
