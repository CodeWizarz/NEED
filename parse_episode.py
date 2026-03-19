#!/usr/bin/env python3
import os, subprocess, pandas as pd, json

episode = 0
ep_dir = f"/home/Balu/logs/episodes/episode_{episode:04d}"
os.makedirs(ep_dir, exist_ok=True)

subprocess.run(["cp", "/home/Balu/alpasim/tutorial/controller/alpasim_controller_3bf6c85c-2386-11f1-8c57-37b1dc1643c0.csv", f"{ep_dir}/controller.csv"], check=True)
subprocess.run(["cp", "/home/Balu/alpasim/tutorial/rollouts/clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb/3bf6c85c-2386-11f1-8c57-37b1dc1643c0/metrics.parquet", f"{ep_dir}/metrics.parquet"], check=True)

df = pd.read_csv(f"{ep_dir}/controller.csv")
print(f"CSV rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Speed (vx) range: {df['vx'].min():.2f} to {df['vx'].max():.2f} m/s")
print(f"X range: {df['x'].min():.2f} to {df['x'].max():.2f}")
print(f"Y range: {df['y'].min():.2f} to {df['y'].max():.2f}")
print(f"Steering range: {df['u_steering_angle'].min():.3f} to {df['u_steering_angle'].max():.3f}")

if 'dist_traveled_m' in df.columns:
    print(f"Traveled: {df['dist_traveled_m'].max():.1f} m")

mdf = pd.read_parquet(f"{ep_dir}/metrics.parquet")
collision_any = mdf[mdf["name"] == "collision_any"]
collisions = [i for i, v in enumerate(collision_any["values"].values) if v > 0]
print(f"\nCollision steps: {collisions}")
print(f"Collision rate: {len(collisions)}/{len(collision_any)} = {len(collisions)/len(collision_any)*100:.1f}%")

print(f"\nSample trajectory (first step):")
print(f"  Ego: x={df['x'].iloc[0]:.3f}, y={df['y'].iloc[0]:.3f}, speed={df['vx'].iloc[0]:.2f}")
print(f"  Traj ref 0: x={df['x_ref_0'].iloc[0]:.3f}, y={df['y_ref_0'].iloc[0]:.3f}")
print(f"  Traj ref 1: x={df['x_ref_0'].iloc[1]:.3f}, y={df['y_ref_0'].iloc[1]:.3f}")

print(f"\nEpisode 0 data ready at {ep_dir}")
