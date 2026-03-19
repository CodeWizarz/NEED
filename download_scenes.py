#!/usr/bin/env python3
import os, json, sys
sys.path.insert(0, "/home/Balu/logs")
from waymo_streamer import get_sample_list, stream_scene
os.environ["HF_TOKEN"] = "<REDACTED>"

samples = get_sample_list(20)
print(f"Available: {len(samples)} scenes")

diverse = [samples[0], samples[9], samples[19]]
for s in diverse:
    labels = stream_scene(s, labels_only=True)
    sid = s["scene_id"]
    size = s["size"] / 1e9
    bh = labels["behavior"]
    ly = labels["layout"]
    wr = labels["weather"]
    vrus = labels["vrus"]
    print(f"  {sid} ({size:.2f} GB): {bh} | {ly} | {wr} | vrus={vrus}")
    print(f"    -> Downloading...")
    result = stream_scene(s)
    print(f"    -> Downloaded to {result['usdz_path']}")
