#!/usr/bin/env python3
import os
import json
import time
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN", "<REDACTED>")
DATASET = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
LOCAL_DIR = "/home/Balu/alpasim/data/nre-artifacts/scenesets/nurec"
os.environ["HF_TOKEN"] = HF_TOKEN

def get_sample_list(n=10):
    api = HfApi(token=HF_TOKEN)
    info = api.dataset_info(DATASET, files_metadata=True)
    samples = []
    for f in info.siblings:
        path = f.rfilename
        if path.startswith("sample_set/") and path.endswith(".usdz"):
            parts = path.split("/")
            scene_id = parts[-1].replace(".usdz", "")
            samples.append({
                "scene_id": scene_id,
                "path": path,
                "size": f.size,
            })
    return sorted(samples, key=lambda x: x["size"])[:n]

def stream_scene(scene_info, labels_only=False):
    scene_id = scene_info["scene_id"]
    release = "26.02_release"
    labels_path = f"sample_set/{release}/{scene_id}/labels.json"
    usdz_path = f"sample_set/{release}/{scene_id}/{scene_id}.usdz"
    
    labels_file = hf_hub_download(
        DATASET,
        labels_path,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
    with open(labels_file) as f:
        labels = json.load(f)
    
    if labels_only:
        return labels
    
    usdz_file = hf_hub_download(
        DATASET,
        usdz_path,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
    
    return {
        "labels": labels,
        "usdz_path": usdz_file,
        "size_gb": scene_info["size"] / 1e9,
    }

if __name__ == "__main__":
    samples = get_sample_list(5)
    print(f"Found {len(samples)} sample scenes")
    for s in samples:
        print(f"  {s['scene_id']} ({s['size']/1e9:.2f} GB)")
    
    labels = stream_scene(samples[0], labels_only=True)
    print(f"\nSample labels: {json.dumps(labels, indent=2)}")
