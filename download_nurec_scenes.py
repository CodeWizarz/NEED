#!/usr/bin/env python3
import os, json
os.environ["HF_TOKEN"] = "<REDACTED>"

from huggingface_hub import HfApi, hf_hub_download

DATASET = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
LOCAL_DIR = "/home/Balu/alpasim/data/nre-artifacts/scenesets/nurec"

api = HfApi(token=os.environ["HF_TOKEN"])
info = api.dataset_info(DATASET, files_metadata=True)

usdz_scenes = []
for f in info.siblings:
    path = f.rfilename
    if path.startswith("sample_set/") and path.endswith(".usdz"):
        parts = path.split("/")
        scene_id = parts[2]
        usdz_scenes.append({"scene_id": scene_id, "path": path, "size": f.size})

usdz_scenes.sort(key=lambda x: x["size"])
print(f"Found {len(usdz_scenes)} scenes. Sizes: {[f'{s['size']/1e9:.2f}GB' for s in usdz_scenes[:5]]}")

scenes_to_download = usdz_scenes[:3]
for scene in scenes_to_download:
    sid = scene["scene_id"]
    print(f"\n--- Scene: {sid} ({scene['size']/1e9:.2f}GB) ---")
    
    labels_path = f"sample_set/26.02_release/{sid}/labels.json"
    try:
        labels_file = hf_hub_download(DATASET, labels_path, repo_type="dataset", local_dir=LOCAL_DIR)
        with open(labels_file) as f:
            labels = json.load(f)
        print(f"  Labels: {labels}")
    except Exception as e:
        print(f"  Labels error: {e}")
    
    usdz_dest = f"{LOCAL_DIR}/sample_set/26.02_release/{sid}/{sid}.usdz"
    if os.path.exists(usdz_dest):
        print(f"  Already downloaded: {usdz_dest} ({os.path.getsize(usdz_dest)/1e9:.2f}GB)")
        continue
    
    print(f"  Downloading USDZ...")
    try:
        usdz_file = hf_hub_download(DATASET, scene["path"], repo_type="dataset", local_dir=LOCAL_DIR)
        print(f"  Downloaded: {usdz_file} ({os.path.getsize(usdz_file)/1e9:.2f}GB)")
    except Exception as e:
        print(f"  USDZ error: {e}")

print("\nDone!")
