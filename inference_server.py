"""
Inference Server - Receives frames from CARLA client and runs Alpamayo
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Run vision on CPU

from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import torch
import sys
import gc

app = FastAPI()

# Load vision model on CPU
print("Loading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded on CPU")

# Load Alpamayo on GPU
print("Loading Alpamayo...")
sys.path.insert(0, "/tmp/alpamayo_repo/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

alpamayo = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager"
).cuda().eval()

tokenizer = alpamayo.tokenizer
print("Alpamayo loaded on GPU")

print("All models ready!")


@app.post("/infer")
async def infer(data: dict):
    """Process frames and return action"""
    
    frames = np.array(data["frames"], dtype=np.float32)
    print(f"Received frames: {frames.shape}")
    
    # Run vision model (on CPU)
    trajectories = []
    for t in range(frames.shape[0]):
        out = vision_model.signatures["serving_default"](
            tf.convert_to_tensor(frames[t:t+1])
        )
        traj = list(out.values())[0].numpy()
        trajectories.append(traj[0])
    
    trajectories = np.array(trajectories)
    print(f"Trajectories: {trajectories.shape}")
    
    # Prepare Alpamayo input
    T = frames.shape[0]
    current_x = float(trajectories[-1, -1, 0])
    current_y = float(trajectories[-1, -1, 1])
    
    text = f"Vehicle driving. Position x={current_x:.2f}, y={current_y:.2f}. What action?"
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    
    B, N_TRAJ = 1, 1
    ego_xyz = np.zeros((B, N_TRAJ, T, 3))
    for i in range(T):
        ego_xyz[0, 0, i, 0] = float(trajectories[i, -1, 0])
        ego_xyz[0, 0, i, 1] = float(trajectories[i, -1, 1])
    ego_xyz = torch.tensor(ego_xyz, dtype=torch.float32).cuda()
    ego_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N_TRAJ, T, 1, 1).cuda()
    
    model_data = {
        "tokenized_data": {
            "input_ids": tokens["input_ids"].cuda(),
            "attention_mask": tokens["attention_mask"].cuda(),
        },
        "ego_history_xyz": ego_xyz,
        "ego_history_rot": ego_rot,
    }
    
    # Run Alpamayo
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, pred_rot, extra = alpamayo.sample_trajectories_from_data_with_vlm_rollout(
            data=model_data,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=32,
            return_extra=True,
        )
    
    # Determine action
    action = "GO"
    if 'answer' in extra:
        answer = str(extra['answer'])
        if 'STOP' in answer.upper():
            action = "STOP"
        elif 'SLOW' in answer.upper():
            action = "SLOW"
    
    return {
        "action": action,
        "trajectory": trajectories.tolist(),
        "prediction": pred_xyz.shape
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
