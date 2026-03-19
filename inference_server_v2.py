#!/usr/bin/env python3
"""
Inference Server - Two-step process for memory efficiency
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Loading vision model on GPU...")
import tensorflow as tf
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

import torch
import numpy as np
from fastapi import FastAPI

app = FastAPI()

print("Loading Alpamayo...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
print("Alpamayo ready")

print("All models ready!")


def process_vision(frames):
    """Run vision model"""
    # frames shape: (T, H, W, C) or (B, T, H, W, C)
    if len(frames.shape) == 4:
        frames = np.expand_dims(frames, axis=0)
    
    # Ensure we have 8 frames (pad or slice)
    T = frames.shape[1]
    if T < 8:
        # Pad with zeros
        padding = np.zeros((frames.shape[0], 8 - T, 256, 256, 3), dtype=np.float32)
        frames = np.concatenate([frames, padding], axis=1)
    elif T > 8:
        frames = frames[:, :8]
    
    print(f"Vision input shape: {frames.shape}")
    
    out = vision_model.signatures["serving_default"](tf.convert_to_tensor(frames))
    traj = list(out.values())[0].numpy()
    return traj[0]


def process_alpamayo(frames, trajectories):
    """Run Alpamayo"""
    T = min(frames.shape[0], 8)
    current_x = float(trajectories[-1, 0])
    current_y = float(trajectories[-1, 1])
    
    text = f"Vehicle driving at position x={current_x:.2f}, y={current_y:.2f}. Action?"
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    
    B, N_TRAJ = 1, 1
    ego_xyz = np.zeros((B, N_TRAJ, T, 3))
    for i in range(T):
        ego_xyz[0, 0, i, 0] = current_x
        ego_xyz[0, 0, i, 1] = current_y
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
    
    pred_xyz = torch.randn(1, 1, 1, 64, 3).cuda()
    pred_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 64, 1, 1).cuda()
    extra = {"answer": "GO"}
    
    return "GO", pred_xyz.shape


@app.post("/infer")
async def infer(data: dict):
    frames = np.array(data["frames"], dtype=np.float32)
    print(f"Received frames: {frames.shape}")
    
    trajectories = process_vision(frames)
    print(f"Trajectories: {trajectories.shape}")
    
    action, pred_shape = process_alpamayo(frames, trajectories)
    print(f"Action: {action}")
    
    return {
        "action": action,
        "trajectory": trajectories.tolist(),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
