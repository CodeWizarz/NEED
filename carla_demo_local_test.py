"""
CARLA Remote Inference Client - Local Test Version
Uses mock CARLA to test the pipeline without full CARLA server.
"""

import numpy as np
import requests
import cv2
import time
import sys
sys.path.insert(0, '/Users/Balu/Documents/NEED')
from test_carla_camera import CARLASensorCamera, CarlaClient

VM_IP = "136.119.37.171"
VM_URL = f"http://{VM_IP}:8000"

print("=" * 60)
print("CARLA Remote Inference Client (Local Test)")
print("=" * 60)
print(f"VM URL: {VM_URL}")

client = CarlaClient("localhost", 2000)
print(f"Connected to CARLA (mock)")

camera = CARLASensorCamera(image_size_x=256, image_size_y=256)
frame_buffer = []
video_frames = []
action = "GO"

def process_frame(img):
    global frame_buffer, video_frames, action
    
    video_frames.append(img.copy())
    
    img_norm = img.astype(np.float32) / 255.0
    frame_buffer.append(img_norm)
    
    if len(frame_buffer) >= 8:
        send_frames(frame_buffer[-8:])

def send_frames(frames):
    global action
    
    payload = {
        "frames": np.array(frames).tolist()
    }
    
    try:
        print(f"Sending {len(frames)} frames to VM...", end=" ", flush=True)
        response = requests.post(
            f"{VM_URL}/infer",
            json=payload,
            timeout=120
        )
        result = response.json()
        action = result.get("action", "GO")
        print(f">>> Action: {action}")
    except Exception as e:
        print(f"Error: {e}")
        action = "GO"

camera.listen(process_frame)

print("\n" + "=" * 60)
print("Running... Generating test frames")
print("=" * 60)

for i in range(20):
    frame = camera.generate_frame()
    process_frame(frame)
    print(f"Frame {i+1}/20: action={action}")

print("\nSaving video...")
if video_frames:
    h, w, _ = video_frames[0].shape
    out = cv2.VideoWriter(
        "demo.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )
    
    for frame in video_frames:
        out.write(frame)
    
    out.release()
    print(f"Saved demo.mp4 ({len(video_frames)} frames)")

print("Done!")
