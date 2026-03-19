"""
CARLA Demo with Mock Camera - No CARLA server needed
Uses mock camera and local inference (no network)
"""

import numpy as np
import cv2
import time
import os

print("=" * 60)
print("CARLA Demo (Mock Camera)")
print("=" * 60)

# Create mock camera frames
def generate_road_frame(t):
    """Generate a simulated road view"""
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Sky
    frame[:128, :] = [135, 206, 250]
    
    # Road
    frame[128:, :] = [60, 60, 60]
    
    # Road markings
    for i in range(128, 256, 20):
        frame[i:i+10, 120:136] = [255, 255, 255]
    
    # Add some movement based on time
    offset = int(10 * np.sin(t * 0.5))
    if offset > 0:
        frame[140:180, 100+offset:156+offset] = [100, 100, 100]  # Car ahead
    
    return frame

# Buffers
frames = []
video_frames = []
action = "GO"
prev_steer = 0.0

MAX_FRAMES = 200

print(f"Generating {MAX_FRAMES} frames...")

for i in range(MAX_FRAMES):
    t = i * 0.1
    
    # Generate mock camera frame
    img = generate_road_frame(t)
    video_frames.append(img)
    
    img_norm = img.astype(np.float32) / 255.0
    frames.append(img_norm)
    
    # Simple steering logic based on road position
    center_bias = np.mean(img[150:200, 100:156])
    steer = (center_bias - 60) / 60
    steer = np.clip(steer * 0.5, -1.0, 1.0)
    steer = 0.7 * prev_steer + 0.3 * steer
    prev_steer = steer
    
    # Simple action (always GO for now)
    action = "GO"
    
    if (i + 1) % 20 == 0:
        print(f"Frame {i+1}/{MAX_FRAMES}: Action={action}, Steer={steer:.2f}")

# Save video
print("\nSaving video...")
h, w, _ = video_frames[0].shape
out = cv2.VideoWriter(
    "/Users/Balu/Documents/NEED/demo.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (w, h)
)

for frame in video_frames:
    out.write(frame)

out.release()
print(f"Saved demo.mp4 ({len(video_frames)} frames)")
print("Done!")
