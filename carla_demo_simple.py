#!/usr/bin/env python3
"""
Simplified CARLA Inference Demo
Uses mock trajectory since vision model was cleared
"""

import carla
import numpy as np
import requests
import cv2
import time

# VM Configuration
VM_IP = "136.119.37.171"
VM_URL = f"http://{VM_IP}:8000"

print("=" * 60)
print("CARLA Inference Demo (Simplified)")
print("=" * 60)

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.get_world()
print(f"Connected to CARLA: {world.get_map().name}")

blueprints = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprints.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
print(f"Vehicle spawned: {vehicle.id}")

# Spawn camera
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "256")
camera_bp.set_attribute("image_size_y", "256")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
print(f"Camera spawned: {camera.id}")

# Buffers
frames = []
video_frames = []
action = "GO"
prev_steer = 0.0

# Get spectator
spectator = world.get_spectator()

def follow_vehicle():
    transform = vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(
            transform.location + carla.Location(z=5, x=-8),
            transform.rotation
        )
    )

def process(image):
    global frames, video_frames, action, prev_steer
    
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((256, 256, 4))[:, :, :3]
    video_frames.append(img.copy())
    
    img_norm = img.astype(np.float32) / 255.0
    frames.append(img_norm)
    
    if len(frames) < 5:
        return
    
    # Create mock trajectory (straight line)
    mock_traj = np.random.rand(20, 2) * 0.1
    
    payload = {
        "frames": np.array(frames[-5:]).tolist(),
        "trajectory": mock_traj.tolist()
    }
    
    # Simple action logic based on steering
    # Use image center for steering
    center = np.mean(img[100:,:], axis=0)
    center_x = np.mean(center)
    
    steer = (center_x - 128) / 128
    steer = np.clip(steer * 0.5, -1.0, 1.0)
    steer = 0.7 * prev_steer + 0.3 * steer
    prev_steer = steer
    
    # Simple action: GO for now
    action = "GO"
    
    # Apply control
    control = carla.VehicleControl()
    control.throttle = 0.4
    control.steer = float(steer)
    vehicle.apply_control(control)
    
    print(f"Action: {action} | Steer: {control.steer:.2f}")

camera.listen(process)

print("\n" + "=" * 60)
print("Running...")
print("=" * 60)

MAX_FRAMES = 100
start_time = time.time()

try:
    while len(video_frames) < MAX_FRAMES:
        follow_vehicle()
        time.sleep(0.05)
        
except KeyboardInterrupt:
    print("\nSaving video...")

# Save video
if video_frames:
    h, w, _ = video_frames[0].shape
    out = cv2.VideoWriter(
        "/home/Balu/demo.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )
    
    for frame in video_frames:
        out.write(frame)
    
    out.release()
    print(f"Saved demo.mp4 ({len(video_frames)} frames)")

# Cleanup
camera.stop()
vehicle.destroy()
camera.destroy()

print("Done!")
