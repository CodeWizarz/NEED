"""
CARLA Remote Inference Client - Uses real CARLA on VM
"""

import numpy as np
import requests
import cv2
import time
import sys
import carla

VM_IP = "136.119.37.171"
VM_URL = f"http://{VM_IP}:8000"

print("=" * 60)
print("CARLA Remote Inference Client (Real CARLA on VM)")
print("=" * 60)
print(f"VM URL: {VM_URL}")
print(f"CARLA: {VM_IP}:2000")

client = carla.Client(VM_IP, 2000)
client.set_timeout(30.0)
world = client.get_world()
print(f"Connected to CARLA: {world.get_map().name}")

blueprints = world.get_blueprint_library()
vehicle_bp = blueprints.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
print(f"Vehicle spawned: {vehicle.id}")

camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "256")
camera_bp.set_attribute("image_size_y", "256")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
print(f"Camera spawned: {camera.id}")

frame_buffer = []
video_frames = []
action = "GO"
frame_count = 0

class FrameBuffer:
    def __init__(self):
        self.data = None
        
frame_obj = FrameBuffer()

def process_frame(image):
    global frame_buffer, video_frames, action, frame_count, frame_obj
    
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((256, 256, 4))[:, :, :3]
    video_frames.append(img.copy())
    
    img_norm = img.astype(np.float32) / 255.0
    frame_buffer.append(img_norm)
    frame_count += 1
    
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
print("Running... Drive around!")
print("=" * 60)

try:
    while len(video_frames) < 100:
        control = carla.VehicleControl()
        
        if "STOP" in action:
            control.throttle = 0.0
            control.brake = 1.0
        elif "SLOW" in action:
            control.throttle = 0.2
            control.brake = 0.3
        else:
            control.throttle = 0.5
            control.brake = 0.0
            control.steer = 0.0
        
        vehicle.apply_control(control)
        
        if len(video_frames) % 10 == 0:
            print(f"Frames: {len(video_frames)}, Action: {action}")
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
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

camera.stop()
vehicle.destroy()
camera.destroy()

print("Done!")
