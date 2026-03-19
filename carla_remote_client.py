"""
CARLA Client - Sends frames to VM for inference
Run this on the machine where CARLA is running (localhost:2000)
"""

import carla
import numpy as np
import requests
import time

# VM Configuration
VM_IP = "VM_EXTERNAL_IP"  # Replace with your VM's external IP
VM_PORT = 8000
VM_URL = f"http://{VM_IP}:{VM_PORT}"

print("="*60)
print("CARLA Client - Remote Inference")
print("="*60)
print(f"VM URL: {VM_URL}")

# Connect to local CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
print(f"Connected to CARLA: {world.get_map().name}")

# Get blueprint library
blueprints = world.get_blueprint_library()

# Create vehicle
vehicle_bp = blueprints.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
print(f"Vehicle spawned: {vehicle.id}")

# Create RGB camera
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "256")
camera_bp.set_attribute("image_size_y", "256")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
print(f"Camera spawned: {camera.id}")

# Frame buffer
frame_buffer = []
T = 5

def callback(image):
    global frame_buffer
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((256, 256, 4))[:, :, :3]
    img = img.astype(np.float32) / 255.0  # Normalize
    frame_buffer.append(img)
    
    # Send to VM when we have enough frames
    if len(frame_buffer) >= T:
        send_to_vm(frame_buffer[-T:])
        frame_buffer = frame_buffer[-T:]  # Keep last T frames

def send_to_vm(frames):
    """Send frames to VM for inference"""
    payload = {
        "frames": np.array(frames).tolist()
    }
    
    try:
        response = requests.post(f"{VM_URL}/infer", json=payload, timeout=30)
        result = response.json()
        print(f"\n>>> Action: {result.get('action', 'UNKNOWN')}")
        print(f">>> Trajectory: {result.get('prediction', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")

camera.listen(callback)

print("\nStreaming frames to VM...")
print("Press Ctrl+C to stop")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")

camera.stop()
vehicle.destroy()
camera.destroy()
print("Done!")
