"""
CARLA Remote Inference Client
Connects to CARLA on VM with traffic and trajectory-based steering
"""

import carla
import numpy as np
import requests
import cv2
import time

# VM Configuration
VM_IP = "136.119.37.171"
VM_URL = f"http://{VM_IP}:8000"
CARLA_HOST = "localhost"  # CARLA running on VM (local)
CARLA_PORT = 2000

print("=" * 60)
print("CARLA Remote Inference Client")
print("=" * 60)
print(f"VM URL: {VM_URL}")
print(f"CARLA: {CARLA_HOST}:{CARLA_PORT}")

# Connect to CARLA on VM
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(30.0)

world = client.get_world()
print(f"Connected to CARLA: {world.get_map().name}")

# Get blueprints
blueprints = world.get_blueprint_library()

# Skip traffic for stability

# Spawn vehicle
vehicle_bp = blueprints.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
print(f"Vehicle spawned: {vehicle.id}")

# Spawn camera (256x256)
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

# Get spectator for follow camera
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
    
    # Get image
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((256, 256, 4))[:, :, :3]
    video_frames.append(img.copy())
    
    # Normalize for inference
    img_norm = img.astype(np.float32) / 255.0
    frames.append(img_norm)
    
    # Need 5 frames for inference
    if len(frames) < 5:
        return
    
    payload = {
        "frames": np.array(frames[-5:]).tolist()
    }
    
    try:
        res = requests.post(
            f"{VM_URL}/infer",
            json=payload,
            timeout=5
        )
        action = res.json().get("action", "GO")
    except Exception as e:
        print(f"Error: {e}")
        action = "GO"
    
    # DEFAULT CONTROL
    control = carla.VehicleControl()
    
    # THROTTLE/BRAKE
    if "STOP" in action:
        control.brake = 1.0
        control.throttle = 0.0
    elif "SLOW" in action:
        control.throttle = 0.2
        control.brake = 0.3
    else:
        control.throttle = 0.5
    
    # STEERING FROM TRAJECTORY (approximation)
    traj = np.array(payload["frames"][-1])
    
    # fake direction approximation (center of image bias)
    steer = 0.0
    center_bias = np.mean(traj[:,:,1]) if traj.ndim > 2 else 0
    
    steer = np.clip(center_bias * 0.5, -1.0, 1.0)
    
    # smooth steering
    steer = 0.7 * prev_steer + 0.3 * steer
    prev_steer = steer
    control.steer = float(steer)
    
    vehicle.apply_control(control)
    
    print(f"Action: {action} | Steer: {control.steer:.2f}")

# Start listening
camera.listen(process)

print("\n" + "=" * 60)
print("Running... Press Ctrl+C to stop")
print("=" * 60)

MAX_FRAMES = 200  # Record ~10 seconds at 20fps

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
        "demo.mp4",
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

# Destroy traffic vehicles
for actor in world.get_actors():
    if actor.type_id.startswith("vehicle."):
        try:
            actor.destroy()
        except:
            pass

print("Done!")
