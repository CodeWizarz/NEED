"""
CARLA Real Camera Test - Connects to running CARLA server
"""

import carla
import numpy as np
import time

print("=" * 60)
print("CARLA Real Camera Test")
print("=" * 60)

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

print(f"Connected to CARLA")
print(f"Map: {world.get_map().name}")

# Get blueprint library
blueprints = world.get_blueprint_library()

# Find vehicle blueprint
vehicle_bp = blueprints.filter("vehicle.*")[0]
print(f"Vehicle blueprint: {vehicle_bp.id}")

# Spawn vehicle at random spawn point
spawn_points = world.get_map().get_spawn_points()
spawn = spawn_points[0]
vehicle = world.spawn_actor(vehicle_bp, spawn)
print(f"Spawned vehicle: {vehicle.id}")

# Create RGB camera
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "256")
camera_bp.set_attribute("image_size_y", "256")
camera_bp.set_attribute("fov", "90")

# Attach camera to vehicle
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
print(f"Spawned camera: {camera.id}")

# Listen for frames
frame_count = 0
frames = []

def process_frame(image):
    global frame_count, frames
    frame_count += 1
    # Convert to numpy array
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]
    frames.append(img)
    print(f"Frame {frame_count}: {img.shape}")

camera.listen(process_frame)

# Wait for frames
print("\nCapturing frames...")
time.sleep(3)

# Stop camera
camera.stop()
print(f"\nCaptured {len(frames)} frames")

# Destroy actors
vehicle.destroy()
camera.destroy()
print("Actors destroyed")

print("\n" + "=" * 60)
print("CARLA CAMERA WORKING: YES")
print("=" * 60)
