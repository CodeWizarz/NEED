"""
CARLA Cinematic Demo - NVIDIA-style visuals
"""

import carla
import numpy as np
import cv2
import time
import os

print("=" * 60)
print("CARLA Cinematic Demo")
print("=" * 60)

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)

world = client.get_world()
settings = world.get_settings()
settings.no_rendering_mode = False
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

blueprints = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprints.filter("model3")[0]
spawn_points = world.get_map().get_spawn_points()
for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        break

print(f"Vehicle spawned: {vehicle.id}")

# High quality camera
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")
camera_bp.set_attribute("fov", "90")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
print(f"Camera spawned: {camera.id}")

# Cinematic follow camera (spectator)
spectator = world.get_spectator()

def update_spectator():
    transform = vehicle.get_transform()
    # Cinematic offset behind and above
    offset = carla.Location(x=-8, z=4)
    new_loc = transform.location + offset
    new_loc.z += 2  # Extra height
    spectator.set_transform(
        carla.Transform(new_loc, transform.rotation)
    )

# Buffers
video = []
prev_steer = 0.0
frame_count = 0

def process_frame(image):
    global video, prev_steer, frame_count
    
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((600, 800, 4))[:, :, :3]
    video.append(img.copy())
    
    frame_count += 1
    
    # Update spectator camera
    update_spectator()
    
    # Get nearby vehicles for obstacle detection
    actors = world.get_actors().filter("vehicle.*")
    
    action = "GO"
    control = vehicle.get_control()
    
    # Simple obstacle detection
    min_dist = float('inf')
    for actor in actors:
        if actor.id != vehicle.id:
            dist = actor.get_location().distance(vehicle.get_location())
            if dist < min_dist:
                min_dist = dist
    
    # Apply braking based on distance
    if min_dist < 8:
        control.brake = 1.0
        control.throttle = 0.0
        action = "STOP"
    elif min_dist < 15:
        control.brake = 0.6
        control.throttle = 0.1
        action = "SLOW"
    else:
        control.brake = 0.0
        control.throttle = 0.35
        action = "GO"
    
    # Smooth steering toward road center
    transform = vehicle.get_transform()
    road_pos = transform.location
    
    # Simple steering based on road position
    road_x = road_pos.x % 10
    steer_target = (road_x - 5) / 5
    steer_target = np.clip(steer_target * 0.3, -0.5, 0.5)
    
    # Smooth steering
    steer = 0.85 * prev_steer + 0.15 * steer_target
    prev_steer = steer
    
    control.steer = float(steer)
    vehicle.apply_control(control)
    
    if frame_count % 10 == 0:
        print(f"Frame {frame_count} | Action: {action} | Dist: {min_dist:.1f}m | Steer: {steer:.2f}")

camera.listen(process_frame)

print("\n" + "=" * 60)
print("Running cinematic demo...")
print("=" * 60)

try:
    while len(video) < 600:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nSaving video...")

# Save video
if video:
    h, w, _ = video[0].shape
    out = cv2.VideoWriter(
        "/home/Balu/demo_cinematic.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (w, h)
    )
    
    for frame in video:
        out.write(frame)
    
    out.release()
    print(f"Saved demo_cinematic.mp4 ({len(video)} frames)")

# Cleanup
camera.stop()
vehicle.destroy()
camera.destroy()

print("Done!")
