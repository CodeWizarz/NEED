"""
CARLA Demo - Simple recording with autopilot
"""

import carla
import numpy as np
import cv2
import time

print("=" * 60)
print("CARLA Demo with Recording")
print("=" * 60)

client = carla.Client("localhost", 2000)
client.set_timeout(60.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprints.filter("model3")[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
if not vehicle:
    for sp in spawn_points[1:10]:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle:
            break

print(f"Vehicle spawned: {vehicle.id}")

# Manual control - no autopilot
# Camera
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
print(f"Camera spawned: {camera.id}")

# Spectator camera
spectator = world.get_spectator()

# Recording
video = []
frame_count = 0
prev_steer = 0.0
map = world.get_map()

def process_frame(image):
    global video, frame_count, prev_steer
    
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((600, 800, 4))[:, :, :3]
    video.append(img.copy())
    frame_count += 1
    
    # Get waypoint for lane following
    waypoint = map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True
    )
    
    # Manual control
    control = carla.VehicleControl()
    control.throttle = 0.4
    control.brake = 0.0
    
    if waypoint:
        # Steer toward lane center
        lane_loc = waypoint.transform.location
        veh_loc = vehicle.get_location()
        dy = lane_loc.y - veh_loc.y
        
        steer_target = np.clip(dy * 0.5, -0.6, 0.6)
        steer = 0.85 * prev_steer + 0.15 * steer_target
        prev_steer = steer
        control.steer = float(steer)
    else:
        control.steer = 0.0
    
    vehicle.apply_control(control)
    
    # Update spectator
    transform = vehicle.get_transform()
    offset = carla.Location(x=-8, z=4)
    new_loc = transform.location + offset
    new_loc.z += 2
    spectator.set_transform(carla.Transform(new_loc, transform.rotation))
    
    if frame_count % 30 == 0:
        vel = vehicle.get_velocity()
        speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2)**0.5
        print(f"Frame {frame_count} | Speed: {speed:.1f} km/h | Steer: {control.steer:.3f}")

camera.listen(process_frame)

print("\nRecording driving...")
print("=" * 60)

try:
    while len(video) < 1200:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

print(f"\nRecorded {len(video)} frames")
print("Saving video...")

if video:
    h, w, _ = video[0].shape
    out = cv2.VideoWriter("/home/Balu/demo_realistic.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for frame in video:
        out.write(frame)
    out.release()
    print("Video saved!")

camera.stop()
vehicle.destroy()
camera.destroy()
print("Done!")
