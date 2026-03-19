"""
CARLA Final Demo - Clean recording at max quality
Focus: Recording only, no ML processing issues
"""

import carla
import numpy as np
import cv2
import time

print("=" * 60)
print("CARLA Final Demo - Clean Recording")
print("=" * 60)

client = carla.Client("localhost", 2000)
client.set_timeout(60.0)

world = client.get_world()
map = world.get_map()

blueprints = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprints.filter("model3")[0]
spawn_points = map.get_spawn_points()
for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        break

print(f"Vehicle spawned: {vehicle.id}")

# Camera - MAX QUALITY
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("fov", "90")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
print(f"Camera spawned: {camera.id}")

# Spectator camera
spectator = world.get_spectator()

# State
video = []
frame_count = 0
prev_error = 0.0
integral = 0.0

def get_target_waypoint():
    current_wp = map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    next_wps = current_wp.next(5.0)
    if len(next_wps) > 0:
        return next_wps[0]
    return current_wp

def process_frame(image):
    global video, frame_count, prev_error, integral
    
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((480, 640, 4))[:, :, :3]
    video.append(img.copy())
    frame_count += 1
    
    # Get waypoint
    waypoint = map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    target_wp = get_target_waypoint()
    
    # Compute heading error
    vehicle_transform = vehicle.get_transform()
    vehicle_forward = vehicle_transform.get_forward_vector()
    
    target_vector = target_wp.transform.location - vehicle_transform.location
    
    dot = vehicle_forward.x * target_vector.x + vehicle_forward.y * target_vector.y
    cross = vehicle_forward.x * target_vector.y - vehicle_forward.y * target_vector.x
    angle = np.arctan2(cross, dot)
    
    # PID control
    Kp, Kd, Ki = 1.2, 0.3, 0.05
    derivative = angle - prev_error
    integral += angle
    integral = np.clip(integral, -1.0, 1.0)
    
    steer = Kp * angle + Kd * derivative + Ki * integral
    steer = np.clip(steer, -1.0, 1.0)
    prev_error = angle
    
    # Speed control
    curvature = abs(angle)
    throttle = 0.5 if curvature < 0.3 else 0.35
    
    # Apply control
    control = carla.VehicleControl()
    control.steer = float(steer)
    control.throttle = throttle
    vehicle.apply_control(control)
    
    # Update spectator
    offset = carla.Location(x=-8, z=4)
    new_loc = vehicle_transform.location + offset
    spectator.set_transform(carla.Transform(new_loc, vehicle_transform.rotation))
    
    if frame_count % 60 == 0:
        vel = vehicle.get_velocity()
        speed = 3.6 * (vel.x**2 + vel.y**2)**0.5
        print(f"Frame {frame_count} | Speed: {speed:.1f} km/h | Steer: {steer:.3f}")

camera.listen(process_frame)

print("\n" + "=" * 60)
print("Recording...")
print("=" * 60)

try:
    while len(video) < 1200:
        time.sleep(0.05)
except KeyboardInterrupt:
    pass

print(f"\nRecorded {len(video)} frames")
print("Saving video...")

if video:
    h, w, _ = video[0].shape
    out = cv2.VideoWriter(
        "/home/Balu/demo_final.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h)
    )
    
    for frame in video:
        out.write(frame)
    
    out.release()
    print(f"Saved demo_final.mp4 ({len(video)} frames)")

camera.stop()
vehicle.destroy()
camera.destroy()
print("Done!")
