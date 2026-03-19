"""
CARLA Ultra Demo - Realistic driving with PID control
- Lookahead waypoints
- PID steering
- Stuck detection
- Max quality
"""

import carla
import numpy as np
import cv2
import time

print("=" * 60)
print("CARLA Ultra Demo - Realistic Driving")
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
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")
camera_bp.set_attribute("fov", "100")

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

# State for PID control
prev_error = 0.0
integral = 0.0

def get_target_waypoint():
    """Get waypoint ahead for lookahead"""
    current_wp = map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    # Look ahead 5 meters
    next_wps = current_wp.next(5.0)
    if len(next_wps) > 0:
        return next_wps[0]
    return current_wp

def process_frame(image):
    global video, frame_count, prev_error, integral
    
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((600, 800, 4))[:, :, :3]
    video.append(img.copy())
    frame_count += 1
    
    # Get waypoint
    waypoint = map.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    # Check if off-road
    if waypoint and waypoint.lane_type != carla.LaneType.Driving:
        print(f"Off-road! Lane: {waypoint.lane_type}")
        waypoint = get_target_waypoint()
    
    # Get target waypoint for lookahead
    target_wp = get_target_waypoint()
    
    # Compute heading error
    vehicle_transform = vehicle.get_transform()
    vehicle_forward = vehicle_transform.get_forward_vector()
    
    target_vector = target_wp.transform.location - vehicle_transform.location
    
    # Normalize
    dot = vehicle_forward.x * target_vector.x + vehicle_forward.y * target_vector.y
    cross = vehicle_forward.x * target_vector.y - vehicle_forward.y * target_vector.x
    angle = np.arctan2(cross, dot)
    
    # PID control
    Kp = 1.5
    Kd = 0.3
    Ki = 0.05
    
    derivative = angle - prev_error
    integral += angle
    integral = np.clip(integral, -1.0, 1.0)  # Anti-windup
    
    steer = Kp * angle + Kd * derivative + Ki * integral
    steer = np.clip(steer, -1.0, 1.0)
    prev_error = angle
    
    # Speed control based on curvature
    curvature = abs(angle)
    if curvature > 0.4:
        throttle = 0.2
    elif curvature > 0.2:
        throttle = 0.35
    else:
        throttle = 0.55
    
    # Stuck detection
    velocity = vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2)
    if speed < 0.5:
        print("Stuck! Nudging...")
        throttle = 0.7
        steer = 0.3
    
    # Apply control
    control = carla.VehicleControl()
    control.steer = float(steer)
    control.throttle = throttle
    control.brake = 0.0
    vehicle.apply_control(control)
    
    # Update spectator camera
    offset = carla.Location(x=-10, z=5)
    new_loc = vehicle_transform.location + offset
    new_loc.z += 3
    spectator.set_transform(carla.Transform(new_loc, vehicle_transform.rotation))
    
    if frame_count % 30 == 0:
        speed_kmh = 3.6 * speed
        print(f"Frame {frame_count} | Speed: {speed_kmh:.1f} km/h | Steer: {steer:.3f} | Throttle: {throttle:.2f} | Angle: {np.degrees(angle):.1f}deg")

camera.listen(process_frame)

print("\n" + "=" * 60)
print("Recording ultra demo...")
print("=" * 60)

try:
    while len(video) < 1800:  # 60 seconds at 30fps
        time.sleep(0.05)
except KeyboardInterrupt:
    pass

print(f"\nRecorded {len(video)} frames")
print("Saving video...")

if video:
    h, w, _ = video[0].shape
    out = cv2.VideoWriter(
        "/home/Balu/demo_ultra.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (w, h)
    )
    
    for frame in video:
        out.write(frame)
    
    out.release()
    print(f"Saved demo_ultra.mp4 ({len(video)} frames)")

camera.stop()
vehicle.destroy()
camera.destroy()
print("Done!")
