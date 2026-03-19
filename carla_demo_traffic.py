"""
CARLA Demo with Traffic - Realistic driving scenarios
"""

import carla
import numpy as np
import requests
import cv2
import time

VM_IP = "136.119.37.171"

print("=" * 60)
print("CARLA Demo with Traffic")
print("=" * 60)

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprints.filter("model3")[0]
spawn_points = world.get_map().get_spawn_points()
for spawn_point in spawn_points[:10]:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        break
if not vehicle:
    raise Exception("Could not spawn vehicle")
print(f"Vehicle spawned: {vehicle.id}")

# Camera
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "256")
camera_bp.set_attribute("image_size_y", "256")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
print(f"Camera spawned: {camera.id}")

# Add traffic (skip traffic manager to avoid bind error)
print("Spawning traffic vehicles...")
vehicles = []
for _ in range(10):
    bp = blueprints.filter("vehicle.*")[np.random.randint(0, len(blueprints.filter("vehicle.*")))]
    spawn = world.get_random_location_from_navigation()
    if spawn:
        try:
            vehicle_npc = world.try_spawn_actor(bp, carla.Transform(spawn))
            if vehicle_npc:
                vehicles.append(vehicle_npc)
        except:
            pass

print(f"Spawned {len(vehicles)} traffic vehicles")

# Force obstacle directly ahead
print("Creating obstacle...")
obstacle = None
spawn_pt = vehicle.get_transform()
forward = spawn_pt.get_forward_vector()
obstacle_loc = spawn_pt.location + forward * 15

obstacle_bp = blueprints.filter("vehicle.*")[0]
try:
    obstacle = world.try_spawn_actor(
        obstacle_bp,
        carla.Transform(obstacle_loc, spawn_pt.rotation)
    )
    if obstacle:
        print("Obstacle spawned ahead!")
except Exception as e:
    print(f"Could not spawn obstacle: {e}")

# Buffers
frames = []
video = []
action = "GO"
prev_steer = 0.0

def process(image):
    global frames, video, action, prev_steer

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((256, 256, 4))[:, :, :3]
    video.append(img)

    img_norm = img.astype(np.float32) / 255.0
    frames.append(img_norm)

    if len(frames) < 5:
        return

    payload = {
        "frames": np.array(frames[-5:]).tolist()
    }

    try:
        res = requests.post(
            f"http://{VM_IP}:8000/infer",
            json=payload,
            timeout=5
        )
        result = res.json()
        action = result.get("action", "GO")
    except Exception as e:
        print(f"Inference error: {e}")
        action = "GO"

    control = carla.VehicleControl()

    if action == "STOP":
        control.brake = 1.0
        control.throttle = 0.0
    elif action == "SLOW":
        control.throttle = 0.3
        control.brake = 0.5
    else:
        control.throttle = 0.4
        control.brake = 0.0

    # Steering from image center
    center_bias = np.mean(img[150:, 100:156])
    steer = (center_bias - 60) / 60
    steer = np.clip(steer * 0.5, -1.0, 1.0)
    steer = 0.7 * prev_steer + 0.3 * steer
    prev_steer = steer
    control.steer = float(steer)

    vehicle.apply_control(control)

    print(f"Action: {action} | Throttle: {control.throttle:.2f} | Brake: {control.brake:.2f} | Steer: {control.steer:.2f}")

camera.listen(process)

print("\n" + "=" * 60)
print("Running... Press Ctrl+C to stop")
print("=" * 60)

try:
    while len(video) < 300:
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nSaving video...")

# Save video
if video:
    h, w, _ = video[0].shape
    out = cv2.VideoWriter(
        "/home/Balu/demo_traffic.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )

    for frame in video:
        out.write(frame)

    out.release()
    print(f"Saved demo_traffic.mp4 ({len(video)} frames)")

# Cleanup
camera.stop()
vehicle.destroy()
camera.destroy()
for v in vehicles:
    try:
        v.destroy()
    except:
        pass
if obstacle:
    try:
        obstacle.destroy()
    except:
        pass

print("Done!")
