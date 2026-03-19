import carla
import numpy as np
import time

print("="*60)
print("CARLA Camera Test")
print("="*60)

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

blueprints = world.get_blueprint_library()
vehicle_bp = blueprints.filter("vehicle.*")[0]
spawn = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn)
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

frames = []
def callback(image):
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((256,256,4))[:,:,:3]
    frames.append(img)
    print(f"Frame OK: {img.shape}")

camera.listen(callback)
time.sleep(10)
camera.stop()

vehicle.destroy()
camera.destroy()

print(f"\nTotal frames: {len(frames)}")
if len(frames) > 0:
    print("SUCCESS: Camera working")
else:
    print("FAIL: No frames")
