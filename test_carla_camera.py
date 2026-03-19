"""
CARLA Camera Interface - Simulates CARLA sensor.camera.rgb
Use this to test the pipeline without running full CARLA server.
"""

import numpy as np
import time


class CARLASensorCamera:
    """
    Simulates CARLA sensor.camera.rgb for testing purposes.
    In production, replace with actual CARLA sensor.
    """
    
    def __init__(self, image_size_x=256, image_size_y=256):
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.callback = None
        
    def listen(self, callback):
        """Register callback for camera frames."""
        self.callback = callback
        
    def generate_frame(self):
        """Generate a simulated camera frame."""
        # Simulate road scene with gradients
        frame = np.zeros((self.image_size_y, self.image_size_x, 3), dtype=np.uint8)
        
        # Sky (top half)
        frame[:self.image_size_y//2, :, 0] = 135  # Blue sky
        frame[:self.image_size_y//2, :, 1] = 206
        frame[:self.image_size_y//2, :, 2] = 250
        
        # Road (bottom half)
        frame[self.image_size_y//2:, :, 0] = 50  # Dark gray road
        frame[self.image_size_y//2:, :, 1] = 50
        frame[self.image_size_y//2:, :, 2] = 50
        
        # Road markings (center line)
        frame[self.image_size_y//2:, self.image_size_x//2-2:self.image_size_x//2+2, :] = 255
        
        return frame


class CarlaClient:
    """Mock CARLA client for testing."""
    
    def __init__(self, host="localhost", port=2000):
        self.host = host
        self.port = port
        self.timeout = 10.0
        
    def get_world(self):
        return MockWorld()
    
    def set_timeout(self, timeout):
        self.timeout = timeout


class MockWorld:
    """Mock CARLA world."""
    
    def get_blueprint_library(self):
        return MockBlueprintLibrary()
    
    def get_map(self):
        return MockMap()


class MockMap:
    def get_spawn_points(self):
        return [MockTransform()]


class MockTransform:
    """Mock CARLA transform."""
    pass


class MockBlueprintLibrary:
    def filter(self, name):
        return [MockBlueprint("vehicle.*")]
    
    def find(self, name):
        return MockBlueprint(name)


class MockBlueprint:
    def __init__(self, name):
        self.name = name
        
    def set_attribute(self, key, value):
        pass


# Test the interface
if __name__ == "__main__":
    print("=" * 60)
    print("CARLA Camera Interface Test")
    print("=" * 60)
    
    # Simulate CARLA client
    client = CarlaClient("localhost", 2000)
    print(f"Connected to CARLA at {client.host}:{client.port}")
    
    # Create camera sensor
    camera = CARLASensorCamera(image_size_x=256, image_size_y=256)
    
    frame_count = 0
    
    def process_frame(frame):
        global frame_count
        frame_count += 1
        print(f"Frame received: {frame.shape}")
    
    camera.listen(process_frame)
    
    # Simulate camera capture
    print("\nGenerating test frames...")
    for i in range(5):
        frame = camera.generate_frame()
        process_frame(frame)
        time.sleep(0.1)
    
    print(f"\n{'=' * 60}")
    print(f"CARLA Camera Interface: WORKING")
    print(f"Frames captured: {frame_count}")
    print(f"Resolution: {camera.image_size_x}x{camera.image_size_y}")
    print("=" * 60)
