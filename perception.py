"""
Perception Module - Object Detection with YOLO
"""

from ultralytics import YOLO
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("YOLO loaded!")

def detect_objects(frame):
    """Detect objects in frame using YOLO"""
    results = model(frame, verbose=False, conf=0.5)[0]
    
    detections = []
    
    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # COCO class names: 0=person, 2=car, 3=motorcycle, 7=truck
            if cls in [0, 2, 3, 7]:  # Focus on vehicles and people
                detections.append({
                    "class": cls,
                    "class_name": ["person", "car", "motorcycle", "truck"][cls] if cls < 4 else "vehicle",
                    "confidence": conf,
                    "bbox": box.xyxy[0].cpu().numpy().tolist()
                })
    
    return detections

def decide_action(detections, frame_shape):
    """Rule-based decision from detections"""
    h, w = frame_shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Check for obstacles in path
    for d in detections:
        bbox = d["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Check if object is in front (center of image, lower half)
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # In front of car?
        in_path = (center_y < obj_center_y < h) and (w * 0.2 < obj_center_x < w * 0.8)
        
        # Close distance? (large bbox)
        obj_area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        is_close = obj_area > frame_area * 0.05
        
        if in_path and is_close:
            return "STOP", d
        
        if in_path and not is_close:
            return "SLOW", d
    
    return "GO", None
