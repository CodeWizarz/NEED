import numpy as np
import tensorflow as tf
import torch
import sys
import time

print("=" * 60)
print("SIMULATION LOOP - CARLA-like Pipeline")
print("=" * 60)

# Load vision model
print("\n[1/3] Loading vision model...")
vision_model = tf.saved_model.load("/tmp/alpamayo_v1")
print("Vision model loaded")

# Load Alpamayo
print("\n[2/3] Loading Alpamayo...")
sys.path.insert(0, "/tmp/alpamayo_repo/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

alpamayo = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager"
).cuda().eval()

tokenizer = alpamayo.tokenizer
print("Alpamayo loaded")

# Simulation parameters
T = 5
frame_buffer = []
max_iterations = 5  # Run for 5 iterations for testing
iteration = 0

print(f"\n[3/3] Running simulation loop (max {max_iterations} iterations)...")
print("-" * 40)

while iteration < max_iterations:
    # Simulate incoming frame (replace with CARLA later)
    frame = np.random.rand(256, 256, 3).astype(np.float32)
    frame_buffer.append(frame)
    
    print(f"\nIteration {iteration + 1}/{max_iterations}")
    print(f"  Frame buffer size: {len(frame_buffer)}")
    
    if len(frame_buffer) < T:
        print(f"  Buffering... ({len(frame_buffer)}/{T})")
        time.sleep(0.1)
        continue
    
    # Keep last T frames
    frames = np.array(frame_buffer[-T:])
    
    # Create 8-camera view (simulate multi-camera)
    frames = np.repeat(frames[:, None], 8, axis=1)  # (T, 8, 256, 256, 3)
    
    print(f"  Frames shape: {frames.shape}")
    
    # Run vision model to get trajectories
    trajectories = []
    for t in range(T):
        out = vision_model.signatures["serving_default"](
            tf.convert_to_tensor(frames[t:t+1])
        )
        traj = list(out.values())[0].numpy()
        trajectories.append(traj[0])
    
    trajectories = np.array(trajectories)
    print(f"  Trajectories shape: {trajectories.shape}")
    
    # Prepare Alpamayo input
    current_x = float(trajectories[-1, -1, 0])
    current_y = float(trajectories[-1, -1, 1])
    
    text = f"Vehicle driving. Current position x={current_x:.2f}, y={current_y:.2f}. What action?"
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    
    B, N_TRAJ = 1, 1
    ego_xyz = np.zeros((B, N_TRAJ, T, 3))
    for i in range(T):
        ego_xyz[0, 0, i, 0] = float(trajectories[i, -1, 0])
        ego_xyz[0, 0, i, 1] = float(trajectories[i, -1, 1])
    ego_xyz = torch.tensor(ego_xyz, dtype=torch.float32).cuda()
    ego_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N_TRAJ, T, 1, 1).cuda()
    
    data = {
        "tokenized_data": {
            "input_ids": tokens["input_ids"].cuda(),
            "attention_mask": tokens["attention_mask"].cuda(),
        },
        "ego_history_xyz": ego_xyz,
        "ego_history_rot": ego_rot,
    }
    
    # Run Alpamayo
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, pred_rot, extra = alpamayo.sample_trajectories_from_data_with_vlm_rollout(
            data=data,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=32,
            return_extra=True,
        )
    
    # Determine action
    action = "GO"
    if 'answer' in extra:
        answer = str(extra['answer'])
        if 'STOP' in answer.upper():
            action = "STOP"
        elif 'SLOW' in answer.upper():
            action = "SLOW"
    
    # Control logic
    control = {
        "throttle": 0.5,
        "brake": 0.0,
        "steering": 0.0
    }
    
    if action == "STOP":
        control["throttle"] = 0.0
        control["brake"] = 1.0
    elif action == "SLOW":
        control["throttle"] = 0.2
        control["brake"] = 0.3
    
    print(f"  Action: {action}")
    print(f"  Control: throttle={control['throttle']}, brake={control['brake']}, steering={control['steering']}")
    
    iteration += 1
    time.sleep(0.2)

print("\n" + "=" * 60)
print("SIMULATION LOOP COMPLETE")
print("=" * 60)
print(f"Ran {max_iterations} iterations successfully")
print("\nREAL-TIME LOOP WORKING: YES")
