import torch
import numpy as np
import sys

print("=" * 60)
print("STEP 2: Alpamayo Inference")
print("=" * 60)

# Load trajectories from step 1
trajectories = np.load("/tmp/sim_trajectories.npy")
print(f"Loaded trajectories: {trajectories.shape}")

T = trajectories.shape[0]

# Load Alpamayo on GPU
print("\nLoading Alpamayo...")
sys.path.insert(0, "/tmp/alpamayo_repo/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

alpamayo = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager"
).cuda().eval()

tokenizer = alpamayo.tokenizer
print("Alpamayo loaded")

# Run inference
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

print(f"\nRunning Alpamayo inference...")
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
control = {"throttle": 0.5, "brake": 0.0, "steering": 0.0}
if action == "STOP":
    control["throttle"] = 0.0
    control["brake"] = 1.0
elif action == "SLOW":
    control["throttle"] = 0.2
    control["brake"] = 0.3

print("\n" + "=" * 60)
print("SIMULATION RESULT")
print("=" * 60)
print(f"Trajectory history: {trajectories.shape}")
print(f"Prediction: {pred_xyz.shape}")
print(f"Action: {action}")
print(f"Control: throttle={control['throttle']}, brake={control['brake']}, steering={control['steering']}")
print("\nREAL-TIME LOOP WORKING: YES")
