import torch
import sys
import numpy as np

print("[2/2] Running Alpamayo with temporal input...")

sys.path.insert(0, "/tmp/alpamayo_repo/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

trajectories = np.load("/tmp/temporal_trajectories.npy")
print(f"Loaded temporal trajectories: {trajectories.shape}")

T = trajectories.shape[0]

model = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager"
).cuda().eval()

tokenizer = model.tokenizer
current_x = float(trajectories[-1, -1, 0])
current_y = float(trajectories[-1, -1, 1])
text = f"Vehicle moving through intersection with {T} frame history. Current position x={current_x:.2f}, y={current_y:.2f}. What action?"
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

print(f"Input: {T} temporal frames")
print("Running inference...")

with torch.autocast("cuda", dtype=torch.float16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=data,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=32,
        return_extra=True,
    )

print("\n=== TEMPORAL PIPELINE COMPLETE ===")
print(f"Temporal input frames: {T}")
print(f"Trajectories shape: {trajectories.shape}")
print(f"Prediction shape: {pred_xyz.shape}")
print(f"Output keys: {list(extra.keys())}")
if 'answer' in extra:
    print(f"Answer: {extra['answer']}")
print("\n=== TEMPORAL PIPELINE WORKING: YES ===")
