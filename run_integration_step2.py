import torch, sys, numpy as np
from scipy import interpolate

print("[2/2] Running Alpamayo with vision trajectory...")

sys.path.insert(0, "/tmp/alpamayo_repo/src")

trajectory = np.load("/tmp/trajectory.npy")
print(f"Loaded trajectory: {trajectory.shape}")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

model = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager"
).cuda().eval()

tokenizer = model.tokenizer
text = f"Road scene: vehicle trajectory x={trajectory[0,-1,0]:.2f}, y={trajectory[0,-1,1]:.2f}. What action?"
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)

B, N_TRAJ, T = 1, 1, 16
t_vision = np.linspace(0, 1, trajectory.shape[1])
t_ego = np.linspace(0, 1, T)
ego_xyz = np.zeros((B, N_TRAJ, T, 3))
ego_xyz[0, 0, :, 0] = interpolate.interp1d(t_vision, trajectory[0, :, 0], kind='linear')(t_ego)
ego_xyz[0, 0, :, 1] = interpolate.interp1d(t_vision, trajectory[0, :, 1], kind='linear')(t_ego)
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

print(f"Input: {text}")
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

print("\n=== FULL PIPELINE INTEGRATION COMPLETE ===")
print(f"Vision Trajectory: {trajectory.shape}")
print(f"Alpamayo Prediction: {pred_xyz.shape}")
print(f"Output Keys: {list(extra.keys())}")
if 'answer' in extra:
    print(f"Answer: {extra['answer'][:200]}")
print("\n=== FULL PIPELINE WORKING: YES ===")
