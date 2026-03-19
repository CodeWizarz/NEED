import torch
import sys

sys.path.insert(0, "/tmp/alpamayo_repo/src")

print("=" * 60)
print("ALPAMAYO-R1-10B EAGER MODE TEST")
print("=" * 60)

print("\nLoading Alpamayo model...")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

# Use float16 for model
model = AlpamayoR1.from_pretrained(
    "/tmp/alpamayo_model",
    dtype=torch.float16,
    attn_implementation="eager",
)
model = model.cuda()
model.eval()

print("Model ready (EAGER MODE)")

# Create minimal data dict
print("\nPreparing input data...")

tokenizer = model.tokenizer
text = "What do you see?"
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)

input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Use float32 for action space (required for cholesky)
B, N_TRAJ, T = 1, 1, 16
ego_xyz = torch.randn(B, N_TRAJ, T, 3, dtype=torch.float32).cuda()
ego_rot = torch.randn(B, N_TRAJ, T, 3, 3, dtype=torch.float32).cuda()

data = {
    "tokenized_data": {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
    },
    "ego_history_xyz": ego_xyz,
    "ego_history_rot": ego_rot,
}

print(f"Input IDs: {input_ids.shape}")
print(f"Ego XYZ: {ego_xyz.shape}")

print("\nRunning sample_trajectories_from_data_with_vlm_rollout...")

try:
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=data,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=32,
            return_extra=True,
        )
    
    print("\n" + "=" * 60)
    print("✅ ALPAMAYO FORWARD PASS SUCCESS!")
    print("=" * 60)
    print(f"Pred XYZ shape: {pred_xyz.shape}")
    print(f"Pred Rot shape: {pred_rot.shape}")
    print(f"Extra keys: {list(extra.keys()) if isinstance(extra, dict) else 'N/A'}")
    
except Exception as e:
    import traceback
    print(f"\n❌ Forward failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
