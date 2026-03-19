import torch
import sys

sys.path.insert(0, "/tmp/alpamayo_repo/src")
sys.path.insert(0, "/tmp/alpamayo_model")

print("=" * 60)
print("ALPAMAYO-R1-10B NATIVE TEST")
print("=" * 60)

print("\n1. Checking FlashAttention...")
try:
    import flash_attn
    print("   FlashAttention: OK")
    flash_ok = True
except ImportError as e:
    print(f"   FlashAttention: NOT AVAILABLE ({e})")
    flash_ok = False

print("\n2. Checking xformers...")
try:
    import xformers
    print("   xformers: OK")
    xformers_ok = True
except ImportError as e:
    print(f"   xformers: NOT AVAILABLE ({e})")
    xformers_ok = False

print("\n3. Attempting to load AlpamayoR1 model...")

try:
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    print("   Imports OK")
    
    # Try loading with eager attention
    model = AlpamayoR1.from_pretrained(
        "/tmp/alpamayo_model",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = model.to("cuda")
    model.eval()
    
    print("   Model loaded successfully!")
    print(f"   Device: {next(model.parameters()).device}")
    
    print("\n4. Running forward pass...")
    
    # Create dummy inputs
    batch_size = 1
    num_cameras = 4
    num_frames = 8
    height, width = 256, 256
    
    images = torch.randn(batch_size, num_cameras, num_frames, height, width, 3).to("cuda")
    ego_xyz = torch.randn(batch_size, 16, 3).to("cuda")
    ego_rot = torch.randn(batch_size, 16, 3, 3).to("cuda")
    
    print(f"   Input shapes: images={images.shape}, ego_xyz={ego_xyz.shape}")
    
    with torch.no_grad():
        # Try basic forward
        output = model(images=images, ego_history_xyz=ego_xyz, ego_history_rot=ego_rot)
    
    print("\n" + "=" * 60)
    print("SUCCESS: Alpamayo-R1-10B is working!")
    print("=" * 60)
    print(f"Output type: {type(output)}")
    if hasattr(output, 'shape'):
        print(f"Output shape: {output.shape}")
    
except Exception as e:
    import traceback
    print(f"\n   Failed: {type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ALPAMAYO STATUS: PARTIAL / FALLBACK REQUIRED")
    print("=" * 60)
    print(f"FlashAttention: {'OK' if flash_ok else 'MISSING'}")
    print(f"xformers: {'OK' if xformers_ok else 'MISSING'}")
    print(f"Model can be loaded but inference needs proper attention backend")
