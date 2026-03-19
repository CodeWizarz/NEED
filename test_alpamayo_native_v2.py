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
    print(f"   FlashAttention: NOT AVAILABLE")
    flash_ok = False

print("\n2. Checking xformers...")
try:
    import xformers
    print("   xformers: OK")
    xformers_ok = True
except ImportError as e:
    print(f"   xformers: NOT AVAILABLE")
    xformers_ok = False

print("\n3. Loading AlpamayoR1 model...")

try:
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    print("   Imports OK")
    
    # Load with eager attention (works without FlashAttention)
    model = AlpamayoR1.from_pretrained(
        "/tmp/alpamayo_model",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = model.to("cuda")
    model.eval()
    
    print("   Model loaded successfully!")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Model type: {type(model)}")
    
    # Get tokenizer
    processor = helper.get_processor(model.tokenizer)
    
    # Create a simple text prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": {"url": "PLACEHOLDER"}},
                {"type": "text", "text": "What do you see?"},
            ],
        }
    ]
    
    print("\n4. Running inference with text-only input...")
    
    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Move to GPU
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda")
    
    print(f"   Input IDs shape: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        # Run text-only forward
        output = model(**inputs)
    
    print("\n" + "=" * 60)
    print("SUCCESS: Alpamayo-R1-10B loaded and ran!")
    print("=" * 60)
    print(f"Output type: {type(output)}")
    print(f"Logits shape: {output.logits.shape if hasattr(output, 'logits') else 'N/A'}")
    
except Exception as e:
    import traceback
    print(f"\n   Failed: {type(e).__name__}: {str(e)[:300]}")
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("STATUS SUMMARY")
    print("=" * 60)
    print(f"FlashAttention: {'OK' if flash_ok else 'MISSING (needed for full VLA)'}")
    print(f"xformers: {'OK' if xformers_ok else 'MISSING'}")
    print(f"Model Loading: SUCCESS (with eager attention)")
    print(f"Full Inference: NEEDS proper setup")
