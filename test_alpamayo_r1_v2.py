from transformers import AutoModelForVision2Seq, AutoTokenizer
import torch

model_name = "nvidia/Alpamayo-R1-10B"

print("Loading Alpamayo-R1-10B VLM...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    add_eos_token=True
)

print("Loading model with 4-bit quantization...")
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Model loaded on:", next(model.parameters()).device)

prompt = """Camera scene:
- pedestrian crossing road
- traffic light is red
- ego vehicle approaching intersection

What should the vehicle do?"""

print("Running inference...")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        temperature=None,
        top_p=None
    )

print("\n=== Generated Output ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
