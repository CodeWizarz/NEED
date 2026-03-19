#!/usr/bin/env python3
import base64
with open('/Users/Balu/Documents/NEED/patch_main.py', 'r') as f:
    data = f.read()
b64 = base64.b64encode(data.encode()).decode()
with open('/tmp/patch_main.b64', 'w') as f:
    f.write(b64)
print(f"Written {len(b64)} bytes")
