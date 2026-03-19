#!/usr/bin/env python3
# Patch main.py to register AlpamayoModel
with open('/home/Balu/alpasim/src/driver/src/alpasim_driver/main.py', 'r') as f:
    content = f.read()

# Check if already patched
if 'alpamayo_model' in content:
    print("Already patched")
    exit(0)

# Remove wrongly placed patch at end
wrong = '''# -- Alpamayo model registration --
try:
    from .models.alpamayo_model import AlpamayoModel
    model_registry.plugins["alpamayo"] = AlpamayoModel
except Exception as e:
    import logging
    logging.warning(f"Could not register AlpamayoModel: {e}")
'''
content = content.replace(wrong, '')
print("Removed wrongly placed patch")

# Find insertion point: after "from alpasim_plugins.plugins import models as model_registry"
insert_after = 'from alpasim_plugins.plugins import models as model_registry'

alpamayo_patch = '''from alpasim_plugins.plugins import models as model_registry
try:
    from .models.alpamayo_model import AlpamayoModel
    model_registry.plugins["alpamayo"] = AlpamayoModel
except Exception:
    pass'''

if insert_after not in content:
    print("ERROR: Could not find insertion point")
    exit(1)

content = content.replace(insert_after, alpamayo_patch, 1)
print("Added AlpamayoModel registration")

with open('/home/Balu/alpasim/src/driver/src/alpasim_driver/main.py', 'w') as f:
    f.write(content)

print("Done patching main.py")
