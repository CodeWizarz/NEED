#!/usr/bin/env python3
import re
with open('/home/Balu/alpasim/src/driver/src/alpasim_driver/main.py', 'r') as f:
    content = f.read()

# Check what we have
has_model_import = 'alpamayo_model' in content
has_registry_import = 'from alpasim_plugins.plugins import models as model_registry' in content

print(f"Has model import: {has_model_import}")
print(f"Has registry import: {has_registry_import}")

# Find where the wrongly placed patch is
wrong_start = content.find('# -- Alpamayo model registration --')
if wrong_start >= 0:
    # Find the next function definition after it
    wrong_end = content.find('\n\nif __name__', wrong_start)
    if wrong_end < 0:
        wrong_end = len(content)
    else:
        wrong_end = content.rfind('\n', wrong_start, wrong_end)
    
    wrong_code = content[wrong_start:wrong_end]
    content = content[:wrong_start] + content[wrong_end:]
    print(f"Removed wrongly placed patch ({len(wrong_code)} chars)")

# Now check if properly patched
if 'from .models.alpamayo_model import AlpamayoModel' in content and has_registry_import:
    # Check if it's in the right place (after model_registry import)
    import_line = 'from alpasim_plugins.plugins import models as model_registry'
    idx = content.find(import_line)
    next_line = content.find('\n', idx + len(import_line))
    next_20 = content[idx:idx+300]
    if 'alpamayo' in next_20:
        print("Already properly patched")
    else:
        # Need to insert after the import line
        alpamayo_block = '''
try:
    from .models.alpamayo_model import AlpamayoModel
    model_registry.plugins["alpamayo"] = AlpamayoModel
except Exception:
    pass
'''
        insert_pos = content.find(import_line) + len(import_line)
        content = content[:insert_pos] + alpamayo_block + content[insert_pos:]
        print("Added AlpamayoModel registration")
else:
    print("Cannot patch: model_registry import not found or wrong format")

with open('/home/Balu/alpasim/src/driver/src/alpasim_driver/main.py', 'w') as f:
    f.write(content)

print("Done")
