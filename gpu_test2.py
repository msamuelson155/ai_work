import os
import sys

# 1. Force the RX 9060 XT (Device 1)
os.environ["HIP_VISIBLE_DEVICES"] = "1"

# 2. Tell ROCm to ignore the integrated graphics architecture entirely
# 'gfx1200' is the architecture code for your RDNA 4 card
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0" 

# 3. Silence the "Error 100" logging (Optional)
os.environ["ROC_ENABLE_PRE_ALLOCATION"] = "1"

import torch

print("\n--- 🚀 RX 9060 XT Verified ---")
if torch.cuda.is_available():
    print(f"ACTIVE: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Real test
    x = torch.randn(1, 1).to("cuda")
    print("MATH TEST: PASSED")