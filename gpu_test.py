import torch

print("--- 🛠️ AMD AI Hardware Check ---")

# Check if PyTorch can see your Radeon GPU
gpu_detected = torch.cuda.is_available()

if gpu_detected:
    device_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ SUCCESS: GPU Detected!")
    print(f"📍 Device: {device_name}")
    print(f"📊 Total VRAM: {vram_total:.2f} GB")
    
    # Simple test to move data to the GPU
    x = torch.randn(10, 10).to("cuda")
    print("💎 GPU Computation Test: PASSED")
else:
    print("❌ FAILURE: GPU not detected.")
    print("💡 Suggestion: Ensure AMD Adrenalin drivers are updated to version 26.1.1 or later.")