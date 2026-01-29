import os
import sys

# 1. Force the 16GB GPU
os.environ["HIP_VISIBLE_DEVICES"] = "1"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0"

# 2. MONKEY PATCH: Hide torchvision from transformers
# This stops the "operator torchvision::nms does not exist" crash
sys.modules['torchvision'] = None 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("--- 🧠 Loading Llama 3.2 1B (Stability Mode) ---")

model_id = "unsloth/Llama-3.2-1B-Instruct"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Chat logic
prompt = "Explain why an AMD RX 9060 XT is great for AI in one short sentence."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n--- 🗨️ AI Response ---")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(output[0], skip_special_tokens=True))