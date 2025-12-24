
import torch

# PHANTASM Configuration
MODEL_NAME = "llm-jp/llm-jp-3-3.7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Directive 01 Parameters
TEMP = 1.25
TARGET_LAYERS = [26] # Layer 26 only (Directive 12)
GHOST_ALPHA = 0.002 # Strength of ghost injection (Directive 12: 0.2%)
