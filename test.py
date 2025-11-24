from exllamav3 import Config, Model
import torch

print("=" * 60)
print("Step 1: Initializing model...")
print("=" * 60)
config = Config.from_directory('/srv/data/llm-models/Kimi-Linear-48B-A3B-Instruct')
model = Model.from_config(config)
print('✓ Model initialized successfully')
print(f'  Architecture: {config.architecture}')
print(f'  Config class: {type(config).__name__}')
print(f'  Model class: {type(model).__name__}')
print(f'  Number of layers: {config.num_hidden_layers}')
print(f'  Hidden size: {config.hidden_size}')
print(f'  KDA layers: {len(config.kda_layers)}')
print(f'  MLA layers: {len(config.full_attn_layers)}')
print(f'  MoE experts: {config.num_experts}')
print()

print("=" * 60)
print("Step 2: Loading model weights...")
print("=" * 60)

# Split across 3 L40s (48GB each = 144GB total)
gpu_split = [47.0, 47.0, 47.0]  # GB per device
print(f"  Using GPU split: {gpu_split}")
print(f"  Total VRAM: {sum(gpu_split)}GB across {len(gpu_split)} GPUs")

try:
    model.load(split=gpu_split, progress=True)
    print('✓ Model loaded successfully!')
    print()

    print("=" * 60)
    print("Step 3: Model structure verification")
    print("=" * 60)

    # Check a few layers
    for idx in [0, 3]:  # KDA and MLA layer
        layer = model.modules[model.first_block_idx + idx]
        attn_type = type(layer.attn).__name__
        mlp_type = type(layer.mlp).__name__
        print(f"  Layer {idx}:")
        print(f"    Attention: {attn_type}")
        print(f"    MLP: {mlp_type}")

    print()
    print("=" * 60)
    print("SUCCESS! Model is ready for conversion/inference")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\nThis likely means there's a tensor key mismatch.")
