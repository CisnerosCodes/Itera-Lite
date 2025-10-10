import torch

cp = torch.load('checkpoints/int4/itera_lite_int4.pt', map_location='cpu', weights_only=False)
if 'config' in cp:
    config = cp['config']
    print(f"Config found: {config}")
else:
    print("No config in checkpoint")
    
embed_shape = cp['model_state_dict']['embedding.weight'].shape
print(f"Vocab size: {embed_shape[0]}, Hidden size: {embed_shape[1]}")

# Check num layers
num_layers = 0
for key in cp['model_state_dict'].keys():
    if key.startswith('layers.'):
        layer_idx = int(key.split('.')[1])
        num_layers = max(num_layers, layer_idx + 1)

print(f"Number of layers: {num_layers}")
