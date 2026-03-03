"""Final build verification — checks imports, models, and sizes."""
import torch, os

print('=== Image Upscaler Serverless — Final Check ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
import runpod; print(f'RunPod SDK: {runpod.__version__}')
import openai; print(f'OpenAI: {openai.__version__}')

# Model path → minimum expected size in bytes (90% of real size from Content-Length)
models = {
    'Diffusion':  ('models/diffusion_models/fluxSigmaVision_fp16.safetensors', 21_400_000_000),  # real: 23.78GB
    'VAE':        ('models/vae/ae.safetensors',                                 301_000_000),      # real: 335MB
    'Upscale':    ('models/upscale_models/4xNomos8k_atd_jpg.safetensors',       73_000_000),       # real: 82MB
    'ControlNet': ('models/controlnet/fluxControlnetUpscale_v10.safetensors',    3_224_000_000),    # real: 3.58GB
    'CLIP-G':     ('models/clip/clip_g.safetensors',                            1_250_000_000),     # real: 1.39GB
    'ViT-L':      ('models/clip/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors', 291_000_000),  # real: 323MB
    'T5XXL':      ('models/clip/t5xxl_fp16.safetensors',                        8_809_000_000),     # real: 9.79GB
    'LoRA':       ('models/loras/Hyper-FLUX.1-dev-8steps-lora.safetensors',     1_249_000_000),     # real: 1.39GB
}

errors = []
for name, (path, min_size) in models.items():
    full = f'/opt/ComfyUI/{path}'
    if not os.path.exists(full):
        print(f'  {name}: MISSING!')
        errors.append(f'{name}: file not found')
    else:
        actual = os.path.getsize(full)
        sz = f'{actual/1e9:.2f}GB'
        if actual < min_size:
            print(f'  {name}: {sz} — TOO SMALL (expected >{min_size/1e9:.1f}GB)')
            errors.append(f'{name}: {sz} < {min_size/1e9:.1f}GB (corrupt/incomplete?)')
        else:
            print(f'  {name}: {sz} OK')

nodes = os.listdir('/opt/ComfyUI/custom_nodes')
print(f'Custom nodes: {len(nodes)} installed')

if errors:
    print('\nFATAL ERRORS:')
    for e in errors:
        print(f'  - {e}')
    raise SystemExit(1)

print('Build OK!')
