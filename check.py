"""Final build verification — checks imports, models, and sizes."""
import torch, os

print('=== Image Upscaler Serverless — Final Check ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
import runpod; print(f'RunPod SDK: {runpod.__version__}')
import openai; print(f'OpenAI: {openai.__version__}')

# Model path → minimum expected size in bytes
models = {
    'Diffusion':  ('models/diffusion_models/fluxSigmaVision_fp16.safetensors', 20_000_000_000),  # ~22GB
    'VAE':        ('models/vae/ae.safetensors',                                 300_000_000),      # ~335MB
    'Upscale':    ('models/upscale_models/4xNomos8k_atd_jpg.safetensors',       70_000_000),       # ~82MB
    'ControlNet': ('models/controlnet/fluxControlnetUpscale_v10.safetensors',    3_000_000_000),    # ~3.6GB
    'CLIP-G':     ('models/clip/clip_g.safetensors',                            1_000_000_000),     # ~1.4GB
    'ViT-L':      ('models/clip/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors', 800_000_000),  # ~0.9GB
    'T5XXL':      ('models/clip/t5xxl_fp16.safetensors',                        9_000_000_000),     # ~9.5GB
    'LoRA':       ('models/loras/Hyper-FLUX.1-dev-8steps-lora.safetensors',     1_000_000_000),     # ~1.4GB
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
