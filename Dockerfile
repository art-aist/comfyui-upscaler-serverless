# =============================================================
# ComfyUI Image Upscaler — RunPod SERVERLESS
# =============================================================
# FLUX-based image upscaling with ControlNet guidance.
#
# What's inside:
#   - ComfyUI (latest)
#   - 7 custom nodes for upscaling workflow
#   - 8 models baked in (~39GB total)
#   - RunPod serverless handler with OpenAI captioning
#
# Build on RunPod Pod (Docker-in-Docker):
#   bash build_and_push.sh <dockerhub_user>
#
# Build with CivitAI token for fluxSigmaVision model:
#   docker build --build-arg CIVITAI_TOKEN=<your_token> -t upscaler .
# =============================================================

FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

LABEL description="ComfyUI Image Upscaler — RunPod Serverless, FLUX + ControlNet"

# --- Build args ---
ARG CIVITAI_TOKEN=""

# --- Env ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV COMFYUI_PATH=/opt/ComfyUI
ENV COMFYUI_PORT=8188

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Clone ComfyUI ---
RUN git clone https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_PATH

WORKDIR $COMFYUI_PATH

# --- Install ComfyUI requirements (skip torch/torchvision/torchaudio — already in base image) ---
RUN grep -v -E "^(torch|torchvision|torchaudio)([ ><=!]|$)" requirements.txt > /tmp/comfy_req.txt && \
    pip install --no-cache-dir -r /tmp/comfy_req.txt

# --- Install RunPod SDK + OpenAI SDK ---
RUN pip install --no-cache-dir \
    "runpod>=1.7.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0"

# --- Install custom nodes (7 repos) ---
RUN cd $COMFYUI_PATH/custom_nodes && \
    git clone --depth 1 https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git && \
    git clone --depth 1 https://github.com/filliptm/ComfyUI_Fill-Nodes.git && \
    git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
    git clone --depth 1 https://github.com/EllangoK/ComfyUI-post-processing-nodes.git && \
    git clone --depth 1 https://github.com/robertvoy/ComfyUI-Flux-Continuum.git && \
    git clone --depth 1 https://github.com/Jonseed/ComfyUI-Detail-Daemon.git && \
    git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git

# --- Install node pip requirements ---
RUN FAILED="" && \
    for d in $COMFYUI_PATH/custom_nodes/*/; do \
      name=$(basename "$d"); \
      if [ -f "$d/requirements.txt" ]; then \
        echo "=== pip: $name ==="; \
        grep -v -E "^(torch|torchvision|torchaudio)([ ><=!]|$)" "$d/requirements.txt" > /tmp/node_req.txt 2>/dev/null; \
        pip install --no-cache-dir -r /tmp/node_req.txt 2>&1 || FAILED="$FAILED $name"; \
      fi; \
      if [ -f "$d/install.py" ]; then \
        echo "=== install.py: $name ==="; \
        (cd "$d" && python3 install.py) 2>&1 || echo "  [warn] install.py failed for $name"; \
      fi; \
    done && \
    if [ -n "$FAILED" ]; then \
      echo "WARNING: pip failed for:$FAILED (may be ok, check at runtime)"; \
    else \
      echo "All node deps installed OK"; \
    fi

# =============================================================
# Download models (8 models, ~39GB total)
# Each large model is a separate RUN for Docker layer caching.
# =============================================================

# --- Small models first (cache-friendly) ---

# 4x Upscale Model (~82MB)
RUN mkdir -p $COMFYUI_PATH/models/upscale_models && \
    echo "Downloading 4xNomos8k_atd_jpg.pth..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/upscale_models/4xNomos8k_atd_jpg.pth \
    "https://huggingface.co/Phips/4xNomos8k_atd_jpg/resolve/main/4xNomos8k_atd_jpg.pth"

# FLUX VAE (~335MB)
RUN mkdir -p $COMFYUI_PATH/models/vae && \
    echo "Downloading ae.safetensors..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/vae/ae.safetensors \
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"

# CLIP-G (~1.4GB) + ViT-L-14 (~0.9GB)
RUN mkdir -p $COMFYUI_PATH/models/clip && \
    echo "Downloading clip_g.safetensors..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/clip/clip_g.safetensors \
    "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_g.safetensors" && \
    echo "Downloading ViT-L-14..." && \
    wget -q --show-progress -O "$COMFYUI_PATH/models/clip/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors" \
    "https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/resolve/main/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors"

# Hyper-FLUX LoRA (~1.39GB)
RUN mkdir -p $COMFYUI_PATH/models/loras && \
    echo "Downloading Hyper-FLUX LoRA..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/loras/Hyper-FLUX.1-dev-8steps-lora.safetensors \
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors"

# --- Large models (separate layers) ---

# ControlNet Upscale (~3.58GB) — renamed from diffusion_pytorch_model.safetensors
RUN mkdir -p $COMFYUI_PATH/models/controlnet && \
    echo "Downloading ControlNet Upscale..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/controlnet/fluxControlnetUpscale_v10.safetensors \
    "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors"

# T5-XXL FP16 (~9.5GB)
RUN echo "Downloading t5xxl_fp16.safetensors..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/clip/t5xxl_fp16.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"

# FLUX Sigma Vision Alpha1 (~22.15GB) — from CivitAI, renamed
RUN mkdir -p $COMFYUI_PATH/models/diffusion_models && \
    echo "Downloading fluxSigmaVision_fp16.safetensors (~22GB, this will take a while)..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/diffusion_models/fluxSigmaVision_fp16.safetensors \
    "https://civitai.com/api/download/models/1378381?token=${CIVITAI_TOKEN}"

# --- Copy handler ---
COPY handler.py /opt/handler.py

# --- Final check ---
RUN python3 -c "\
import torch; \
print('=== Image Upscaler Serverless — Final Check ==='); \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA: {torch.version.cuda}'); \
import runpod; print(f'RunPod SDK: {runpod.__version__}'); \
import openai; print(f'OpenAI: {openai.__version__}'); \
import os; \
models = { \
    'Diffusion': 'models/diffusion_models/fluxSigmaVision_fp16.safetensors', \
    'VAE': 'models/vae/ae.safetensors', \
    'Upscale': 'models/upscale_models/4xNomos8k_atd_jpg.pth', \
    'ControlNet': 'models/controlnet/fluxControlnetUpscale_v10.safetensors', \
    'CLIP-G': 'models/clip/clip_g.safetensors', \
    'ViT-L': 'models/clip/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors', \
    'T5XXL': 'models/clip/t5xxl_fp16.safetensors', \
    'LoRA': 'models/loras/Hyper-FLUX.1-dev-8steps-lora.safetensors', \
}; \
for name, path in models.items(): \
    full = f'/opt/ComfyUI/{path}'; \
    sz = f'{os.path.getsize(full)/1e9:.2f}GB' if os.path.exists(full) else 'MISSING!'; \
    print(f'  {name}: {sz}'); \
nodes = os.listdir('/opt/ComfyUI/custom_nodes'); \
print(f'Custom nodes: {len(nodes)} installed'); \
print('Build OK!'); \
"

# --- Entrypoint ---
CMD ["python3", "/opt/handler.py"]
