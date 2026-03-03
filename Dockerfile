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
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_PATH || \
    (sleep 5 && git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_PATH)

WORKDIR $COMFYUI_PATH

# --- Install ComfyUI requirements (skip torch/torchvision/torchaudio — already in base image) ---
RUN grep -v -E "^(torch|torchvision|torchaudio)([ ><=!]|$)" requirements.txt > /tmp/comfy_req.txt && \
    pip install --no-cache-dir -r /tmp/comfy_req.txt

# --- Install custom nodes (7 repos, with retry for transient GitHub 500s) ---
RUN cd $COMFYUI_PATH/custom_nodes && \
    retry() { git clone --depth 1 "$1" || (sleep 5 && git clone --depth 1 "$1") || (sleep 10 && git clone --depth 1 "$1"); } && \
    retry https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git && \
    retry https://github.com/filliptm/ComfyUI_Fill-Nodes.git && \
    retry https://github.com/rgthree/rgthree-comfy.git && \
    retry https://github.com/EllangoK/ComfyUI-post-processing-nodes.git && \
    retry https://github.com/robertvoy/ComfyUI-Flux-Continuum.git && \
    retry https://github.com/Jonseed/ComfyUI-Detail-Daemon.git && \
    retry https://github.com/yolain/ComfyUI-Easy-Use.git

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

# --- Install RunPod SDK + OpenAI SDK (AFTER node deps, so they don't get overwritten) ---
RUN pip install --no-cache-dir \
    "runpod>=1.7.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0"

# =============================================================
# Download ALL models in one RUN (single layer = saves disk on CI)
# Total: ~39GB
# =============================================================

RUN mkdir -p $COMFYUI_PATH/models/upscale_models \
             $COMFYUI_PATH/models/vae \
             $COMFYUI_PATH/models/clip \
             $COMFYUI_PATH/models/loras \
             $COMFYUI_PATH/models/controlnet \
             $COMFYUI_PATH/models/diffusion_models && \
    echo "=== [1/8] 4xNomos8k_atd_jpg (82MB) ===" && \
    wget -q -O $COMFYUI_PATH/models/upscale_models/4xNomos8k_atd_jpg.safetensors \
    "https://huggingface.co/Phips/4xNomos8k_atd_jpg/resolve/main/4xNomos8k_atd_jpg.safetensors" && \
    echo "=== [2/8] ae.safetensors VAE (335MB) ===" && \
    wget -q -O $COMFYUI_PATH/models/vae/ae.safetensors \
    "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors" && \
    echo "=== [3/8] clip_g (1.4GB) ===" && \
    wget -q -O $COMFYUI_PATH/models/clip/clip_g.safetensors \
    "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_g.safetensors" && \
    echo "=== [4/8] ViT-L-14 (0.9GB) ===" && \
    wget -q -O "$COMFYUI_PATH/models/clip/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors" \
    "https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/resolve/main/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors" && \
    echo "=== [5/8] Hyper-FLUX LoRA (1.4GB) ===" && \
    wget -q -O $COMFYUI_PATH/models/loras/Hyper-FLUX.1-dev-8steps-lora.safetensors \
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors" && \
    echo "=== [6/8] ControlNet Upscale (3.6GB) ===" && \
    wget -q -O $COMFYUI_PATH/models/controlnet/fluxControlnetUpscale_v10.safetensors \
    "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors" && \
    echo "=== [7/8] t5xxl_fp16 (9.5GB) ===" && \
    wget -q -O $COMFYUI_PATH/models/clip/t5xxl_fp16.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" && \
    echo "=== [8/8] fluxSigmaVision_fp16 (22GB) ===" && \
    wget -q -O $COMFYUI_PATH/models/diffusion_models/fluxSigmaVision_fp16.safetensors \
    "https://civitai.com/api/download/models/1378381?token=${CIVITAI_TOKEN}" && \
    echo "=== All models downloaded ==="

# --- Copy handler + check script ---
COPY handler.py /opt/handler.py
COPY check.py /tmp/check.py

# --- Final check (validates imports + model sizes) ---
RUN python3 /tmp/check.py

# --- Entrypoint ---
CMD ["python3", "/opt/handler.py"]
