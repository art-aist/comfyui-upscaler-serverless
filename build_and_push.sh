#!/bin/bash
# =============================================================
# Build and push Docker image for ComfyUI Image Upscaler
# Run this on a RunPod Pod with Docker access.
#
# Usage:
#   bash build_and_push.sh <dockerhub_username> [civitai_token]
#
# Example:
#   bash build_and_push.sh art1aist civ_abc123...
#
# Build time: ~30-40 minutes (mostly model downloads)
# =============================================================

set -e

DOCKERHUB_USER="${1:?Usage: bash build_and_push.sh <dockerhub_user> [civitai_token]}"
CIVITAI_TOKEN="${2:-}"
IMAGE_NAME="comfyui-upscaler-serverless"
TAG="latest"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "============================================="
echo "  Building: ${FULL_IMAGE}"
echo "============================================="

# Login to DockerHub
echo ""
echo "--- Step 1: Docker login ---"
docker login -u "${DOCKERHUB_USER}"

# Build
echo ""
echo "--- Step 2: Building image (~30-40 min) ---"
echo "  CivitAI token: $([ -n "$CIVITAI_TOKEN" ] && echo 'provided' || echo 'NOT provided (fluxSigmaVision may fail)')"
echo ""

docker build \
    --build-arg CIVITAI_TOKEN="${CIVITAI_TOKEN}" \
    -t "${FULL_IMAGE}" \
    .

# Verify
echo ""
echo "--- Step 3: Verifying image ---"
docker run --rm "${FULL_IMAGE}" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
import runpod
print(f'RunPod: {runpod.__version__}')
print('Image OK!')
"

# Push
echo ""
echo "--- Step 4: Pushing to DockerHub ---"
docker push "${FULL_IMAGE}"

echo ""
echo "============================================="
echo "  Done! Image pushed to: ${FULL_IMAGE}"
echo ""
echo "  Next steps:"
echo "  1. Go to RunPod Serverless"
echo "  2. Create new Endpoint"
echo "  3. Image: ${FULL_IMAGE}"
echo "  4. GPU: RTX 5090 32GB or A40 48GB"
echo "  5. Min Workers: 0, Max Workers: 3"
echo "  6. Idle Timeout: 30s"
echo "  7. Execution Timeout: 1800s"
echo "  8. Container Disk: 50GB"
echo "  9. Env vars: COMFYUI_STARTUP_TIMEOUT=180"
echo "     OPENAI_API_KEY=sk-..."
echo "============================================="
