"""
RunPod Serverless Handler for ComfyUI Image Upscaler.

Supports one mode:
  - "upscale": Upload image -> OpenAI caption -> ComfyUI FLUX upscale -> return result

Input format:
{
    "input": {
        "mode": "upscale",
        "image": "<base64>",
        "image_name": "photo.png",

        # Main settings:
        "denoise": 0.2,
        "upscale_by": 2,

        # Advanced settings (all optional):
        "prompt": "...",
        "controlnet_strength": 0.6,
        "controlnet_start_percent": 0,
        "controlnet_end_percent": 0.6,
        "steps": 20,
        "cfg": 4,
        "sigma_multiplier": 0.5,
        "slicing": "2x2",
        "film_grain_intensity": 0.05,
        "seed": 12345,
        "mask_blur": 8,
        "tile_padding": 64,
        "seam_fix_mode": "None",
        "seam_fix_denoise": 1,
        "seam_fix_width": 64,
        "seam_fix_mask_blur": 8,
        "seam_fix_padding": 16,
        "lora_strength": 0.12,
    }
}
"""

import os
import sys
import copy
import time
import uuid
import random
import base64
import subprocess
import threading
import traceback
import requests
from io import BytesIO

from PIL import Image
import runpod

# --- Configuration ---
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/opt/ComfyUI")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_HOST = f"http://127.0.0.1:{COMFYUI_PORT}"
COMFYUI_STARTUP_TIMEOUT = int(os.environ.get("COMFYUI_STARTUP_TIMEOUT", "180"))
COMFYUI_ARGS = os.environ.get("COMFYUI_ARGS", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

comfyui_process = None

MAX_IMAGE_DIM = 4000
MAX_RESPONSE_BYTES = 20 * 1024 * 1024  # 20MB RunPod response limit
OOM_MAX_RETRIES = 2
OOM_UPSCALE_REDUCTION = 0.5


# ============================================================
# Embedded Workflow (converted from upscale_comfy.json)
# Node 216 (OpenAIChatNode) REMOVED — captioning done in Python.
# Disabled LoRAs removed — only Hyper-FLUX active.
# ============================================================

UPSCALE_WORKFLOW = {
    "1": {
        "inputs": {"vae_name": "ae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "4": {
        "inputs": {"model_name": "4xNomos8k_atd_jpg.safetensors"},
        "class_type": "UpscaleModelLoader",
        "_meta": {"title": "Load Upscale Model"},
    },
    "7": {
        "inputs": {
            "upscale_by": ["69", 3],
            "seed": 10665517828115,
            "steps": 20,
            "cfg": 4,
            "sampler_name": "euler",
            "scheduler": "beta",
            "denoise": ["38", 0],
            "mode_type": "Linear",
            "tile_width": ["69", 1],
            "tile_height": ["69", 2],
            "mask_blur": 8,
            "tile_padding": 64,
            "seam_fix_mode": "None",
            "seam_fix_denoise": 1,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16,
            "force_uniform_tiles": True,
            "tiled_decode": False,
            "image": ["69", 0],
            "model": ["179", 0],
            "positive": ["81", 0],
            "negative": ["81", 1],
            "vae": ["1", 0],
            "upscale_model": ["4", 0],
            "custom_sigmas": ["68", 0],
        },
        "class_type": "UltimateSDUpscaleCustomSample",
        "_meta": {"title": "Ultimate SD Upscale (Custom Sample)"},
    },
    "38": {
        "inputs": {"value": 0.2},
        "class_type": "DenoiseSlider",
        "_meta": {"title": "Upscaling strength"},
    },
    "67": {
        "inputs": {
            "scheduler": "beta",
            "steps": 20,
            "denoise": 1,
            "model": ["179", 0],
        },
        "class_type": "BasicScheduler",
        "_meta": {"title": "BasicScheduler"},
    },
    "68": {
        "inputs": {
            "factor": 0.5,
            "start": 0,
            "end": 1,
            "sigmas": ["67", 0],
        },
        "class_type": "MultiplySigmas",
        "_meta": {"title": "Multiply Sigmas (stateless)"},
    },
    "69": {
        "inputs": {
            "slicing": "2x2",
            "multiplier": 2,
            "image": ["186", 0],
        },
        "class_type": "FL_SDUltimate_Slices",
        "_meta": {"title": "FL SDUltimate Slices"},
    },
    "80": {
        "inputs": {"control_net_name": "fluxControlnetUpscale_v10.safetensors"},
        "class_type": "ControlNetLoader",
        "_meta": {"title": "Load ControlNet Model"},
    },
    "81": {
        "inputs": {
            "strength": 0.6,
            "start_percent": 0,
            "end_percent": 0.6,
            "positive": ["180", 0],
            "negative": ["181", 0],
            "control_net": ["80", 0],
            "image": ["186", 0],
            "vae": ["1", 0],
        },
        "class_type": "ControlNetApplyAdvanced",
        "_meta": {"title": "Apply ControlNet"},
    },
    "88": {
        "inputs": {"image": "input.png"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "155": {
        "inputs": {
            "unet_name": "fluxSigmaVision_fp16.safetensors",
            "weight_dtype": "fp8_e4m3fn",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "179": {
        "inputs": {
            "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
            "lora_1": {
                "on": True,
                "lora": "Hyper-FLUX.1-dev-8steps-lora.safetensors",
                "strength": 0.12,
            },
            "\u2795 Add Lora": "",
            "model": ["155", 0],
            "clip": ["189", 0],
        },
        "class_type": "Power Lora Loader (rgthree)",
        "_meta": {"title": "Power Lora Loader (rgthree)"},
    },
    "180": {
        "inputs": {
            "text": "",  # Filled by handler with OpenAI caption or user prompt
            "clip": ["179", 1],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "181": {
        "inputs": {
            "text": "",
            "clip": ["179", 1],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "186": {
        "inputs": {
            "intensity": 0.05,
            "scale": 1,
            "temperature": 0,
            "vignette": 0,
            "image": ["88", 0],
        },
        "class_type": "FilmGrain",
        "_meta": {"title": "FilmGrain"},
    },
    "189": {
        "inputs": {
            "clip_name1": "clip_g.safetensors",
            "clip_name2": "ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors",
            "clip_name3": "t5xxl_fp16.safetensors",
        },
        "class_type": "TripleCLIPLoader",
        "_meta": {"title": "TripleCLIPLoader"},
    },
    "200": {
        "inputs": {"anything": ["7", 0]},
        "class_type": "easy clearCacheAll",
        "_meta": {"title": "Clear Cache All"},
    },
    "201": {
        "inputs": {"anything": ["200", 0]},
        "class_type": "easy cleanGpuUsed",
        "_meta": {"title": "Clean VRAM Used"},
    },
    "215": {
        "inputs": {"images": ["7", 0]},
        "class_type": "PreviewImage",
        "_meta": {"title": "Preview Image"},
    },
}


# ============================================================
# ComfyUI Management
# ============================================================

def start_comfyui():
    """Start ComfyUI server in background."""
    global comfyui_process

    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1",
        "--port", str(COMFYUI_PORT),
        "--disable-auto-launch",
    ]
    if COMFYUI_ARGS:
        cmd.extend(COMFYUI_ARGS.split())

    print(f"[handler] Starting ComfyUI: {' '.join(cmd)}")
    comfyui_process = subprocess.Popen(
        cmd,
        cwd=COMFYUI_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def log_output():
        for line in comfyui_process.stdout:
            print(f"[comfyui] {line}", end="")

    log_thread = threading.Thread(target=log_output, daemon=True)
    log_thread.start()


def wait_for_comfyui():
    """Wait until ComfyUI is ready to accept requests."""
    print(f"[handler] Waiting for ComfyUI (timeout: {COMFYUI_STARTUP_TIMEOUT}s)...")
    start = time.time()

    while time.time() - start < COMFYUI_STARTUP_TIMEOUT:
        try:
            resp = requests.get(f"{COMFYUI_HOST}/system_stats", timeout=2)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"[handler] ComfyUI ready in {elapsed:.1f}s")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[handler] Health check error: {e}")

        if comfyui_process and comfyui_process.poll() is not None:
            print(f"[handler] ComfyUI crashed with code {comfyui_process.returncode}")
            return False

        time.sleep(2)

    print(f"[handler] ComfyUI startup timeout ({COMFYUI_STARTUP_TIMEOUT}s)")
    return False


# ============================================================
# ComfyUI API Functions
# ============================================================

def upload_image(name, image_base64):
    """Upload a base64 image to ComfyUI's input directory."""
    image_data = base64.b64decode(image_base64)

    resp = requests.post(
        f"{COMFYUI_HOST}/upload/image",
        files={"image": (name, BytesIO(image_data), "image/png")},
        data={"overwrite": "true"},
    )

    if resp.status_code == 200:
        result = resp.json()
        print(f"[handler] Uploaded: {name} -> {result.get('name', name)}")
        return result.get("name", name)
    else:
        raise RuntimeError(f"Image upload failed ({resp.status_code}): {resp.text}")


def queue_workflow(workflow):
    """Send workflow to ComfyUI's /prompt endpoint."""
    client_id = str(uuid.uuid4())

    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }

    resp = requests.post(f"{COMFYUI_HOST}/prompt", json=payload)

    if resp.status_code == 200:
        result = resp.json()
        prompt_id = result.get("prompt_id")
        print(f"[handler] Queued workflow: prompt_id={prompt_id}")
        return prompt_id, client_id
    else:
        raise RuntimeError(f"Workflow queue failed ({resp.status_code}): {resp.text}")


def wait_for_completion(prompt_id, timeout=1800, poll_interval=2):
    """Poll ComfyUI until the workflow completes or fails."""
    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{COMFYUI_HOST}/history/{prompt_id}", timeout=5)
            if resp.status_code == 200:
                history = resp.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False):
                        elapsed = time.time() - start
                        print(f"[handler] Workflow completed in {elapsed:.1f}s")
                        return entry
                    if status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        raise RuntimeError(f"Workflow error: {messages}")
        except requests.exceptions.ConnectionError:
            pass
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[handler] Poll error: {e}")

        time.sleep(poll_interval)

    raise RuntimeError(f"Workflow timeout after {timeout}s")


def fetch_image_from_history(history_entry, node_id):
    """Download an image from a specific node in ComfyUI history."""
    outputs = history_entry.get("outputs", {})
    node_output = outputs.get(node_id, {})

    if "images" not in node_output or not node_output["images"]:
        raise RuntimeError(f"No images in node {node_id} output")

    img_info = node_output["images"][0]
    resp = requests.get(
        f"{COMFYUI_HOST}/view",
        params={
            "filename": img_info["filename"],
            "subfolder": img_info.get("subfolder", ""),
            "type": img_info.get("type", "output"),
        },
        timeout=60,
    )

    if resp.status_code == 200:
        print(f"[handler] Fetched node {node_id}: {img_info['filename']} "
              f"({len(resp.content) // 1024}KB)")
        return resp.content
    else:
        raise RuntimeError(f"Failed to fetch node {node_id}: {resp.status_code}")


# ============================================================
# Image Processing
# ============================================================

def caption_image(image_base64):
    """Call OpenAI API to generate an image caption for use as prompt."""
    if not OPENAI_API_KEY:
        print("[handler] No OPENAI_API_KEY — using empty prompt")
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image as a prompt for an image generation model"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "low",
                    }},
                ],
            }],
            max_completion_tokens=300,
        )
        caption = response.choices[0].message.content.strip()
        print(f"[handler] Caption ({len(caption)} chars): {caption[:100]}...")
        return caption
    except Exception as e:
        print(f"[handler] OpenAI captioning failed: {e}")
        return ""


def validate_and_resize(image_base64):
    """Validate input image and resize if any dimension > MAX_IMAGE_DIM.
    Returns (new_base64, pil_image)."""
    image_data = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image_data))
    if img.mode == "RGBA":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        ratio = MAX_IMAGE_DIM / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[handler] Resized {w}x{h} -> {new_w}x{new_h}")

    buf = BytesIO()
    img.save(buf, format="PNG")
    new_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return new_b64, img


def adaptive_compress(raw_bytes):
    """Compress output to fit within RunPod response limit.
    PNG if <10MB, JPEG q=95 otherwise. Progressive fallback."""
    if len(raw_bytes) <= 10 * 1024 * 1024:
        return raw_bytes

    img = Image.open(BytesIO(raw_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")

    for quality in [95, 90, 85, 80]:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        compressed = buf.getvalue()
        size_kb = len(compressed) // 1024
        print(f"[handler] JPEG q={quality}: {size_kb}KB")

        # base64 adds ~33% overhead
        if len(compressed) * 1.34 < MAX_RESPONSE_BYTES:
            print(f"[handler] Compressed {len(raw_bytes) // 1024}KB -> {size_kb}KB (JPEG q={quality})")
            return compressed

    # Last resort: downscale
    w, h = img.size
    ratio = 0.75
    img_small = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img_small.save(buf, format="JPEG", quality=85)
    compressed = buf.getvalue()
    print(f"[handler] Downscaled {w}x{h} -> {int(w*ratio)}x{int(h*ratio)} + JPEG q=85: "
          f"{len(compressed) // 1024}KB")
    return compressed


def is_oom_error(error_msg):
    """Check if an error message indicates an out-of-memory condition."""
    oom_keywords = ["out of memory", "oom", "cuda out of memory", "allocat"]
    error_lower = str(error_msg).lower()
    return any(kw in error_lower for kw in oom_keywords)


# ============================================================
# Upscale Handler
# ============================================================

def build_workflow(job_input, upscale_override=None):
    """Build the ComfyUI workflow dict from job_input parameters."""
    wf = copy.deepcopy(UPSCALE_WORKFLOW)

    # Randomize seed
    wf["7"]["inputs"]["seed"] = random.randint(1, 2**53)

    # --- Main settings ---
    if "denoise" in job_input:
        wf["38"]["inputs"]["value"] = float(job_input["denoise"])

    upscale_by = upscale_override or job_input.get("upscale_by")
    if upscale_by is not None:
        wf["69"]["inputs"]["multiplier"] = float(upscale_by)

    # --- Advanced settings ---
    if "controlnet_strength" in job_input:
        wf["81"]["inputs"]["strength"] = float(job_input["controlnet_strength"])
    if "controlnet_start_percent" in job_input:
        wf["81"]["inputs"]["start_percent"] = float(job_input["controlnet_start_percent"])
    if "controlnet_end_percent" in job_input:
        wf["81"]["inputs"]["end_percent"] = float(job_input["controlnet_end_percent"])
    if "steps" in job_input:
        wf["7"]["inputs"]["steps"] = int(job_input["steps"])
        wf["67"]["inputs"]["steps"] = int(job_input["steps"])
    if "cfg" in job_input:
        wf["7"]["inputs"]["cfg"] = float(job_input["cfg"])
    if "seed" in job_input:
        wf["7"]["inputs"]["seed"] = int(job_input["seed"])
    if "sigma_multiplier" in job_input:
        wf["68"]["inputs"]["factor"] = float(job_input["sigma_multiplier"])
    if "slicing" in job_input:
        wf["69"]["inputs"]["slicing"] = str(job_input["slicing"])
    if "film_grain_intensity" in job_input:
        wf["186"]["inputs"]["intensity"] = float(job_input["film_grain_intensity"])
    if "mask_blur" in job_input:
        wf["7"]["inputs"]["mask_blur"] = int(job_input["mask_blur"])
    if "tile_padding" in job_input:
        wf["7"]["inputs"]["tile_padding"] = int(job_input["tile_padding"])
    if "seam_fix_mode" in job_input:
        wf["7"]["inputs"]["seam_fix_mode"] = str(job_input["seam_fix_mode"])
    if "seam_fix_denoise" in job_input:
        wf["7"]["inputs"]["seam_fix_denoise"] = float(job_input["seam_fix_denoise"])
    if "seam_fix_width" in job_input:
        wf["7"]["inputs"]["seam_fix_width"] = int(job_input["seam_fix_width"])
    if "seam_fix_mask_blur" in job_input:
        wf["7"]["inputs"]["seam_fix_mask_blur"] = int(job_input["seam_fix_mask_blur"])
    if "seam_fix_padding" in job_input:
        wf["7"]["inputs"]["seam_fix_padding"] = int(job_input["seam_fix_padding"])
    if "lora_strength" in job_input:
        wf["179"]["inputs"]["lora_1"]["strength"] = float(job_input["lora_strength"])

    return wf


def handle_upscale(job_input):
    """Upscale mode: validate -> caption -> queue workflow -> return result.
    Includes OOM retry with reduced upscale factor."""
    image_b64 = job_input.get("image", "")
    if not image_b64:
        return {"error": "Missing 'image' in input"}

    # --- Step 1: Validate and resize ---
    image_b64, pil_image = validate_and_resize(image_b64)
    w, h = pil_image.size
    print(f"[handler] Input image: {w}x{h}")

    # --- Step 2: Caption with OpenAI (unless prompt provided) ---
    prompt = job_input.get("prompt", "")
    if not prompt:
        prompt = caption_image(image_b64)
    print(f"[handler] Prompt: {prompt[:80]}...")

    # --- Step 3: Upload image to ComfyUI ---
    image_name = job_input.get("image_name", f"input_{uuid.uuid4().hex[:8]}.png")
    uploaded_name = upload_image(image_name, image_b64)

    # --- Step 4: Run workflow with OOM retry ---
    upscale_by = job_input.get("upscale_by", 2)
    last_error = None

    for attempt in range(OOM_MAX_RETRIES + 1):
        current_upscale = upscale_by - (attempt * OOM_UPSCALE_REDUCTION)
        if current_upscale < 1:
            current_upscale = 1

        if attempt > 0:
            print(f"[handler] OOM retry {attempt}/{OOM_MAX_RETRIES}: "
                  f"reducing upscale {upscale_by} -> {current_upscale}")

        wf = build_workflow(job_input, upscale_override=current_upscale)
        wf["88"]["inputs"]["image"] = uploaded_name
        wf["180"]["inputs"]["text"] = prompt

        try:
            prompt_id, _ = queue_workflow(wf)
            history = wait_for_completion(prompt_id, timeout=1800)

            # --- Step 5: Fetch result ---
            result_bytes = fetch_image_from_history(history, "215")
            result_bytes = adaptive_compress(result_bytes)
            result_b64 = base64.b64encode(result_bytes).decode("utf-8")

            response = {
                "status": "success",
                "mode": "upscale",
                "upscale_by": current_upscale,
                "images": [{"node_id": "215", "image": result_b64, "type": "upscaled"}],
            }

            if attempt > 0:
                response["warning"] = (
                    f"OOM detected. Upscale reduced from {upscale_by}x to {current_upscale}x"
                )

            return response

        except RuntimeError as e:
            last_error = str(e)
            if is_oom_error(last_error) and attempt < OOM_MAX_RETRIES:
                print(f"[handler] OOM detected: {last_error[:100]}")
                continue
            raise

    return {"error": f"Failed after {OOM_MAX_RETRIES} OOM retries: {last_error}"}


# ============================================================
# Main Handler
# ============================================================

def handler(job):
    """RunPod serverless handler — main entry point."""
    print("[handler] === Job received ===")
    job_input = job.get("input", {})
    mode = job_input.get("mode", "upscale")
    print(f"[handler] Mode: {mode}, keys: {list(job_input.keys())}")

    try:
        if mode == "upscale":
            return handle_upscale(job_input)
        else:
            return {"error": f"Unknown mode: {mode}. Supported: upscale"}
    except Exception as e:
        print(f"[handler] Error: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
# Startup
# ============================================================

if __name__ == "__main__":
    # Run from /opt/ComfyUI directly (no copy to /workspace — saves cold start time)
    if os.path.exists("/opt/ComfyUI"):
        os.environ["COMFYUI_PATH"] = "/opt/ComfyUI"
        COMFYUI_PATH = "/opt/ComfyUI"

    start_comfyui()

    if not wait_for_comfyui():
        print("[handler] FATAL: ComfyUI failed to start")
        sys.exit(1)

    print("[handler] Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
