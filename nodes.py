"""Alex LLM API — one unified node.

Merges, in a single node:
  * comfyui-art-venture "OpenAI API"        (openai_api_key + endpoint)
  * comfyui-art-venture "LLM API Config"          (model + max_tokens + temperature)
  * comfyui-art-venture "LLM Chat / Completion"   (messages -> chat/completions)
  * ComfyUI_qwen_chat_models "qwen_VL_model"      (image + video-frames + preset prompt)
  * ComfyUI_qwen_chat_models "qwen_chat_model"    (system prompt + prompt)

The `model` dropdown is filled from the remote endpoint: enter api key +
endpoint, click the "Refresh Models" button on the node, and the combo is
updated live (and persisted server-side for future sessions).
"""

import base64
import io
import os
from typing import List, Optional

import numpy as np
import requests
import torch
from PIL import Image

from .model_cache import DEFAULT_ENDPOINT, get_model_options

# ------------------------------------------------------------
# Preset prompt templates (from qwen_VL_model)
# ------------------------------------------------------------
PRESET_PROMPTS = {
    "Original": "",
    "Detailed Description": "Please provide a detailed description of this image, including the main subject, setting, colors, lighting, composition, and any notable details. Be thorough and specific about what you see.",
    "Ultra Detailed Description": "Please provide an ultra-detailed description of this image. Cover all aspects: the main subject(s), background, foreground, colors, lighting conditions, shadows, textures, composition, style, mood, atmosphere, and every notable detail. Describe the scene as if painting a picture with words. Include spatial relationships between objects, the time of day suggested, and any subtle details that might be easily missed.",
    "Short Description": "Briefly describe what you see in this image in 1-2 sentences.",
    "Poetic Description": "Describe this image in a poetic, artistic manner. Use vivid imagery, metaphors, and evocative language to capture the mood and essence of the scene.",
    "Technical Analysis": "Analyze this image from a technical perspective: composition (rule of thirds, leading lines, etc.), lighting (direction, quality, color temperature), color palette, depth of field, focus, and overall photographic or artistic technique.",
    "Storytelling": "Tell a story about this image. What is happening? What led up to this moment? What might happen next? Give life to the scene with narrative details about the subject's emotions and surroundings.",
    "Object Detection": "List and describe each distinct object or element visible in this image. Identify the primary subject, secondary elements, and any background objects. Include approximate positions and relative sizes.",
}


# ------------------------------------------------------------
# Tensor / image helpers (adapted from ComfyUI_qwen_chat_models)
# ------------------------------------------------------------
def tensor_to_pil(image_tensor: torch.Tensor, batch_index: int = 0) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor frame to a PIL RGB image.

    Handles [0,1] and [0,255] float scales, grayscale and RGBA inputs.
    """
    img = image_tensor[batch_index].unsqueeze(0)
    if float(img.max()) > 1.5:
        img = img / 255.0
    arr = np.clip(img.cpu().numpy() * 255.0, 0, 255).astype(np.uint8).squeeze()

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            return Image.fromarray(arr[:, :, :3], mode="RGB")
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unexpected image tensor shape: {arr.shape}")


def pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as base64 PNG (for OpenAI image_url parts)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frames_from_video(video: torch.Tensor, frame_count: int = 8) -> List[Image.Image]:
    """Sample `frame_count` frames uniformly from a ComfyUI video IMAGE
    tensor [N, H, W, 3] and return them as PIL images (audio ignored).
    """
    if video is None:
        return []
    if video.ndim == 4:
        n = video.shape[0]
    elif video.ndim == 3:
        n = 1
    else:
        return []

    frame_count = max(1, int(frame_count))
    if n <= 1:
        indices = [0]
    elif n <= frame_count:
        indices = list(range(n))
    else:
        # Uniform sampling across the whole clip, not just the head.
        indices = np.round(np.linspace(0, n - 1, frame_count)).astype(int).tolist()

    pil_frames = []
    for i in indices:
        frame = video[i]
        if frame.ndim == 4:
            frame = frame[0]
        arr = np.clip(frame.cpu().numpy() * 255.0, 0, 255).astype(np.uint8).squeeze()
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        pil_frames.append(Image.fromarray(arr, mode="RGB"))
    return pil_frames


def image_tensor_to_base64_list(tensor: torch.Tensor) -> List[str]:
    """Convert a ComfyUI IMAGE input (single or batch) to base64 PNGs."""
    out: List[str] = []
    if tensor is None:
        return out
    if tensor.ndim == 4:
        for i in range(tensor.shape[0]):
            out.append(pil_to_base64(tensor_to_pil(tensor, i)))
    elif tensor.ndim == 3:
        out.append(pil_to_base64(tensor_to_pil(tensor, 0)))
    return out


# ------------------------------------------------------------
# The unified node
# ------------------------------------------------------------
class AlexLLMNode:
    """One node = API credentials + model config + chat/complete call.

    Inputs: openai_api_key, endpoint, model (refreshable dropdown),
    max_tokens, temperature, prompt; optional system / preset_prompt /
    custom_model / image / video / video_frame_count / timeout / seed.
    Output: STRING text response.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_options()
        return {
            "required": {
                "openai_api_key": (
                    "STRING",
                    {"multiline": False, "default": "", "password": True},
                ),
                "endpoint": (
                    "STRING",
                    {"multiline": False, "default": DEFAULT_ENDPOINT},
                ),
                "model": (models, {"default": models[0]}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 200000}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": False, "default": ""},
                ),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": ""}),
                "preset_prompt": (list(PRESET_PROMPTS.keys()), {"default": "Original"}),
                "custom_model": ("STRING", {"multiline": False, "default": ""}),
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "video_frame_count": ("INT", {"default": 8, "min": 1, "max": 64}),
                "timeout": ("INT", {"default": 120, "min": 5, "max": 900}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Alex/LLM"
    DESCRIPTION = (
        "Unified OpenAI-compatible LLM node: API key + endpoint + model "
        "(click 'Refresh Models' to pull the list from the endpoint) + "
        "max_tokens/temperature + system/prompt + image + video. "
        "Merges art-venture OpenAI API / LLM API Config / LLM Chat "
        "and the qwen_VL_model + qwen_chat_model input styles into one node."
    )

    def run(
        self,
        openai_api_key: str,
        endpoint: str,
        model: str,
        max_tokens: int,
        temperature: float,
        prompt: str,
        system: str = "",
        preset_prompt: str = "Original",
        custom_model: str = "",
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        video_frame_count: int = 8,
        timeout: int = 120,
        seed: int = 0,
    ):
        # --- credentials (env fallback, like art-venture) ---
        api_key = (openai_api_key or "").strip()
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "[AlexLLM] openai_api_key is empty. Enter a key or set OPENAI_API_KEY."
            )

        ep = (endpoint or "").strip().rstrip("/") or DEFAULT_ENDPOINT

        # custom_model overrides the dropdown when non-empty
        final_model = (custom_model or "").strip() or model
        if not final_model:
            raise ValueError("[AlexLLM] No model selected. Pick one or use custom_model.")

        # --- build user text (preset prompt + user prompt) ---
        text = prompt or ""
        preset_text = PRESET_PROMPTS.get(preset_prompt, "")
        if preset_text:
            text = f"{preset_text}\n\n{text}" if text.strip() else preset_text

        # --- build multimodal content parts: images, then video frames, then text ---
        content_parts: List[dict] = []

        if image is not None:
            for b64 in image_tensor_to_base64_list(image):
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        if video is not None:
            frames = frames_from_video(video, video_frame_count)
            print(
                f"[AlexLLM] (video) sending {len(frames)} sampled frame(s) as images "
                f"(audio ignored). Vision models may need a large context window."
            )
            for frm in frames:
                b64 = pil_to_base64(frm)
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        content_parts.append({"type": "text", "text": text})

        messages: List[dict] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": content_parts})

        # --- call chat/completions ---
        url = f"{ep}/chat/completions"
        payload = {
            "model": final_model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=int(timeout))
        except requests.RequestException as exc:
            raise RuntimeError(f"[AlexLLM] request to {url} failed: {exc}")

        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(
                f"[AlexLLM] HTTP {resp.status_code} from {url}, non-JSON response: "
                f"{resp.text[:500]}"
            )

        if resp.status_code >= 400 or data.get("error"):
            err = data.get("error")
            if isinstance(err, dict):
                err = err.get("message") or str(err)
            raise RuntimeError(
                f"[AlexLLM] API error (HTTP {resp.status_code}): {err or resp.text[:500]}"
            )

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"[AlexLLM] No choices in response: {str(data)[:500]}")

        message = choices[0].get("message") or {}
        out_text = message.get("content")
        if out_text is None:
            out_text = ""
        return (out_text,)


NODE_CLASS_MAPPINGS = {
    "AlexLLMNode": AlexLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlexLLMNode": "Alex LLM API (Unified)",
}
