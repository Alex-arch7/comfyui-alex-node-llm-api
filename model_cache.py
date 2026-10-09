"""Model-list cache for the Alex LLM API node.

Fetches the model list from an OpenAI-compatible endpoint (GET {endpoint}/models)
and caches it per-endpoint on disk, so the node's `model` dropdown survives
ComfyUI restarts and is available to INPUT_TYPES at graph-load time.
"""

import json
import os
import threading
from typing import Dict, List

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "models.json")

DEFAULT_ENDPOINT = "https://api.openai.com/v1"

# Shown in the dropdown before the user ever clicks "Refresh Models".
DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3-mini",
    "o4-mini",
    "deepseek-chat",
    "deepseek-reasoner",
    "qwen-vl-max",
    "qwen-vl-plus",
]

_lock = threading.Lock()
# endpoint (normalized) -> sorted list of model ids
_cache: Dict[str, List[str]] = {}


def normalize_endpoint(endpoint: str) -> str:
    ep = (endpoint or "").strip().rstrip("/")
    return ep or DEFAULT_ENDPOINT


def _load() -> None:
    global _cache
    try:
        if os.path.isfile(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _cache = {
                    k: [str(m) for m in v]
                    for k, v in data.items()
                    if isinstance(v, list)
                }
    except Exception as exc:  # pragma: no cover
        print(f"[AlexLLM] could not load model cache: {exc}")


def _save() -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CACHE_FILE)
    except Exception as exc:  # pragma: no cover
        print(f"[AlexLLM] could not save model cache: {exc}")


def _parse_models_payload(payload) -> List[str]:
    """Accept the common /models response shapes:
    {"data":[{"id":...}]}, {"models":[...]}, plain list of ids or objects."""
    models: List[str] = []
    if isinstance(payload, dict):
        items = payload.get("data")
        if items is None:
            items = payload.get("models")
        if items is None and payload.get("object") == "list":
            items = []
    elif isinstance(payload, list):
        items = payload
    else:
        items = None

    if not isinstance(items, list):
        return []

    for item in items:
        if isinstance(item, str) and item.strip():
            models.append(item.strip())
        elif isinstance(item, dict):
            mid = item.get("id") or item.get("name")
            if isinstance(mid, str) and mid.strip():
                models.append(mid.strip())

    # Deduplicate, keep sorted
    return sorted(dict.fromkeys(models))


def fetch_models(endpoint: str, api_key: str, timeout: int = 30) -> List[str]:
    """GET {endpoint}/models and cache the result. Raises on failure."""
    ep = normalize_endpoint(endpoint)
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("API key is required to list models (or set OPENAI_API_KEY).")

    url = f"{ep}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=timeout)

    try:
        payload = resp.json()
    except Exception:
        raise RuntimeError(
            f"GET {url} -> HTTP {resp.status_code}, non-JSON response: "
            f"{resp.text[:300]}"
        )

    if resp.status_code >= 400 or (isinstance(payload, dict) and payload.get("error")):
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            err = err.get("message") or json.dumps(err, ensure_ascii=False)
        raise RuntimeError(
            f"GET {url} -> HTTP {resp.status_code}: {err or resp.text[:300]}"
        )

    models = _parse_models_payload(payload)
    if not models:
        raise RuntimeError(
            f"GET {url} returned no models (endpoint may not support /models)."
        )

    with _lock:
        _cache[ep] = models
        _save()
    print(f"[AlexLLM] refreshed {len(models)} model(s) from {url}")
    return models


def get_cached_models() -> List[str]:
    """Union of all cached model ids (sorted). Empty if nothing cached."""
    with _lock:
        merged: List[str] = []
        for models in _cache.values():
            merged.extend(models)
    return sorted(dict.fromkeys(merged))


def get_model_options() -> List[str]:
    """Options for the node dropdown: cached models, else defaults."""
    models = get_cached_models()
    return models if models else list(DEFAULT_MODELS)


def get_cache_snapshot() -> Dict[str, List[str]]:
    with _lock:
        return {k: list(v) for k, v in _cache.items()}


# Load persisted cache at import time so INPUT_TYPES sees it immediately.
_load()
