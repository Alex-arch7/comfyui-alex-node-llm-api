"""ComfyUI web routes for the Alex LLM API node.

POST /alex_llm/refresh_models  {endpoint, api_key}  -> {ok, models, endpoint}
GET  /alex_llm/models_cache                          -> {ok, cache}

The node's frontend button calls these; the backend cache is what
INPUT_TYPES serves on the next /object_info (page reload).
"""

import asyncio
import os

from aiohttp import web

from .model_cache import fetch_models, get_cache_snapshot, normalize_endpoint


async def refresh_models(request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    endpoint = body.get("endpoint") or os.environ.get("OPENAI_API_BASE", "")
    api_key = body.get("api_key") or ""

    loop = asyncio.get_event_loop()
    try:
        models = await loop.run_in_executor(None, fetch_models, endpoint, api_key)
        return web.json_response(
            {"ok": True, "models": models, "endpoint": normalize_endpoint(endpoint)}
        )
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def models_cache(request):
    return web.json_response({"ok": True, "cache": get_cache_snapshot()})


def register_routes():
    try:
        from server import PromptServer
    except ImportError:
        print("[AlexLLM] server module not found; HTTP routes not registered.")
        return

    routes = PromptServer.instance.routes
    routes.post("/alex_llm/refresh_models")(refresh_models)
    routes.get("/alex_llm/models_cache")(models_cache)
    print("[AlexLLM] routes registered: POST /alex_llm/refresh_models, GET /alex_llm/models_cache")
