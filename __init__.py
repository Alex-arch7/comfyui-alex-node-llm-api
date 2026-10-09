"""comfyui-alex-node-llm-api

One unified ComfyUI node that merges:
  - comfyui-art-venture: "OpenAI API" + "LLM API Config" + "LLM Chat/Completion"
  - ComfyUI_qwen_chat_models: "qwen_VL_model" + "qwen_chat_model" input style
    (image / video / system / prompt)

Enter openai_api_key + endpoint, click "Refresh Models" on the node, and the
model dropdown is filled from the remote endpoint's /models list.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    from .api import register_routes

    register_routes()
except Exception as _exc:  # pragma: no cover
    print(f"[AlexLLM] failed to register HTTP routes: {_exc}")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
