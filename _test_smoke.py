#!/usr/bin/env python3
"""Smoke test for comfyui-alex-node-llm-api (no ComfyUI needed).

Spins up a mock OpenAI-compatible server, imports the node package the same
way ComfyUI's loader does (hyphenated dir -> spec_from_file_location), and
exercises: model refresh, cache persistence, multimodal message building,
chat call, error handling, preset prompts, custom_model override.
"""
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = "/home/alex/deepseek_harness/comfyui-alex-node-llm-api"
PORT = 18930
RECEIVED = {}


class MockAPI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            if "Bearer sk-test" not in self.headers.get("Authorization", ""):
                return self._send(401, {"error": {"message": "bad key"}})
            return self._send(200, {"object": "list", "data": [
                {"id": "zeta-vision"}, {"id": "alpha-chat"}, {"id": "gpt-4o"},
            ]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        if self.path == "/v1/chat/completions":
            if payload.get("model") == "fail-model":
                return self._send(400, {"error": {"message": "model not found"}})
            RECEIVED["payload"] = payload
            RECEIVED["auth"] = self.headers.get("Authorization", "")
            return self._send(200, {"choices": [
                {"message": {"role": "assistant", "content": "hello from mock"}}]})
        self._send(404, {"error": {"message": "not found"}})


def load_package():
    spec = importlib.util.spec_from_file_location(
        "alexllm", os.path.join(ROOT, "__init__.py"),
        submodule_search_locations=[ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alexllm"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import numpy as np
    import torch

    srv = HTTPServer(("127.0.0.1", PORT), MockAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    pkg = load_package()
    mc = sys.modules["alexllm.model_cache"]
    node = pkg.NODE_CLASS_MAPPINGS["AlexLLMNode"]()

    # 1. refresh models
    models = mc.fetch_models(f"http://127.0.0.1:{PORT}/v1", "sk-test")
    assert models == ["alpha-chat", "gpt-4o", "zeta-vision"], models
    assert os.path.isfile(mc.CACHE_FILE), "cache file not written"
    assert mc.get_model_options() == models
    print("PASS 1: refresh_models + cache persistence", models)

    # 2. INPUT_TYPES sees cached models
    it = node.INPUT_TYPES()
    assert it["required"]["model"][0] == models, it["required"]["model"][0]
    print("PASS 2: INPUT_TYPES dropdown uses cached models")

    # 3. multimodal run: image batch(2) + video(30 frames -> 8 sampled)
    img = torch.rand(2, 16, 16, 3)
    vid = torch.rand(30, 16, 16, 3)
    (out,) = node.run(
        openai_api_key="sk-test", endpoint=f"http://127.0.0.1:{PORT}/v1",
        model="zeta-vision", max_tokens=512, temperature=0.3,
        prompt="what is this?", system="You are terse.",
        preset_prompt="Short Description",
        image=img, video=vid, video_frame_count=8, timeout=10, seed=1)
    assert out == "hello from mock", out
    p = RECEIVED["payload"]
    assert RECEIVED["auth"] == "Bearer sk-test"
    assert p["model"] == "zeta-vision" and p["max_tokens"] == 512 and p["temperature"] == 0.3
    assert p["messages"][0]["role"] == "system" and p["messages"][0]["content"] == "You are terse."
    parts = p["messages"][1]["content"]
    n_img = sum(1 for x in parts if x["type"] == "image_url")
    assert n_img == 2 + 8, f"expected 10 image parts, got {n_img}"
    txt = parts[-1]["text"]
    assert "what is this?" in txt and "1-2 sentences" in txt, txt
    print(f"PASS 3: chat call ok — {n_img} image parts (2 image + 8 video frames), system+prompt merged")

    # 4. custom_model override + no media
    node.run(openai_api_key="sk-test", endpoint=f"http://127.0.0.1:{PORT}/v1",
             model="zeta-vision", max_tokens=10, temperature=0.0, prompt="hi",
             custom_model="alpha-chat")
    p = RECEIVED["payload"]
    assert p["model"] == "alpha-chat"
    assert len(p["messages"]) == 1 and len(p["messages"][0]["content"]) == 1
    print("PASS 4: custom_model override, text-only message")

    # 5. API error surfaces as exception
    try:
        node.run(openai_api_key="sk-test", endpoint=f"http://127.0.0.1:{PORT}/v1",
                 model="fail-model", max_tokens=10, temperature=0.0, prompt="x")
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "model not found" in str(e), e
    print("PASS 5: API error raised clearly")

    # 6. env fallback for key + refresh with bad key errors
    try:
        mc.fetch_models(f"http://127.0.0.1:{PORT}/v1", "wrong-key")
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "bad key" in str(e), e
    print("PASS 6: refresh error surfaced")

    # 7. video-only, single-frame edge cases
    node.run(openai_api_key="sk-test", endpoint=f"http://127.0.0.1:{PORT}/v1",
             model="zeta-vision", max_tokens=10, temperature=0.0, prompt="v",
             video=torch.rand(1, 8, 8, 3), video_frame_count=4)
    n_img = sum(1 for x in RECEIVED["payload"]["messages"][0]["content"] if x["type"] == "image_url")
    assert n_img == 1, n_img
    print("PASS 7: single-frame video edge case")

    srv.shutdown()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
