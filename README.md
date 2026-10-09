# comfyui-alex-node-llm-api

**Alex LLM API (Unified)** — 一个统一的 ComfyUI LLM 节点，把以下功能合并为一个节点：

| 来源项目 | 原节点 | 合并后 |
|---|---|---|
| comfyui-art-venture | `OpenAI API`（api_key + endpoint） | ✅ 同一节点 |
| comfyui-art-venture | `LLM API Config`（model / max_token / temperature） | ✅ 同一节点 |
| comfyui-art-venture | `LLM Chat` / `LLM Completion` | ✅ 同一节点（chat/completions） |
| ComfyUI_qwen_chat_models | `qwen_VL_model`（image + video + preset prompt） | ✅ 同一节点（API 版） |
| ComfyUI_qwen_chat_models | `qwen_chat_model`（system + prompt） | ✅ 同一节点（API 版） |

## 核心特性

- **手动刷新模型列表**：填入 `openai_api_key` 和 `endpoint` 后，点击节点上的 **🔄 Refresh Models** 按钮，节点会请求远程 `{endpoint}/models`，把返回的模型列表实时注入 `model` 下拉框（服务端缓存到 `cache/models.json`，重启 ComfyUI 后依然可用）。
- **max_tokens / temperature** 直接设置在节点上。
- **多模态输入**：支持 `image`（ComfyUI IMAGE，含批量）与 `video`（Load Video 输出的 IMAGE 张量，均匀抽帧后逐帧作为图片发送，忽略音频），以及 `prompt` / `system` / `preset_prompt`（沿用 qwen_VL_model 的预设描述模板）。
- **custom_model**：非空时覆盖下拉框，可手动输入任意模型名。
- 兼容任何 OpenAI 风格 endpoint（OpenAI、OpenRouter、DeepSeek、Qwen/DashScope 兼容模式、vLLM、Ollama `/v1`、本地网关等）。

## 安装

```
ComfyUI/custom_nodes/
└── comfyui-alex-node-llm-api/
```

```bash
cd ComfyUI/custom_nodes/comfyui-alex-node-llm-api
pip install -r requirements.txt   # requests pillow numpy
```

重启 ComfyUI。

## 使用

1. 添加节点：**Alex/LLM → Alex LLM API (Unified)**。
2. 填 `openai_api_key`（留空则回退到环境变量 `OPENAI_API_KEY`）和 `endpoint`（默认 `https://api.openai.com/v1`）。
3. 点击 **🔄 Refresh Models** → 按钮显示模型数量，`model` 下拉框即被刷新。
4. 设置 `max_tokens`、`temperature`，接入 `image` / `video`，填写 `prompt`（可选 `system`、`preset_prompt`），运行。
5. 输出为 `STRING`（模型回复文本）。

> 示例工作流：`examples/alex_llm_api_workflow.json`（LoadImage → AlexLLMNode）。

## 输入 / 输出

| 输入 | 类型 | 说明 |
|---|---|---|
| openai_api_key | STRING | 留空回退 `OPENAI_API_KEY` 环境变量 |
| endpoint | STRING | OpenAI 兼容根路径（不含 `/chat/completions`） |
| model | COMBO | 由 Refresh Models 刷新；未刷新时用内置默认列表 |
| max_tokens | INT | 1 – 200000 |
| temperature | FLOAT | 0 – 2 |
| prompt | STRING(多行) | 用户提示词 |
| system | STRING(多行, 可选) | 系统提示词（qwen_chat_model 风格） |
| preset_prompt | COMBO(可选) | qwen_VL_model 的预设描述模板 |
| custom_model | STRING(可选) | 非空时覆盖 model 下拉框 |
| image | IMAGE(可选) | 单图或批量图，全部作为 image_url 发送 |
| video | IMAGE(可选) | 视频张量 [N,H,W,3]，均匀抽帧发送 |
| video_frame_count | INT(可选) | 抽帧数 1–64，默认 8 |
| timeout | INT(可选) | HTTP 超时秒数 |
| seed | INT(可选) | 用于强制重新执行（缓存失效） |

| 输出 | 类型 |
|---|---|
| text | STRING |

## 注意事项

- 模型列表刷新走本地 ComfyUI 服务端的 `POST /alex_llm/refresh_models`（api_key 只在本机传输）。若 endpoint 不支持 `GET /models`，按钮会显示错误信息。
- 视频按“图片序列”发送（与 qwen_VL_model 的做法一致），视觉模型需要足够大的上下文窗口。
- API key 会保存在工作流 JSON 中，分享工作流前请注意。

## 文件结构

```
comfyui-alex-node-llm-api/
├── __init__.py          # 注册节点 + WEB_DIRECTORY + HTTP 路由
├── nodes.py             # AlexLLMNode 统一节点（含 image/video 处理）
├── model_cache.py       # /models 抓取 + 按 endpoint 缓存到 cache/models.json
├── api.py               # aiohttp 路由: refresh_models / models_cache
├── web/js/alex_llm.js   # 前端: Refresh Models 按钮 + 下拉框实时刷新
├── examples/            # 示例工作流
├── pyproject.toml / requirements.txt / LICENSE
```
