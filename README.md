# Pollinations OpenAI-Compatible Proxy (FastAPI)

A thin FastAPI layer that exposes OpenAI-compatible endpoints and proxies to Pollinations services (https://pollinations.ai/) for text, image, and audio (TTS). Works with existing OpenAI SDKs/clients.

## Features
- /v1/models aggregated + domain-specific: /v1/models/text, /v1/models/image, /v1/models/tts (voices)
- /v1/chat/completions and /v1/completions (non-streaming)
- /v1/images/generations (returns b64_json)   -   Untested
- /v1/audio/speech (TTS) with voices          -   Untested
- Optional Bearer auth and per-IP rate limiting
- CORS enabled (configurable)

## Upstream services
- Text: https://text.pollinations.ai
- Image: https://image.pollinations.ai
- Audio: https://audio.pollinations.ai

## Configuration
Environment variables (see .env.example):
- ALLOW_ORIGINS: CORS allowlist (comma-separated). Default: "*"
- API_KEYS: Comma-separated list of acceptable API keys. If set, Authorization: Bearer <key> is required.
- RATE_LIMIT_RPM: Requests per minute per IP. Default: 60. Set to 0 to disable.
- UPSTREAM_TEXT_BASE, UPSTREAM_IMAGE_BASE, UPSTREAM_AUDIO_BASE: Override upstream URLs.
- LOG_LEVEL: INFO|DEBUG|WARNING. Default: INFO.

## LIMITATIONS:
Pollinations Tier based API compatibility not added yet and certain models are locked behind that. <br>
The model lists are found here:
- Text Generation models: https://text.pollinations.ai/models
- Image Generation models: https://image.pollinations.ai/models <br>

Or you can refer to this for model list (from https://pollinations.ai/llms.txt):
```
Model: openai - OpenAI GPT-4.1-nano (supports text and image inputs)
Model: openai-large - OpenAI GPT-4.1 mini (supports text and image inputs)
Model: openai-reasoning - OpenAI o4-mini (reasoning-focused model)
Model: qwen-coder - Qwen 2.5 Coder 32B (code-focused model)
Model: llama - Llama 3.3 70B
Model: llamascout - Llama 4 Scout 17B
Model: mistral - Mistral Small 3 (supports text and image inputs)
Model: unity - Unity Mistral Large (uncensored, supports text and image inputs)
Model: midijourney - Midijourney (music generation focused)
Model: rtist - Rtist (creative writing focused)
Model: searchgpt - SearchGPT (search-augmented model)
Model: evil - Evil (uncensored model with text and image inputs)
Model: deepseek-reasoning - DeepSeek-R1 Distill Qwen 32B (reasoning-focused)
Model: deepseek-reasoning-large - DeepSeek R1 - Llama 70B (reasoning-focused)
Model: phi - Phi-4 Instruct (supports text, image, and audio inputs)
Model: llama-vision - Llama 3.2 11B Vision (supports text and image inputs)
Model: gemini - Gemini 2.5 Flash Preview (supports text, image, and audio inputs/outputs)
Model: hormoz - Hormoz 8b
Model: hypnosis-tracy - Hypnosis Tracy 7B (supports text and audio inputs/outputs)
Model: deepseek - DeepSeek-V3
Model: sur - Sur AI Assistant (Mistral-based, supports text and image inputs)
Model: openai-audio - OpenAI GPT-4o-audio-preview (supports text, image, and audio inputs/outputs)
Model: flux - Latest stable diffusion model
Model: turbo - Fast image generation model
```


## Local development

Create a virtualenv and install deps:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Optional: set API key and CORS

```bash
export API_KEYS=devkey123
export ALLOW_ORIGINS=http://localhost:3000
```

### Quick test

- Root health:
```bash
curl -s http://localhost:8000/
```

- Models:
```bash
curl -s http://localhost:8000/v1/models | jq
```

- Chat (non-stream):
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Say hello in one sentence"}
    ]
  }' | jq
```

- Image generation (b64):
```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a cute corgi in a spacesuit, cartoon style"}' | jq -r '.data[0].b64_json' | base64 -d > out.png
```

- TTS:
```bash
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello from Pollinations proxy", "voice": "alloy"}' > speech.mp3
```

## OpenAI SDK usage examples

Python (openai>=1.0):
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="devkey123")

# Chat
resp = client.chat.completions.create(
    model="openai",
    messages=[
        {"role": "user", "content": "Write a haiku about the sea"}
    ],
)
print(resp.choices[0].message.content)

# Image
img = client.images.generate(prompt="a watercolor painting of mountains at dawn")
print(len(img.data[0].b64_json))

# TTS (SDK may differ between versions; here we call REST)
```

Node (openai@4):
```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "devkey123" });

const chat = await client.chat.completions.create({
  model: "openai",
  messages: [{ role: "user", content: "Tell a joke about robots" }],
});
console.log(chat.choices[0].message.content);
```

## Deployment: Vercel (Docker)
This repo includes `Dockerfile` and `vercel.json` to deploy as a container (recommended for FastAPI and potential streaming support).

Steps:
1. `vercel login`
2. `vercel project add` (or use dashboard)
3. `vercel deploy --prod`

Vercel will build the Docker image and run `uvicorn`. Configure environment variables in Vercel dashboard.

## Notes and limitations
- Streaming (`stream=true`) for chat/completions is not implemented yet. We can add SSE if upstream supports it or emulate token chunking.
- STT (`/v1/audio/transcriptions`) is not implemented because upstream path is unclear; will add when available.
- TTS voices: `/v1/models/tts` returns a curated list until upstream catalog discovery is documented.

## Project structure
```
app/
  main.py
Dockerfile
requirements.txt
vercel.json
README.md
```
