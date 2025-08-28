from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import httpx
from typing import Optional, Dict, Any, List, Tuple
import time
import base64
import os
import logging
from urllib.parse import quote
import json
import asyncio
from collections import deque

# Basic settings (can be extended to env vars)
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]
UPSTREAM_TEXT_BASE = os.getenv("UPSTREAM_TEXT_BASE", "https://text.pollinations.ai")
UPSTREAM_IMAGE_BASE = os.getenv("UPSTREAM_IMAGE_BASE", "https://image.pollinations.ai")
UPSTREAM_AUDIO_BASE = os.getenv("UPSTREAM_AUDIO_BASE", "https://audio.pollinations.ai")
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", os.getenv("API_KEY", "")).split(",") if k.strip()]
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))  # per IP

app = FastAPI(title="Pollinations OpenAI-Compatible Proxy", version="0.1.0", default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pollinations-proxy")

# Rate limiter: per-IP sliding window using deque
_rate_bucket: Dict[str, deque] = {}

# Shared HTTPX client with HTTP/2, connection pooling, retries
_client: Optional[httpx.AsyncClient] = None

# Lightweight cache for models endpoint
_models_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
MODELS_CACHE_TTL = float(os.getenv("MODELS_CACHE_TTL", "30"))

def _check_auth(authorization: Optional[str]):
    if not API_KEYS:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

def _rate_limit(request: Request):
    if RATE_LIMIT_RPM <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0
    dq = _rate_bucket.setdefault(ip, deque())
    cutoff = now - window
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    dq.append(now)


async def fetch_json(client: httpx.AsyncClient, url: str) -> Any:
    r = await client.get(url)
    r.raise_for_status()
    return r.json()


def now_unix() -> int:
    return int(time.time())


@app.on_event("startup")
async def on_startup():
    """Vercel may cold-start; init client on each boot."""
    global _client
    if _client is None:
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        transport = httpx.AsyncHTTPTransport(retries=2, http2=True)
        timeout = httpx.Timeout(120.0, connect=10.0, read=120.0, write=30.0, pool=10.0)
        _client = httpx.AsyncClient(limits=limits, transport=transport, timeout=timeout, headers={
            "User-Agent": "pollinations-proxy/0.1",
            "Accept": "*/*",
        })
        logger.info("Initialized shared HTTPX client with HTTP/2 and pooling")


@app.on_event("shutdown")
async def on_shutdown():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Closed shared HTTPX client")


@app.get("/v1/models")
async def list_models(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    """Aggregate models from text and image endpoints into OpenAI-like list."""
    # Cache hit
    cache_key = "models_all"
    now = time.time()
    cached = _models_cache.get(cache_key)
    if cached and (now - cached[0]) < MODELS_CACHE_TTL:
        return {"object": "list", "data": cached[1]}

    assert _client is not None
    models_text: List[Dict[str, Any]] = []
    models_image: List[Dict[str, Any]] = []
    async def fetch_text():
        nonlocal models_text
        try:
            t = await fetch_json(_client, f"{UPSTREAM_TEXT_BASE}/models")
            if isinstance(t, list):
                models_text = [{"id": m, "object": "model"} if isinstance(m, str) else m for m in t]
        except Exception:
            pass

    async def fetch_image():
        nonlocal models_image
        try:
            i = await fetch_json(_client, f"{UPSTREAM_IMAGE_BASE}/models")
            if isinstance(i, list):
                models_image = [{"id": m, "object": "model"} if isinstance(m, str) else m for m in i]
        except Exception:
            pass

    await asyncio.gather(fetch_text(), fetch_image())

    # Merge and de-dup by id
    seen = set()
    merged: List[Dict[str, Any]] = []
    for m in models_text + models_image:
        mid = m.get("id") if isinstance(m, dict) else str(m)
        if mid in seen:
            continue
        seen.add(mid)
        merged.append({"id": mid, "object": "model"})

    result = {"object": "list", "data": merged}
    _models_cache[cache_key] = (now, merged)
    return result


@app.get("/v1/models/text")
async def list_text_models(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    assert _client is not None
    try:
        t = await fetch_json(_client, f"{UPSTREAM_TEXT_BASE}/models")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    items = (
        [{"id": m, "object": "model"} for m in t]
        if isinstance(t, list) else []
    )
    return {"object": "list", "data": items}


@app.get("/v1/models/image")
async def list_image_models(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    assert _client is not None
    try:
        i = await fetch_json(_client, f"{UPSTREAM_IMAGE_BASE}/models")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    items = (
        [{"id": m, "object": "model"} for m in i]
        if isinstance(i, list) else []
    )
    return {"object": "list", "data": items}


@app.get("/v1/models/tts")
async def list_tts_voices(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    """Expose a list of available TTS voices. Pollinations voice catalog isn't documented;
    return a reasonable default set and allow override later.
    """
    # Common voices used in examples; adjust when upstream catalog is discoverable
    voices = [
        "alloy", "verse", "aria", "amber", "breeze", "flow", "nova", "sage"
    ]
    items = [{"id": v, "object": "voice"} for v in voices]
    return {"object": "list", "data": items}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: Dict[str, Any], authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    """Non-streaming chat completions. If upstream supports OpenAI-compatible POST at /openai,
    forward the body. Otherwise, build a prompt from messages and call GET /{prompt}.
    """
    model = body.get("model", "openai")
    stream = body.get("stream", False)

    # Helper to construct prompt from messages
    def build_prompt(messages: List[Dict[str, Any]]) -> str:
        return "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

    assert _client is not None
    # Try OpenAI-compatible endpoint first
    try:
        r = await _client.post(
            f"{UPSTREAM_TEXT_BASE}/openai", json=body
        )
        r.raise_for_status()
        data = r.json()
        # If upstream already returns OpenAI-like shape, pass through
        if isinstance(data, dict) and ("choices" in data or stream):
            if not stream:
                return data
            # If upstream accepted stream, pass-through is tricky; fall back to emulation unless upstream truly streams.
    except Exception:
        pass

    # Fallback: construct prompt from messages
    messages = body.get("messages", [])
    prompt = build_prompt(messages)
    # GET-based simple text completion
    try:
        r2 = await _client.get(f"{UPSTREAM_TEXT_BASE}/{quote(prompt)}")
        r2.raise_for_status()
        text = r2.text
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    if stream:
        # Emulate SSE streaming: send small chunks as delta events
        async def event_gen():
            chunk_id = f"chatcmpl-{int(time.time()*1000)}"
            created = now_unix()
            role_sent = False
            # Send role first as per OpenAI delta convention
            first_event = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(first_event)}\n\n"
            role_sent = True

            # Naive chunking by words
            words = text.split()
            buf = []
            for w in words:
                buf.append(w)
                if len(buf) >= 20:  # send 20-word chunks
                    delta = " ".join(buf) + " "
                    event = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                    buf = []
                    await asyncio.sleep(0)
            # Flush remainder
            if buf:
                delta = " ".join(buf)
                event = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(event)}\n\n"

            # Final done event
            done_event = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # Wrap in OpenAI-like response (non-stream)
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": now_unix(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    }


@app.post("/v1/completions")
async def completions(request: Request, body: Dict[str, Any], authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    prompt = body.get("prompt", "")
    model = body.get("model", "openai")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    assert _client is not None
    try:
        r = await _client.get(f"{UPSTREAM_TEXT_BASE}/{quote(prompt)}")
        r.raise_for_status()
        text = r.text
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
    return {
        "id": f"cmpl-{int(time.time()*1000)}",
        "object": "text_completion",
        "created": now_unix(),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    }


@app.post("/v1/images/generations")
async def image_generations(request: Request, body: Dict[str, Any], authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    prompt = body.get("prompt") or (body.get("input") if isinstance(body.get("input"), str) else None)
    size = body.get("size", "1024x1024")
    model = body.get("model")  # optional
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Build image URL; Pollinations image GET returns the image bytes at prompt URL
    url = f"{UPSTREAM_IMAGE_BASE}/prompt/{quote(prompt)}"
    if model:
        url += f"?model={model}"

    # Option: return as URL or b64 image; we'll return b64_json for OpenAI compatibility
    assert _client is not None
    try:
        r = await _client.get(url)
        r.raise_for_status()
        img_bytes = r.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return {
        "created": now_unix(),
        "data": [{"b64_json": b64}]
    }


@app.post("/v1/audio/speech")
async def audio_speech(request: Request, body: Dict[str, Any], authorization: Optional[str] = Header(None)) -> Response:
    _rate_limit(request)
    _check_auth(authorization)
    """Text-to-speech. Map to GET /{prompt}?model=openai-audio&voice={voice}"""
    input_text = body.get("input") or body.get("text") or body.get("prompt")
    voice = body.get("voice", "alloy")
    model = body.get("model", "openai-audio")
    format_ = body.get("format", "mp3")
    if not input_text:
        raise HTTPException(status_code=400, detail="input (text) is required")

    url = f"{UPSTREAM_TEXT_BASE}/{quote(input_text)}?model={model}&voice={voice}"

    assert _client is not None
    mime = "audio/mpeg" if format_ == "mp3" else "audio/wav"

    async def audio_stream():
        try:
            async with _client.stream("GET", url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    return StreamingResponse(audio_stream(), media_type=mime)


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    _rate_limit(request)
    _check_auth(authorization)
    """Stub: If Pollinations exposes STT via POST /openai we can implement later."""
    raise HTTPException(status_code=501, detail="Speech-to-text is not supported yet by upstream")


@app.get("/")
async def root():
    return {"status": "ok", "name": "pollinations-openai-proxy"}
