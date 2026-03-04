# Streaming TTS API Documentation

## Overview

The `/tts/stream` endpoint provides **token-level streaming** text-to-speech synthesis. Instead of waiting for the entire audio to be generated before responding, it streams audio chunks as speech tokens are produced by the model. This dramatically reduces time-to-first-audio compared to batch synthesis.

**Typical latency to first audio chunk: ~0.9s** (warm calls)

---

## Endpoint

```
POST /tts/stream
Content-Type: application/json
```

---

## Request Body

```json
{
  "text": "Your text to synthesize",
  "voice_mode": "predefined",
  "predefined_voice_id": "Dina.wav",
  "output_format": "mulaw",
  "chunk_size": 25,
  "context_window": 50,
  "temperature": 0.8,
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "seed": 0,
  "speed_factor": 1.0
}
```

### Parameters

#### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | Text to synthesize. Supports English and Arabic (with fine-tuned model). |

#### Voice Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice_mode` | `"predefined"` \| `"clone"` | `"predefined"` | Use a built-in voice or clone from reference audio. |
| `predefined_voice_id` | string | `null` | Filename of a predefined voice (e.g. `"Dina.wav"`, `"Adrian.wav"`). Required when `voice_mode` is `"predefined"`. |
| `reference_audio_filename` | string | `null` | Filename of an uploaded reference audio for voice cloning. Required when `voice_mode` is `"clone"`. Upload via the `/upload-reference` endpoint first. |

#### Output Format

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | `"wav"` \| `"mulaw"` | `"wav"` | Audio encoding format (see [Output Formats](#output-formats) below). |

#### Streaming Control

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `chunk_size` | int | `25` | 1-100 | Number of speech tokens to generate before decoding and yielding an audio chunk. Lower values reduce latency but increase decode overhead. |
| `context_window` | int | `50` | 0-200 | Number of previously generated speech tokens to include as context when decoding each new chunk. Higher values improve audio continuity at chunk boundaries but increase decode time. |

#### Generation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `temperature` | float | server config | 0.0-1.5 | Controls randomness in token sampling. Lower = more deterministic. |
| `exaggeration` | float | server config | 0.25-2.0 | Controls expressiveness and emotion intensity. |
| `cfg_weight` | float | server config | 0.2-1.0 | Classifier-free guidance weight. Higher = stronger adherence to text/style. |
| `seed` | int | server config | >= 0 | Random seed for reproducibility. `0` uses random generation. |
| `speed_factor` | float | `1.0` | 0.25-4.0 | Post-generation speed adjustment. Applied per chunk. |

---

## Output Formats

### WAV (`output_format: "wav"`)

- **Content-Type:** `audio/wav`
- **Sample rate:** 24,000 Hz
- **Channels:** 1 (mono)
- **Bit depth:** 16-bit signed integer (PCM16)
- **Structure:** First 44 bytes are a standard WAV header (with max size placeholder for streaming), followed by raw PCM16 data chunks.

### MULAW (`output_format: "mulaw"`)

- **Content-Type:** `audio/x-mulaw`
- **Sample rate:** 8,000 Hz
- **Channels:** 1 (mono)
- **Bit depth:** 8-bit mu-law compressed
- **Structure:** Raw mu-law bytes with no header. This is the native format for telephony systems (Twilio, FreeSWITCH, Asterisk).

---

## Response

The response is a **streaming HTTP response** with `Transfer-Encoding: chunked`. Audio data arrives incrementally as the model generates speech tokens.

Each HTTP chunk contains a segment of encoded audio (PCM16 or mu-law bytes). The stream ends when the model finishes generating all speech tokens for the input text.

**There is no JSON envelope** — the response body is raw audio bytes.

---

## Integration Examples

### cURL

Save streamed audio to a file:

```bash
curl -X POST http://localhost:8004/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a streaming test.",
    "voice_mode": "predefined",
    "predefined_voice_id": "Dina.wav",
    "output_format": "mulaw",
    "chunk_size": 25
  }' \
  --output output.ulaw
```

### Python — Stream and Save

```python
import requests

response = requests.post(
    "http://localhost:8004/tts/stream",
    json={
        "text": "مرحبا يا أصدقائي الأعزاء، كيف حالكم النهاردة؟",
        "voice_mode": "predefined",
        "predefined_voice_id": "Dina.wav",
        "output_format": "mulaw",
        "chunk_size": 25,
        "context_window": 50,
    },
    stream=True,  # Important: enables chunked reading
)

with open("output.ulaw", "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        f.write(chunk)
        # Each chunk is a segment of audio that can be played immediately
```

### Python — Real-Time Playback (PyAudio)

```python
import requests
import pyaudio
import audioop

# Open audio output stream
p = pyaudio.PyAudio()
speaker = p.open(format=pyaudio.paInt16, channels=1, rate=8000, output=True)

response = requests.post(
    "http://localhost:8004/tts/stream",
    json={
        "text": "Hello, this is a real-time playback test.",
        "voice_mode": "predefined",
        "predefined_voice_id": "Dina.wav",
        "output_format": "mulaw",
        "chunk_size": 25,
    },
    stream=True,
)

# Play audio as it arrives
for chunk in response.iter_content(chunk_size=1024):
    pcm_data = audioop.ulaw2lin(chunk, 2)  # Decode mu-law to PCM16
    speaker.write(pcm_data)

speaker.stop_stream()
speaker.close()
p.terminate()
```

### Python — Async with aiohttp

```python
import aiohttp
import asyncio

async def stream_tts():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8004/tts/stream",
            json={
                "text": "Async streaming example.",
                "voice_mode": "predefined",
                "predefined_voice_id": "Dina.wav",
                "output_format": "mulaw",
                "chunk_size": 25,
            },
        ) as response:
            async for chunk in response.content.iter_chunked(1024):
                # Process each audio chunk as it arrives
                # e.g., forward to a WebSocket, write to a buffer, etc.
                process_audio(chunk)

asyncio.run(stream_tts())
```

### JavaScript / Node.js

```javascript
const response = await fetch("http://localhost:8004/tts/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "Hello from JavaScript.",
    voice_mode: "predefined",
    predefined_voice_id: "Dina.wav",
    output_format: "mulaw",
    chunk_size: 25,
  }),
});

const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  // value is a Uint8Array of mu-law audio bytes
  // Forward to WebSocket, AudioWorklet, Twilio stream, etc.
  handleAudioChunk(value);
}
```

### JavaScript — Browser with Web Audio API

```javascript
async function streamAndPlay(text) {
  const response = await fetch("http://localhost:8004/tts/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: text,
      voice_mode: "predefined",
      predefined_voice_id: "Dina.wav",
      output_format: "wav",
      chunk_size: 25,
    }),
  });

  // For WAV format, collect all chunks and decode
  const arrayBuffer = await response.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 24000 });
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.start();
}
```

### Twilio Media Streams (WebSocket)

MULAW at 8kHz is Twilio's native audio format. Forward chunks directly:

```python
import websockets
import requests
import base64
import json

async def twilio_stream_handler(websocket):
    """Handle a Twilio WebSocket media stream connection."""
    stream_sid = None

    async for message in websocket:
        data = json.loads(message)

        if data["event"] == "start":
            stream_sid = data["start"]["streamSid"]

            # Start TTS streaming in background
            response = requests.post(
                "http://localhost:8004/tts/stream",
                json={
                    "text": "مرحبا، كيف يمكنني مساعدتك؟",
                    "voice_mode": "predefined",
                    "predefined_voice_id": "Dina.wav",
                    "output_format": "mulaw",
                    "chunk_size": 25,
                },
                stream=True,
            )

            # Forward audio chunks to Twilio
            for chunk in response.iter_content(chunk_size=640):  # 80ms at 8kHz
                media_message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": base64.b64encode(chunk).decode("utf-8"),
                    },
                }
                await websocket.send(json.dumps(media_message))
```

---

## Performance

### Benchmarks (NVIDIA GPU, Arabic fine-tuned model)

| Metric | Cold (1st call) | Warm (subsequent) |
|--------|-----------------|-------------------|
| Latency to first audio chunk | ~2.1s | **~0.9s** |
| Real-time factor (RTF) | ~1.2 | **~0.8** |
| Audio chunks per request | 3-6 | 3-6 |
| Audio duration per chunk | ~0.6-1.0s | ~0.6-1.0s |

> **RTF < 1.0** means audio is generated faster than real-time.
>
> The first call after server startup is slower due to one-time model compilation overhead.

### Tuning Guide

| Goal | Adjustment |
|------|------------|
| **Lowest latency** | `chunk_size: 15`, `context_window: 25` |
| **Best quality** | `chunk_size: 50`, `context_window: 100` |
| **Balanced (recommended)** | `chunk_size: 25`, `context_window: 50` |
| **Telephony optimized** | `output_format: "mulaw"`, `chunk_size: 25` |

---

## Comparison with Non-Streaming `/tts`

| Feature | `/tts` (batch) | `/tts/stream` (token-level) |
|---------|---------------|---------------------------|
| Time to first audio | Full generation time (~2-5s) | **~0.9s** |
| Output | Complete audio file | Incremental audio chunks |
| Text splitting | Splits long text into sentences | Processes full text as single stream |
| Best for | File generation, offline processing | Real-time playback, telephony, conversational AI |

---

## Error Handling

| HTTP Status | Meaning |
|-------------|---------|
| `200` | Success — audio stream follows |
| `422` | Validation error — check request parameters |
| `503` | Model not loaded — server is still starting up |

If the model encounters an error mid-stream, the stream will end prematurely. Monitor the server logs for error details.

---

## Available Predefined Voices

Query the available voices via:

```bash
curl http://localhost:8004/api/predefined-voices
```

Common voices: `Dina.wav`, `Adrian.wav`, `Alexander.wav`, `Alice.wav`, `Emily.wav`, `Gabriel.wav`, `Olivia.wav`, `Ryan.wav`, and more.
