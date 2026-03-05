import time
import requests

URL = "http://localhost:8004/tts/stream"
PAYLOAD = {
    "text": "Okay, we have fifteen minutes left! Fifteen minutes! [groan] How have we been in here for forty-five minutes and only solved two puzzles?",
    "voice_mode": "predefined",
    "predefined_voice_id": "Dina-ref.wav",
    "output_format": "mulaw",
    "chunk_size": 25,
}

start = time.perf_counter()

with requests.post(URL, json=PAYLOAD, stream=True) as resp:
    resp.raise_for_status()
    ttfb = None
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            if ttfb is None:
                ttfb = time.perf_counter() - start
                print(f"Time to first binary chunk: {ttfb * 1000:.1f} ms")

total = time.perf_counter() - start
print(f"Total time:                  {total * 1000:.1f} ms")
