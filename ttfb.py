import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

PORT = os.environ.get("SERVER_PORT", "8004")
URL = f"http://localhost:{PORT}/stream/audio/speech"
OUTPUT_FILE = "outputs/output_stream.mulaw"
PAYLOAD = {
    "input": "ألو مين معايا؟",
    "voice_id": "Dina-ref",
    "output_format": "mulaw",
    "chunk_size": 25,
}

start = time.perf_counter()

with requests.post(URL, json=PAYLOAD, stream=True) as resp:
    resp.raise_for_status()
    ttfb = None
    with open(OUTPUT_FILE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                if ttfb is None:
                    ttfb = time.perf_counter() - start
                    print(f"Time to first binary chunk: {ttfb * 1000:.1f} ms")
                f.write(chunk)

total = time.perf_counter() - start
print(f"Total time:                  {total * 1000:.1f} ms")
print(f"Saved to {OUTPUT_FILE}")
