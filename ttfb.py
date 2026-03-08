import time
import requests

URL = "http://localhost:8004/stream/audio/speech"
OUTPUT_FILE = "outputs/output_stream.mulaw"
PAYLOAD = {
    "input": "صباح الخير",
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
