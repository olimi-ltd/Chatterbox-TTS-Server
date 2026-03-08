#!/usr/bin/env python3
"""Download a file from Hugging Face with a tqdm progress bar."""

import sys
import requests
from pathlib import Path
from tqdm import tqdm

URL = "https://huggingface.co/NAMAA-Space/NAMAA-Egyptian-TTS/resolve/main/model.safetensors?download=true"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model.safetensors")


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Saved to {dest}")


if __name__ == "__main__":
    download(URL, OUTPUT)
