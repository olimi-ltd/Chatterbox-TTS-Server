# Installation Guide

This guide covers installing the Chatterbox TTS Server dependencies on a CPU-only setup (Linux/x86_64, Python 3.11).

## Prerequisites

- Python 3.11
- pip
- git
- cmake >= 3.28

## Known Dependency Conflicts

Two issues arise when installing `requirements.txt` from scratch:

### 1. `descript-audiotools` protobuf constraint

`descript-audiotools==0.7.2` declares `protobuf<3.20`, but `onnx>=1.13.0` requires `protobuf>=3.20`. This causes pip to fall back to `onnx==1.12.0`, which has no pre-built wheel and must be compiled from source — requiring `protoc` (the protobuf compiler), which is typically not available.

**Fix:** Patch the `descript-audiotools` wheel before installing to relax the protobuf upper bound.

### 2. numpy upper bound

`requirements.txt` originally pinned `numpy>=1.24.0,<1.26.0`. This is overly restrictive — PyTorch 2.5.1 works fine with numpy 2.x. The constraint has been updated to `numpy>=1.24.0`.

---

## Step-by-Step Installation

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Patch and pre-install `descript-audiotools`

Download the wheel, remove the `protobuf<3.20` upper bound, and install it before running the main requirements.

```bash
# Download the wheel
pip download descript-audiotools==0.7.2 --no-deps -d /tmp/descript_pkg

# Unpack it
mkdir -p /tmp/patched_descript
cd /tmp/patched_descript
unzip -q /tmp/descript_pkg/descript_audiotools-0.7.2-py2.py3-none-any.whl

# Remove the protobuf upper bound
sed -i 's/Requires-Dist: protobuf (<3.20,>=3.9.2)/Requires-Dist: protobuf (>=3.9.2)/' \
    descript_audiotools-0.7.2.dist-info/METADATA

# Repack as a valid wheel filename
zip -r /tmp/descript_audiotools-0.7.2-py2.py3-none-any.whl .

# Install the patched wheel
pip install /tmp/descript_audiotools-0.7.2-py2.py3-none-any.whl --no-deps
cd -
```

### 4. Install all requirements

```bash
pip install -r requirements.txt
```

This will install PyTorch 2.5.1 (CPU), chatterbox-tts, FastAPI, uvicorn, and all audio processing dependencies.

---

## Verifying the Installation

```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import chatterbox; print('chatterbox: ok')"
python -c "import fastapi; print('fastapi: ok')"
```

---

## GPU Installation

For NVIDIA or ROCm GPU support, use the corresponding requirements file instead:

```bash
# NVIDIA (CUDA 12.x)
pip install -r requirements-nvidia.txt

# NVIDIA (CUDA 12.8)
pip install -r requirements-nvidia-cu128.txt

# AMD ROCm
pip install -r requirements-rocm.txt
```

Apply the same `descript-audiotools` patch (Step 3) before running any of these.
