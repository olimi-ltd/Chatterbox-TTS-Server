# Installation Guide

This guide covers installing the Chatterbox TTS Server dependencies on Linux/x86_64 with Python 3.11, for both CPU-only and GPU setups.

## Prerequisites

- Python 3.11
- pip
- git
- cmake >= 3.28
- unzip, zip (for patching the `descript-audiotools` wheel)

## Known Dependency Conflicts

Two issues arise when installing requirements from scratch:

### 1. `descript-audiotools` protobuf constraint

`descript-audiotools==0.7.2` declares `protobuf<3.20`, but `onnx>=1.13.0` requires `protobuf>=3.20`. This causes pip to fall back to `onnx==1.12.0`, which has no pre-built wheel and must be compiled from source — requiring `protoc` (the protobuf compiler), which is typically not available.

**Fix:** Patch the `descript-audiotools` wheel before installing to relax the protobuf upper bound. This patch must be applied regardless of whether you are doing a CPU or GPU installation.

### 2. numpy upper bound

`requirements.txt` originally pinned `numpy>=1.24.0,<1.26.0`. This is overly restrictive — PyTorch 2.5.1 works fine with numpy 2.x. The constraint has been updated to `numpy>=1.24.0`.

---

## Step-by-Step Installation (CPU)

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
# Clean up any previous patch artifacts
rm -rf /tmp/descript_pkg /tmp/patched_descript

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

## Step-by-Step Installation (NVIDIA GPU)

If you have an NVIDIA GPU, follow these steps instead of the CPU installation above. Steps 1–3 are identical; only Step 4 differs.

### 1–3. Same as CPU

Follow Steps 1, 2, and 3 from the CPU installation above (create venv, upgrade pip, patch descript-audiotools).

### 4. Choose the right requirements file

Check your GPU and CUDA driver version:

```bash
nvidia-smi
```

Then install the matching requirements file:

```bash
# NVIDIA GPU with CUDA 12.1 (widely compatible — recommended for most GPUs)
pip install -r requirements-nvidia.txt

# NVIDIA GPU with CUDA 12.8 (for RTX 5090 / Blackwell architecture)
pip install -r requirements-nvidia-cu128.txt
```

This will install CUDA-enabled PyTorch (e.g. `torch==2.5.1+cu121`) along with all other dependencies.

### 5. Verify CUDA is available

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You should see `CUDA available: True` and your GPU name.

> **Important:** The `config.yaml` file ships with `device: cuda` by default. If PyTorch was installed without CUDA support (e.g. from `requirements.txt`), the server will silently fall back to CPU. Always use the correct requirements file for your hardware.

---

## Step-by-Step Installation (AMD ROCm)

Follow Steps 1–3 from the CPU installation, then:

```bash
pip install -r requirements-rocm.txt
```

---

## Verifying the Installation

```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import chatterbox; print('chatterbox: ok')"
python -c "import fastapi; print('fastapi: ok')"
```

---

## Running the Server

```bash
source .venv/bin/activate
python server.py
```

The server starts on `http://0.0.0.0:8004` by default (configurable in `config.yaml`).

To verify the server is working, you can run the TTFB (time-to-first-byte) benchmark in a separate terminal:

```bash
source .venv/bin/activate
python ttfb.py
```

### Expected Performance

| Metric | CPU | NVIDIA L4 (CUDA 12.1) |
|--------|-----|------------------------|
| Time to first byte | ~26,000 ms | ~2,200 ms |
| Total generation time | ~150,000 ms | ~8,600 ms |

*Measured with the default ttfb.py payload (~140 words). GPU performance may vary with model warm-up and hardware.*
