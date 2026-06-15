#!/usr/bin/env bash
# FlashBoundary — Setup & Launch Script
# Designed for AMD MI300X GPU server (AUM GPU environment)
#
# NOTE: This pipeline requires a real GPU + real audio/OCR transcription —
# there is NO cloud fallback and NO synthetic/demo data. If any required
# component is missing, generate_highlights() raises a RuntimeError with
# install instructions, and the Streamlit UI surfaces it directly.

set -e
echo "==========================================="
echo "  FlashBoundary | Cricket Highlight Setup  "
echo "  AMD MI300X GPU | AUM Hackathon 2025      "
echo "==========================================="

# ── System dependencies ──
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg tesseract-ocr libsm6 libxext6 python3-pip 2>/dev/null || true

# ── Python packages ──
echo "[2/5] Installing Python packages..."
pip install -q --break-system-packages \
    streamlit pandas requests Pillow \
    faster-whisper easyocr soundfile numpy

# ── vLLM (AMD ROCm) ──
echo "[3/5] Starting vLLM server on AMD MI300X GPU..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  vLLM server already running at http://localhost:8000"
else
    echo "  Starting vLLM with Qwen3-30B-A3B on AMD GPU..."
    nohup bash -c "VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Qwen/Qwen3-30B-A3B --served-model-name Qwen3-30B-A3B --api-key abc-123 --port 8000 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code" > vllm_server.log 2>&1 &
    echo "  vLLM starting (PID=$!)... waiting 20s for init..."
    sleep 20
fi

# ── Environment variables ──
echo "[4/5] Setting environment variables..."
export VLLM_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="abc-123"
export VLLM_MODEL="Qwen3-30B-A3B"

{
  echo "VLLM_BASE_URL=http://localhost:8000/v1"
  echo "OPENAI_API_KEY=abc-123"
  echo "VLLM_MODEL=Qwen3-30B-A3B"
} > .env
echo "  .env written"

# ── Verify GPU ──
echo "  GPU Status:"
rocm-smi 2>/dev/null || nvidia-smi 2>/dev/null || echo "  GPU tool not found — check ROCm installation"

# ── Launch app ──
echo "[5/5] Launching FlashBoundary Streamlit app..."
echo ""
echo "  App URL: http://localhost:8501"
echo "  vLLM:    http://localhost:8000"
echo ""
echo "  Monitor GPU: watch rocm-smi"
echo ""

streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.maxUploadSize 500 \
    --browser.gatherUsageStats false
