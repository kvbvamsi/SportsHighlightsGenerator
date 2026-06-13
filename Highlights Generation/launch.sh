#!/usr/bin/env bash
# FlashBoundary — Setup & Launch Script
# Designed for AMD MI300X GPU server (AUM GPU environment)
# Based on build_airbnb_agent_mcp.ipynb + Tutorial_Powering_Google_ADK notebooks

set -e
echo "==========================================="
echo "  FlashBoundary | Cricket Highlight Setup  "
echo "  AMD MI300X GPU | AUM Hackathon 2025      "
echo "==========================================="

# ── System dependencies ──
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg tesseract-ocr libsm6 libxext6 python3-pip nodejs npm 2>/dev/null || true

# ── Python packages ──
echo "[2/6] Installing Python packages..."
pip install -q streamlit pandas requests Pillow openai
pip install -q pydantic-ai-slim
pip install -q faster-whisper easyocr pytesseract
pip install -q soundfile scipy numpy

# ── vLLM (AMD ROCm) — from build_airbnb_agent_mcp.ipynb ──
echo "[3/6] Starting vLLM server on AMD MI300X GPU..."
# Check if already running
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  vLLM server already running at http://localhost:8000"
else
    echo "  Starting vLLM with Qwen3-30B-A3B on AMD GPU..."
    nohup bash -c "
        VLLM_USE_TRITON_FLASH_ATTN=0 \
        vllm serve Qwen/Qwen3-30B-A3B \
            --served-model-name Qwen3-30B-A3B \
            --api-key abc-123 \
            --port 8000 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --trust-remote-code
    " > vllm_server.log 2>&1 &
    echo "  vLLM starting (PID=$!)... waiting 20s for init..."
    sleep 20
fi

# ── Environment variables ──
echo "[4/6] Setting environment variables..."
export VLLM_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="abc-123"
export VLLM_MODEL="Qwen3-30B-A3B"
# For Claude fallback mode, set: export ANTHROPIC_API_KEY="your-key-here"

# Write .env file
cat > .env << EOF
VLLM_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=abc-123
VLLM_MODEL=Qwen3-30B-A3B
# ANTHROPIC_API_KEY=your-key-here  # uncomment for fallback
EOF
echo "  .env written"

# ── Verify GPU ──
echo "[5/6] GPU Status:"
rocm-smi 2>/dev/null || nvidia-smi 2>/dev/null || echo "  GPU tool not found — check ROCm installation"

# ── Launch app ──
echo "[6/6] Launching FlashBoundary Streamlit app..."
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
