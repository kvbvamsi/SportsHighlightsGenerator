# 🏏 FlashBoundary — Real-time Cricket Highlight Generator

> Multi-modal AI pipeline for auto-generating cricket highlights, captions, and thumbnails  
> Built for AMD MI300X GPU · AUM GPU Hackathon 2025  
> Stack: vLLM · Pydantic AI · Whisper · EasyOCR · ffmpeg · Streamlit

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FlashBoundary Pipeline                       │
│                                                                  │
│  INPUT: Match video (MP4 / URL)                                 │
│         ↓                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Stage 1: Vision Agent                                    │   │
│  │  ffmpeg frame extraction @ 1fps → JPEG frames            │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Stage 2: Audio Agent                                     │   │
│  │  faster-whisper (AMD GPU) → timestamped commentary        │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Stage 3: Crowd Signal Agent                              │   │
│  │  Audio RMS analysis → energy peak timestamps             │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Stage 4: OCR Agent                                       │   │
│  │  EasyOCR / Tesseract → scorecard data extraction         │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Stage 5: Decision Agent (LLM on AMD MI300X)             │   │
│  │  vLLM (Qwen3-30B-A3B) fuses all signals → events         │   │
│  │  Fallback: Claude API (claude-sonnet-4-6)                 │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Stage 6/7: Packaging Agent                               │   │
│  │  ffmpeg clip cutting + caption overlay → segments         │   │
│  │  Concat → final_highlight_reel.mp4                       │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│         ↓                                                        │
│  OUTPUT: Highlight reel MP4 + thumbnails + CSV metadata         │
└──────────────────────────────────────────────────────────────┘
```

## Files

```
flashboundary/
├── app.py            ← Streamlit UI (6 tabs, real-time pipeline status)
├── backend.py        ← Full multi-modal pipeline (all 7 stages)
├── requirements.txt  ← Python dependencies
├── launch.sh         ← One-command setup + launch script
└── README.md         ← This file
```

## Quick Start

### On AMD MI300X (AUM GPU Server)

```bash
# 1. Start vLLM server (from build_airbnb_agent_mcp.ipynb)
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code

# 2. Launch FlashBoundary
bash launch.sh
# OR manually:
pip install -r requirements.txt
streamlit run app.py --server.maxUploadSize 500
```

### Claude API Fallback (no GPU required)
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
streamlit run app.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | AMD GPU vLLM server URL |
| `OPENAI_API_KEY` | `abc-123` | vLLM auth key |
| `VLLM_MODEL` | `Qwen3-30B-A3B` | Model served by vLLM |
| `ANTHROPIC_API_KEY` | _(unset)_ | Claude API fallback key |

## Connections to Workshop Notebooks

### `build_airbnb_agent_mcp.ipynb`
- **vLLM server setup**: Same `vllm serve` command used in `backend.py`'s `_call_vllm()`
- **Pydantic AI agent pattern**: `AGENTS` list in `backend.py` mirrors the agent-tool structure
- **MCP server pattern**: Each pipeline stage is structured as an independent "agent" with a clear role
- **OpenAI-compatible endpoint**: `_call_vllm()` uses identical auth/header pattern from notebook

### `Tutorial_Powering_Google_ADK_on_AMD_Platform_and_Local_LLMs.ipynb`
- **A2A multi-agent pattern**: The 7-stage pipeline is a direct implementation of A2A orchestration
- **Agent roles**: Supervisor/Vision/Audio/OCR/Decision/Packaging mirrors Burger/Pizza/Root agent pattern
- **Tool calling**: Event detection via LLM uses the same tool-call parser (`hermes`) from the ADK tutorial
- **vLLM GPU utilization**: Same `--gpu-memory-utilization` best practices applied

## Supported Events

| Event | Detection Signal |
|---|---|
| 🏏 **Six** | Commentary keywords + crowd energy peak + visual analysis |
| 🏏 **Four** | Commentary keywords + boundary frame detection |
| 🎯 **Wicket** | Commentary keywords + celebration detection |
| ⭐ **Player Milestone** | Commentary (fifty/hundred) + scorecard OCR |
| 🎉 **Celebration** | Visual frames + crowd energy + commentary |

## System Requirements

- **GPU**: AMD MI300X (ROCm 6.0+) or any CUDA GPU
- **RAM**: 32GB+ recommended
- **Storage**: 20GB+ for model weights
- **Python**: 3.10+
- **System**: `ffmpeg`, `tesseract-ocr`

## Test Video

The provided `Match_2__India_vs_United_Arab_Emirates__Match_Highlights__DP_World_Asia_Cup_2025.mp4`  
(India vs UAE, DP World Asia Cup 2025) is the primary test video. Upload it via the Home tab.
