# 🏏 FlashBoundary — Real-time Cricket Highlight Generator

> Multi-modal AI pipeline that auto-generates cricket highlight reels, captions, and thumbnails
> from raw match video — by fusing visual frames, commentary audio, crowd energy, and live
> scoreboard OCR through an LLM running on **AMD MI300X GPU via vLLM**.
>
> Built for AUM GPU Hackathon 2025. **No cloud LLM fallback. No synthetic/demo data. No
> hardcoded player names — ever.**

---

## Files

```
flashboundary/
├── app.py            ← Streamlit UI (7 tabs, live progress + ETA, GPU status)
├── backend.py        ← Full 8-stage multi-modal pipeline
├── requirements.txt  ← Python dependencies (all required, no optional cloud fallback)
├── launch.sh         ← One-command setup: installs deps, starts vLLM, launches the app
└── README.md         ← This file
```

---

## Architecture — 8-Stage Pipeline

```
INPUT: Match video (upload or direct .mp4 URL)
  │
  ├─▶ 1. Vision Agent       — ffmpeg frame extraction @ 1fps
  ├─▶ 2. Audio Agent        — faster-whisper transcription (AMD GPU)
  ├─▶ 3. Crowd Signal Agent — audio RMS energy-peak detection
  ├─▶ 4a. OCR Agent         — dense EasyOCR scoreboard timeline (every ~5s)
  ├─▶ 4b. Roster Agent      — player names extracted from OCR (zero hardcoding)
  ├─▶ 4b-2. Cross-Ref Agent — matches commentary names ↔ scoreboard names
  ├─▶ 4c. Wicket Delta Agent— wicket detection via scorecard score/batter deltas
  ├─▶ 5. Decision Agent     — vLLM (Qwen3-30B-A3B on MI300X) fuses all signals → events
  ├─▶ 6. Dedup Agent        — per-type + global minimum gap enforcement
  ├─▶ 7. Packaging Agent    — event-duration-aware clip cutting + caption overlay
  ├─▶ 7b. Thumbnail Agent   — per-event thumbnail generation
  └─▶ 8. Reel Agent         — concatenates clips into final_highlight_reel.mp4

OUTPUT: Highlight reel (.mp4) + thumbnails + segment/analytics CSVs + player roster
```

Every stage reports progress via a weighted `progress_callback`, so the Streamlit UI shows
**live elapsed time + estimated time remaining** as the pipeline runs.

---

## Quick Start

### On AMD MI300X (AUM GPU Server)

```bash
bash launch.sh
```

This installs system + Python dependencies, starts the vLLM server (Qwen3-30B-A3B on
ROCm), writes a `.env`, and launches the Streamlit app on port 8501.

### Manual setup

```bash
# 1. System deps
apt-get install -y ffmpeg tesseract-ocr

# 2. Python deps
pip install -r requirements.txt --break-system-packages

# 3. Start vLLM on AMD MI300X
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Qwen/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --api-key abc-123 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --trust-remote-code

# 4. Launch the app
streamlit run app.py --server.maxUploadSize 500
```

---

## Environment Variables

| Variable          | Default                     | Description                          |
|-------------------|------------------------------|---------------------------------------|
| `VLLM_BASE_URL`   | `http://localhost:8000/v1`   | AMD GPU vLLM OpenAI-compatible endpoint |
| `OPENAI_API_KEY`  | `abc-123`                     | vLLM auth key (any value vLLM was started with) |
| `VLLM_MODEL`      | `Qwen3-30B-A3B`                | Model name served by vLLM            |

There is **no `ANTHROPIC_API_KEY` / cloud fallback path** — this was deliberately removed
(see Changelog). If the vLLM server is unreachable, the app shows a clear setup error
instead of silently degrading.

---

## Design Principles (and why they matter)

### 1. No synthetic/demo data, anywhere
Earlier versions had fallback paths that generated fake commentary lines, fake crowd
peaks, and a hardcoded scoreboard (`"Rohit Sharma"`, `"Virat Kohli"`, etc.) whenever a
dependency was missing. **All of these were removed.** Each of `transcribe_audio()`,
`detect_crowd_energy()`, and `ocr_frames_full()` now raises a `RuntimeError` with
specific install instructions if its required library/engine is unavailable, or if it
runs but produces zero usable output. The Streamlit UI surfaces this as "Pipeline
Stopped — Real Data Required" with the exact missing dependency.

### 2. Player identification with zero hardcoding
Player names are discovered **live**, per video, by cross-referencing two independent
signals:
- **Commentary audio** (faster-whisper) → name-like tokens extracted via regex,
  filtered against a stopword list, with leading-filler-word stripping
  (e.g. "And Sharma" → "Sharma").
- **Scoreboard OCR** (EasyOCR, sampled every ~5s) → structured batter
  name/runs/balls/striker data parsed from the on-screen scorecard.

`cross_reference_players()` matches these within a ±12s window using:
1. exact match, 2. surname/word-subset match (handles "Kohli" ↔ "Virat Kohli" —
the most common real-world case since commentators say surnames), 3. fuzzy
similarity for ASR noise.

The result feeds the LLM prompt as **CONFIRMED** (both signals agree — preferred)
vs **SCOREBOARD-ONLY** (seen on scoreboard, not yet heard) names. The LLM is
instructed to never invent a name outside these lists.

### 3. Event-aware clip durations, one event at a time
Each event type has a tuned action duration (`EVENT_DURATIONS`), plus shared
pre-roll/post-roll padding:

| Event Type        | Action Duration | Total Clip (with padding) |
|--------------------|-----------------|----------------------------|
| Six                | 8.0s            | 12.5s |
| Four               | 6.0s            | 10.5s |
| Wicket             | 12.0s           | 16.5s |
| Player Milestone   | 10.0s           | 14.5s |
| Celebration        | 8.0s            | 12.5s |

(Pre-roll = 2.5s, post-roll = 2.0s)

### 4. No duplicate/overlapping events
`deduplicate_events()` enforces both a **per-event-type minimum gap**
(`MIN_EVENT_GAP_SEC` — e.g. Wickets ≥30s apart, Sixes ≥15s apart) and a
**global minimum gap** (10s, any event type) so two events never combine
or fire back-to-back from noisy detections. When two candidates collide,
the higher-confidence one is kept.

### 5. Wicket detection via scorecard delta, not just commentary
`detect_wicket_events_from_ocr()` compares consecutive OCR scoreboard
snapshots — a wicket is flagged when the wicket count increases **or**
the batter lineup changes (one name disappears, another appears). This
runs independently of the LLM and is merged with LLM-detected events
before deduplication, catching wickets the commentary might phrase
ambiguously.

### 6. GPU status transparency
The sidebar shows live vLLM health (`check_vllm_health()`). If the vLLM
server is offline, `detect_local_gpu()` queries `rocm-smi`/`nvidia-smi`
directly so the user can see **what GPU hardware is actually present**
on the host even when the inference server hasn't been started —
distinguishing "no GPU at all" from "GPU present, server not running".

### 7. Robust URL input handling
Direct video URLs are validated before processing: Content-Type is
checked (rejects HTML/JSON — e.g. a YouTube/Drive share page instead of
a direct file link), file size is sanity-checked (>10KB), and
`_get_video_duration()` returns `0.0` (never a fake default) on probe
failure so the pipeline fails fast with a clear message instead of
silently proceeding with a corrupted file.

### 8. Robust reel stitching
`stitch_reel()` tries fast stream-copy concat first, then falls back to
a full re-encode concat if segments have incompatible codec parameters
(common with URL-sourced videos). `cut_segment()` validates output file
existence/size and logs ffmpeg stderr on failure. The pipeline
distinguishes and reports three distinct empty-reel causes: zero events
detected, all segment cuts failed, or stitching failed — each with an
actionable message in the execution log (visible in the Final Reel tab).

### 9. Robust LLM JSON parsing
`_parse_llm_event_json()` handles `<think>...</think>` reasoning
preambles (Qwen3), markdown fences, leading commentary text, and
truncated arrays (recovers complete objects if `max_tokens` cut the
response short). One automatic retry with a stricter "JSON only" prompt
if the first response is unparsable.

---

## Supported Events & Detection Signals

| Event | Primary Signals |
|---|---|
| 🏏 **Six** | Commentary keywords + crowd energy peak + visual frame analysis |
| 🏏 **Four** | Commentary keywords + crowd energy + visual frame analysis |
| 🎯 **Wicket** | Scorecard delta (wicket count / batter change) + commentary |
| ⭐ **Player Milestone** | Commentary ("fifty"/"hundred") + scorecard OCR (run total) |
| 🎉 **Celebration** | Visual frames + crowd energy + commentary |

---

## UI Tabs

1. **Home** — upload video / paste direct URL, configure event types + confidence
   threshold + player filter, live progress bar with elapsed time + ETA.
2. **Final Reel** — plays the stitched reel; if generation produced no reel, shows
   diagnostic warnings and the full execution log explaining why.
3. **Thumbnails** — per-event captioned thumbnails, filterable by event type.
4. **Player View** — filter highlights by player (auto-populated from the
   confirmed/OCR roster), with per-player Six/Four/Wicket counts.
5. **Agent Insights** — live status from each of the 12 pipeline agents plus the
   full execution log.
6. **Analytics** — event-type distribution, per-player event counts, and an
   LLM-vs-OCR-delta source breakdown.
7. **Exports** — CSV downloads (segments, analytics), log download, and a
   roster breakdown (confirmed / scoreboard-only / unmatched commentary names).

---

## System Requirements

- **GPU**: AMD MI300X (ROCm) running vLLM — or any CUDA GPU with vLLM as a substitute
- **RAM**: 32GB+ recommended
- **Python**: 3.10+
- **System packages**: `ffmpeg` (required), `tesseract-ocr` (only if using the
  pytesseract OCR fallback instead of EasyOCR)

---

## Test Video

`Match_2__India_vs_United_Arab_Emirates__Match_Highlights__DP_World_Asia_Cup_2025.mp4`
(India vs UAE, DP World Asia Cup 2025) — upload via the Home tab to test the full
pipeline end-to-end, including roster cross-referencing against the real on-screen
scoreboard.

---

## Changelog — Refinements Made During Development

1. **Initial build**: 7-stage pipeline (frames → Whisper → crowd RMS → OCR → vLLM
   fusion with Claude API fallback → clip cutting → reel stitch), Streamlit UI with
   7 tabs, AMD MI300X setup via vLLM (Qwen3-30B-A3B).

2. **Player roster from scoreboard, no hardcoding** — added `extract_player_roster()`
   to parse batter names from OCR instead of a hardcoded list; event-specific clip
   durations (`EVENT_DURATIONS`); per-type + global minimum event gaps
   (`MIN_EVENT_GAP_SEC`, `GLOBAL_MIN_GAP_SEC`) so events never combine; wicket
   detection via scorecard delta (`detect_wicket_events_from_ocr`); removed Claude
   fallback entirely — GPU required, clear error if unavailable; added
   `detect_local_gpu()` to show actual GPU hardware name even when vLLM is offline.

3. **Live progress + ETA** — added `progress_callback` and `STAGE_WEIGHTS` to
   `generate_highlights()` so the UI shows elapsed time and an estimated-time-remaining
   that self-corrects as stages complete.

4. **Robust LLM JSON parsing** — fixed `JSONDecodeError` crashes caused by Qwen3
   `<think>...</think>` reasoning preambles and truncated responses. Added
   `_parse_llm_event_json()` with bracket-depth tracking, truncation recovery, and
   one automatic retry with a stricter prompt.

5. **Removed ALL synthetic/demo fallbacks** — this was the biggest correctness fix.
   Previously, `transcribe_audio()`, `detect_crowd_energy()`, and `ocr_frames_full()`
   silently substituted fake commentary, fake crowd peaks, and a hardcoded
   Rohit-Sharma/Virat-Kohli scoreboard if their respective libraries were missing.
   All three now raise `RuntimeError` with install instructions if unavailable, or if
   they run but extract zero usable signal.

6. **Audio ↔ OCR player cross-referencing** — added `cross_reference_players()`,
   `_extract_name_candidates()`, and `_fuzzy_match()` (exact → surname/subset →
   fuzzy) to build a verified roster from real commentary + real scoreboard data,
   with confirmed/scoreboard-only/unmatched breakdowns surfaced in the UI and fed
   to the LLM prompt.

7. **Fixed URL-input reel generation** — added Content-Type and file-size
   validation on downloads (rejects HTML share-page links), made
   `_get_video_duration()` return `0.0` instead of a fake 300s default on probe
   failure, hardened `cut_segment()` (bounds-checking, output validation, detailed
   ffmpeg error logging) and `stitch_reel()` (stream-copy → re-encode fallback), and
   made the pipeline distinguish/report "no events detected" vs "all clips failed"
   vs "stitch failed" with actionable messages in the Final Reel tab.

8. **Project deck** — generated an 8-slide presentation (`FlashBoundary.pptx`)
   covering problem statement, architecture, the player cross-referencing
   innovation, AMD MI300X tech stack, UI walkthrough, and roadmap.
