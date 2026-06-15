"""
backend.py — FlashBoundary Cricket Highlight Generator
Multi-modal pipeline: video frames + commentary audio + crowd signals
Connects to AMD MI300X GPU via vLLM (OpenAI-compatible endpoint)

Key improvements:
- OCR-based player roster extraction from scoreboard frames (no hardcoding)
- Scorecard delta detection for Wicket events (score/player change tracking)
- Event-type-aware clip durations
- Minimum gap enforcement to prevent duplicate events
- No Claude fallback — GPU required; clear error if unavailable
"""

from __future__ import annotations

import base64
import difflib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("flashboundary.backend")
logger.setLevel(logging.DEBUG)

# ─────────────────────────────────────────────
# Server config — AMD MI300X vLLM ONLY
# ─────────────────────────────────────────────
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY  = os.getenv("OPENAI_API_KEY", "abc-123")
VLLM_MODEL    = os.getenv("VLLM_MODEL", "Qwen3-30B-A3B")

CAPTION_FONT_SIZE = 24

# ── Per-event clip durations (seconds of action, before padding is added) ──
EVENT_DURATIONS: dict[str, float] = {
    "Six":              8.0,   # ball → full arc → landing
    "Four":             6.0,   # shot → ball crossing boundary
    "Wicket":          12.0,   # delivery → dismissal → reaction + replay
    "Player Milestone": 10.0,  # moment + crowd + celebration
    "Celebration":      8.0,   # team huddle / pumping
}
DEFAULT_DURATION  = 7.0
PRE_ROLL_SEC      = 2.5   # seconds before event centre to start clip
POST_ROLL_SEC     = 2.0   # extra seconds after action duration

# ── Minimum gap between any two events (seconds) — prevents duplicates ──
MIN_EVENT_GAP_SEC: dict[str, float] = {
    "Six":              15.0,
    "Four":             12.0,
    "Wicket":           30.0,  # wickets are rare; bigger gap
    "Player Milestone": 45.0,
    "Celebration":      20.0,
}
GLOBAL_MIN_GAP_SEC = 10.0   # absolute minimum between any two events

CRICKET_EVENT_KEYWORDS = {
    "Six":              ["six", "maximum", "over the boundary", "that's a six", "50 metres", "huge hit", "into the crowd"],
    "Four":             ["four", "boundary", "races away", "to the fence", "off the edge", "through the gap"],
    "Wicket":           ["wicket", "out", "caught", "bowled", "lbw", "stumped", "run out", "gone", "dismissed"],
    "Player Milestone": ["fifty", "hundred", "century", "milestone", "50 runs", "100 runs", "50 up", "century up"],
    "Celebration":      ["celebration", "high five", "team huddle", "pumped", "fist pump", "roar"],
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.debug("CMD: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _get_video_duration(video_path: str) -> float:
    """
    Probe video duration via ffprobe. Returns 0.0 if the file is missing,
    unreadable, or not a valid media file — callers must check for this
    rather than silently proceeding with a fake duration.
    """
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Install with: apt-get install ffmpeg")
    if not video_path or not Path(video_path).exists():
        return 0.0
    try:
        result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "csv=p=0", video_path])
        if result.returncode != 0:
            logger.warning("ffprobe error for %s: %s", video_path, result.stderr[-300:])
            return 0.0
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# LLM gateway — vLLM on AMD MI300X ONLY
# Raises RuntimeError if server unreachable (no fallback)
# ─────────────────────────────────────────────

def check_vllm_health() -> tuple[bool, str]:
    """Returns (online, message). Does not raise."""
    try:
        r = requests.get(
            f"{VLLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            timeout=4,
        )
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            return True, f"Online — {', '.join(models)}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def detect_local_gpu() -> str:
    """
    Detect the physical GPU on this host via rocm-smi (AMD) or nvidia-smi (NVIDIA).
    Used to surface the GPU name even when the vLLM server itself is offline.
    Returns a human-readable string, e.g. 'AMD Instinct MI300X' or 'No GPU detected'.
    """
    # AMD ROCm
    try:
        result = _run(["rocm-smi", "--showproductname"])
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                if "Card series" in line or "Product Name" in line:
                    name = line.split(":", 1)[-1].strip()
                    if name:
                        return f"AMD {name}"
            if "GPU" in result.stdout:
                return "AMD GPU (ROCm) — name unparsed"
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # NVIDIA CUDA
    try:
        result = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return "No GPU detected on this host"


def _llm(messages: list[dict], max_tokens: int = 512) -> str:
    """
    Call AMD MI300X vLLM server. Raises RuntimeError if unavailable.
    No Claude fallback — GPU is required for this pipeline.
    """
    headers = {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       VLLM_MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.15,
    }
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"AMD MI300X vLLM server is not reachable at {VLLM_BASE_URL}. "
            f"Start the server with: vllm serve {VLLM_MODEL} --port 8000 --api-key abc-123"
        ) from e
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"vLLM server at {VLLM_BASE_URL} timed out. "
            "The GPU may be overloaded or the model is still loading."
        )
    except Exception as e:
        raise RuntimeError(f"vLLM call failed: {e}") from e


# ─────────────────────────────────────────────
# Stage 1: Frame extraction
# ─────────────────────────────────────────────

def extract_frames(video_path: str, workdir: str, fps: float = 1.0) -> list[str]:
    """Extract frames at `fps` using ffmpeg."""
    frames_dir = _ensure_dir(os.path.join(workdir, "frames"))
    if _ffmpeg_available():
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "3",
            os.path.join(frames_dir, "frame_%05d.jpg"),
            "-y",
        ]
        result = _run(cmd)
        if result.returncode != 0:
            logger.warning("ffmpeg frame extraction failed: %s", result.stderr[-400:])
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


# ─────────────────────────────────────────────
# Stage 2: Audio transcription (Whisper on AMD GPU)
# ─────────────────────────────────────────────

def transcribe_audio(video_path: str, workdir: str) -> list[dict]:
    """
    Transcribe match commentary using faster-whisper (AMD GPU) or openai-whisper.
    Raises RuntimeError if no transcription backend is available — no synthetic
    fallback, since fake commentary produces fake players/events downstream.
    """
    segments_out = []
    audio_path = os.path.join(workdir, "audio.wav")

    if not _ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed/available on PATH. "
            "Install it with: apt-get install ffmpeg"
        )

    extract = _run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", audio_path, "-y"])
    if extract.returncode != 0 or not Path(audio_path).exists():
        raise RuntimeError(f"ffmpeg failed to extract audio track: {extract.stderr[-400:]}")

    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel("base", device="cuda", compute_type="float16")
        segs, _ = model.transcribe(audio_path, beam_size=5)
        for seg in segs:
            segments_out.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        logger.info("faster_whisper: %d segments", len(segments_out))
        if segments_out:
            return segments_out
        raise RuntimeError("faster_whisper returned zero commentary segments — check audio track has speech")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("faster_whisper unavailable: %s", e)

    try:
        import whisper  # type: ignore
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        for seg in result.get("segments", []):
            segments_out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()})
        logger.info("openai-whisper: %d segments", len(segments_out))
        if segments_out:
            return segments_out
        raise RuntimeError("openai-whisper returned zero commentary segments — check audio track has speech")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("whisper unavailable: %s", e)

    raise RuntimeError(
        "No audio transcription backend available. Install one of:\n"
        "  pip install faster-whisper   (recommended — GPU accelerated)\n"
        "  pip install openai-whisper\n"
        "FlashBoundary requires real commentary transcription — "
        "no synthetic/demo commentary is used."
    )


# ─────────────────────────────────────────────
# Stage 3: Crowd audio energy detection
# ─────────────────────────────────────────────

def detect_crowd_energy(video_path: str, workdir: str) -> list[dict]:
    """
    Detect high-energy crowd moments using audio RMS analysis.
    Raises RuntimeError if audio analysis libraries are unavailable or the
    audio file can't be read — no synthetic crowd peaks are substituted.
    """
    events = []
    audio_path = os.path.join(workdir, "audio.wav")

    if not Path(audio_path).exists():
        raise RuntimeError(
            f"Audio file not found at {audio_path}. "
            "Audio extraction must run before crowd signal analysis."
        )

    import numpy as np
    data = None
    sr = None

    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(audio_path)
    except Exception as e1:
        try:
            from scipy.io import wavfile  # type: ignore
            sr, data = wavfile.read(audio_path)
            data = data.astype(np.float32) / 32768.0
        except Exception as e2:
            raise RuntimeError(
                "No audio-reading library available for crowd signal analysis. Install one of:\n"
                "  pip install soundfile\n"
                "  pip install scipy\n"
                f"soundfile error: {e1}\nscipy error: {e2}"
            )

    if data is None or sr is None or len(data) == 0:
        raise RuntimeError(f"Audio file at {audio_path} is empty or unreadable.")

    if data.ndim > 1:
        data = data.mean(axis=1)

    window    = int(sr * 2.0)
    hop       = int(sr * 1.0)   # 1-second hop — prevents dense duplicates
    threshold = np.percentile(np.abs(data), 92)

    prev_peak = -999.0
    for start_i in range(0, len(data) - window, hop):
        chunk = data[start_i: start_i + window]
        rms   = float(np.sqrt(np.mean(chunk ** 2)))
        ts    = start_i / sr
        # Enforce minimum 5s between crowd peaks to reduce noise
        if rms > threshold and (ts - prev_peak) >= 5.0:
            events.append({"timestamp": ts, "energy": rms, "type": "crowd_roar"})
            prev_peak = ts

    logger.info("Crowd peaks: %d", len(events))
    return events


# ─────────────────────────────────────────────
# Stage 4a: OCR ALL frames for scorecard tracking
# Returns list of {frame_path, timestamp_sec, text, wickets, score, batters}
# ─────────────────────────────────────────────

_SCORE_RE   = re.compile(r"(\d{1,3})[/\-](\d{1,2})\s*[\(\[]?\s*(\d{1,2}(?:\.\d)?)\s*(?:ov|overs?)?", re.I)
_BATTER_RE  = re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(\d+)\s*\((\d+)\)", re.I)
_STRIKER_RE = re.compile(r"\*\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")


def _parse_scorecard_text(text: str) -> dict:
    """Extract structured data from a raw OCR text line."""
    info: dict = {"raw": text, "batters": [], "score": None, "wickets": None, "overs": None}

    m = _SCORE_RE.search(text)
    if m:
        info["score"]   = int(m.group(1))
        info["wickets"] = int(m.group(2))
        info["overs"]   = float(m.group(3))

    for bm in _BATTER_RE.finditer(text):
        name = bm.group(1).strip()
        runs = int(bm.group(2))
        balls = int(bm.group(3))
        is_striker = bool(_STRIKER_RE.search(text[max(0, bm.start()-5): bm.start()]))
        info["batters"].append({
            "name": name, "runs": runs, "balls": balls, "striker": is_striker
        })

    return info


def ocr_frames_full(frames: list[str], video_duration: float) -> list[dict]:
    """
    OCR every Nth frame (denser than before: every 5s of video).
    Returns time-stamped scorecard snapshots with parsed data.
    Raises RuntimeError if no OCR engine is available, or if OCR runs but
    extracts zero usable text — no hardcoded/synthetic scorecard data is
    substituted, since fake player names defeat the purpose of this pipeline.
    """
    if not frames:
        raise RuntimeError("No frames available for OCR — frame extraction must run first.")

    ocr_timeline: list[dict] = []
    step = max(1, len(frames) // max(1, int(video_duration / 5)))
    sampled = frames[::step][:120]  # cap at 120 frames

    reader = None
    use_tesseract = False

    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["en"], gpu=True)
        logger.info("OCR: using EasyOCR (GPU)")
    except Exception as e1:
        try:
            import pytesseract  # type: ignore
            use_tesseract = True
            logger.info("OCR: using Tesseract")
        except Exception as e2:
            raise RuntimeError(
                "No OCR engine available. Install one of:\n"
                "  pip install easyocr        (recommended — GPU accelerated)\n"
                "  pip install pytesseract && apt-get install tesseract-ocr\n"
                f"easyocr error: {e1}\npytesseract error: {e2}\n"
                "FlashBoundary requires real scoreboard OCR to identify players — "
                "no hardcoded player names are used."
            )

    fps_approx = len(frames) / max(video_duration, 1)

    for i, frame_path in enumerate(sampled):
        # Estimate timestamp from frame index in original list
        orig_idx = frames.index(frame_path) if frame_path in frames else i * step
        ts = orig_idx / max(fps_approx, 1)

        text = ""
        try:
            if reader:
                results = reader.readtext(frame_path)
                text = " | ".join(r[1] for r in results if r[2] > 0.45)
            elif use_tesseract:
                from PIL import Image
                import pytesseract  # type: ignore
                text = pytesseract.image_to_string(Image.open(frame_path), config="--psm 6")
        except Exception as ex:
            logger.debug("OCR frame %s failed: %s", frame_path, ex)

        if text.strip():
            parsed = _parse_scorecard_text(text)
            parsed["frame_path"]    = frame_path
            parsed["timestamp_sec"] = ts
            ocr_timeline.append(parsed)

    if not ocr_timeline:
        raise RuntimeError(
            "OCR ran but extracted zero usable text from any sampled frame. "
            "This usually means the scoreboard overlay is too small/low-contrast "
            "for the OCR engine, or the video has no on-screen scorecard. "
            "Try a higher-resolution source video, or a different OCR engine."
        )

    logger.info("OCR timeline: %d snapshots", len(ocr_timeline))
    return ocr_timeline


# ─────────────────────────────────────────────
# Stage 4b: Extract player roster from scoreboard
# ─────────────────────────────────────────────

def extract_player_roster(ocr_timeline: list[dict]) -> dict[str, list[str]]:
    """
    Build a roster of all player names seen in scorecard OCR across both teams.
    Returns: {"batting": [...], "bowling": [...], "all": [...]}
    """
    seen: set[str] = set()
    for snap in ocr_timeline:
        for b in snap.get("batters", []):
            name = b.get("name", "").strip()
            if name and len(name) > 3:
                seen.add(name)
        # Also scan raw text for name-like tokens (Capital First Last pattern)
        raw = snap.get("raw", "")
        for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b", raw):
            candidate = m.group(1)
            # Filter out common false positives
            if candidate.lower() not in {"over", "overs", "runs", "balls", "wicket",
                                          "target", "required", "rate", "power", "play",
                                          "last", "this", "total", "india", "emirates",
                                          "batting", "bowling", "run"}:
                seen.add(candidate)

    roster = sorted(seen)
    logger.info("Player roster extracted from OCR: %s", roster)
    return {"all": roster, "batting": roster, "bowling": []}


# ─────────────────────────────────────────────
# Stage 4b-2: Cross-reference commentary audio with
# scoreboard OCR to verify/extend the player roster
# ─────────────────────────────────────────────

_COMMENTARY_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")

_COMMON_NON_NAME_WORDS = {
    "over", "overs", "runs", "run", "balls", "wicket", "wickets", "target",
    "required", "rate", "power", "play", "last", "this", "that", "total",
    "india", "emirates", "batting", "bowling", "six", "four", "fifty",
    "hundred", "century", "out", "lbw", "caught", "bowled", "stumped",
    "umpire", "review", "drs", "boundary", "maximum", "crowd", "innings",
    "match", "asia", "cup", "world", "what", "well", "great", "good",
    "now", "here", "there", "today", "welcome", "back", "let", "look",
    "and", "but", "for", "with", "from", "into", "onto", "off", "the",
    "his", "her", "they", "their", "very", "just", "still", "again",
    "another", "another's", "such", "some", "all",
}

# Leading filler words that sometimes get capitalised by Whisper and merged
# into a name candidate (e.g. "And Sharma" -> "Sharma")
_LEADING_FILLER_WORDS = {"and", "but", "so", "now", "well", "ok", "okay", "oh", "yes", "no"}


def _extract_name_candidates(text: str) -> list[str]:
    """Extract capitalised name-like tokens from a piece of text, filtering common words."""
    out = []
    for m in _COMMENTARY_NAME_RE.finditer(text):
        candidate = m.group(1).strip()
        words = candidate.split()

        # Strip a leading filler word if the candidate is multi-word
        # (e.g. "And Sharma" -> "Sharma")
        if len(words) > 1 and words[0].lower() in _LEADING_FILLER_WORDS:
            words = words[1:]
            candidate = " ".join(words)

        if not words:
            continue
        if all(w.lower() not in _COMMON_NON_NAME_WORDS for w in words):
            out.append(candidate)
    return out


def _fuzzy_match(name: str, candidates: list[str], cutoff: float = 0.72) -> Optional[str]:
    """
    Return the best match for `name` among `candidates`, or None.

    Handles three cases:
      1. Exact match (case-insensitive)
      2. Surname-only commentary mention matches a full-name candidate
         (e.g. "Kohli" matches "Virat Kohli") — the most common real case,
         since commentators usually say surnames.
      3. Fuzzy string similarity for spelling variations (Whisper transcription noise)
    """
    if not candidates:
        return None

    name_l = name.strip().lower()
    name_words = set(name_l.split())

    # 1. Exact match
    for c in candidates:
        if c.strip().lower() == name_l:
            return c

    # 2. Surname/word-subset match: every word in `name` appears as a word
    #    in `c`, or vice versa (handles "Kohli" <-> "Virat Kohli")
    best_subset = None
    best_subset_len = 0
    for c in candidates:
        c_words = set(c.strip().lower().split())
        if name_words and (name_words <= c_words or c_words <= name_words):
            overlap = len(name_words & c_words)
            if overlap > best_subset_len:
                best_subset = c
                best_subset_len = overlap
    if best_subset:
        return best_subset

    # 3. Fuzzy similarity fallback (typos / ASR noise)
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def cross_reference_players(
    commentary_segments: list[dict],
    ocr_timeline: list[dict],
    roster: dict[str, list[str]],
    window_sec: float = 12.0,
) -> dict:
    """
    Cross-reference player names mentioned in commentary audio against names
    visible on the scoreboard (OCR) near the same timestamp.

    For each name-like token heard in commentary at time T, look at OCR
    snapshots within +/- window_sec of T and fuzzy-match the spoken name
    against scoreboard names. A match = "confirmed" (high confidence the
    name is correct and currently active in the match).

    Returns:
      {
        "confirmed": [names seen in both commentary AND nearby scoreboard],
        "ocr_only":  [names only ever seen on scoreboard],
        "commentary_only": [names only ever heard in commentary, unmatched],
        "mentions": [{"name": str, "timestamp": float, "matched_scoreboard_name": str|None}],
        "all": [confirmed + ocr_only]  -- the roster to give the LLM
      }
    """
    ocr_names_all = set(roster.get("all", []))

    mentions = []
    confirmed: set[str] = set()
    commentary_only: set[str] = set()

    for seg in commentary_segments:
        ts = float(seg.get("start", 0.0))
        text = seg.get("text", "")
        for candidate in _extract_name_candidates(text):
            # OCR snapshots within the time window around this commentary segment
            nearby_ocr_names: set[str] = set()
            for snap in ocr_timeline:
                if abs(snap.get("timestamp_sec", -1e9) - ts) <= window_sec:
                    for b in snap.get("batters", []):
                        n = b.get("name", "").strip()
                        if n:
                            nearby_ocr_names.add(n)

            match = _fuzzy_match(candidate, list(nearby_ocr_names)) or _fuzzy_match(candidate, list(ocr_names_all))

            if match:
                confirmed.add(match)
            else:
                commentary_only.add(candidate)

            mentions.append({
                "name": candidate,
                "timestamp": ts,
                "matched_scoreboard_name": match,
            })

    ocr_only = ocr_names_all - confirmed

    result = {
        "confirmed":        sorted(confirmed),
        "ocr_only":         sorted(ocr_only),
        "commentary_only":  sorted(commentary_only),
        "mentions":         mentions,
        "all":              sorted(confirmed | ocr_only),
    }
    logger.info(
        "Cross-reference roster — confirmed: %s | ocr_only: %s | commentary_only(unmatched): %s",
        result["confirmed"], result["ocr_only"], result["commentary_only"]
    )
    return result


# ─────────────────────────────────────────────
# Stage 4c: Detect wicket events from scorecard deltas
# ─────────────────────────────────────────────

def detect_wicket_events_from_ocr(ocr_timeline: list[dict]) -> list[dict]:
    """
    Compare consecutive scorecard snapshots. A wicket is signalled when:
      - wicket count increases, OR
      - a batter name disappears and a new one appears
    Returns synthetic event dicts suitable for merging with LLM events.
    """
    wicket_events = []
    prev = None

    for snap in ocr_timeline:
        if prev is None:
            prev = snap
            continue

        prev_w  = prev.get("wickets")
        curr_w  = snap.get("wickets")
        prev_ts = prev["timestamp_sec"]
        curr_ts = snap["timestamp_sec"]

        wicket_detected   = False
        dismissed_player  = "Unknown"
        incoming_player   = "Unknown"

        # ── Signal 1: wicket count increased ──
        if prev_w is not None and curr_w is not None and curr_w > prev_w:
            wicket_detected = True
            # Try to identify who got out by comparing batter lists
            prev_names = {b["name"] for b in prev.get("batters", [])}
            curr_names = {b["name"] for b in snap.get("batters", [])}
            gone_names = prev_names - curr_names
            new_names  = curr_names - prev_names
            if gone_names:
                dismissed_player = list(gone_names)[0]
            if new_names:
                incoming_player = list(new_names)[0]

        # ── Signal 2: batter lineup changed even if wicket count not parsed ──
        elif prev.get("batters") and snap.get("batters"):
            prev_names = {b["name"] for b in prev["batters"]}
            curr_names = {b["name"] for b in snap["batters"]}
            gone = prev_names - curr_names
            new  = curr_names - prev_names
            # Only flag as wicket if exactly one player replaced
            if len(gone) == 1 and len(new) == 1:
                wicket_detected  = True
                dismissed_player = list(gone)[0]
                incoming_player  = list(new)[0]

        if wicket_detected:
            # Use midpoint between snapshots as event timestamp
            event_ts = (prev_ts + curr_ts) / 2.0
            caption  = f"WICKET! {dismissed_player} is OUT!"
            wicket_events.append({
                "event_type":    "Wicket",
                "timestamp_sec": event_ts,
                "player":        dismissed_player,
                "incoming":      incoming_player,
                "confidence":    0.93,
                "caption":       caption,
                "reasoning":     (
                    f"Scorecard delta at {prev_ts:.0f}s→{curr_ts:.0f}s: "
                    f"wickets {prev_w}→{curr_w}, "
                    f"dismissed={dismissed_player}, incoming={incoming_player}"
                ),
                "source":        "ocr_delta",
            })

        prev = snap

    logger.info("Scorecard-delta wicket events: %d", len(wicket_events))
    return wicket_events


# ─────────────────────────────────────────────
# Stage 5: Multi-modal event detection via LLM (vLLM on AMD GPU)
# ─────────────────────────────────────────────

def detect_events_llm(
    commentary_segments: list[dict],
    crowd_events:        list[dict],
    ocr_timeline:        list[dict],
    roster:              dict,
    frames:              list[str],
    selected_events:     list[str],
    confidence_threshold:float,
    video_duration:      float,
    cross_ref:           Optional[dict] = None,
) -> list[dict]:
    """
    Send multi-modal signals to vLLM on AMD MI300X for event detection.
    Uses the extracted roster (no hardcoded names).
    Raises RuntimeError if GPU server is unavailable.
    """
    commentary_text = "\n".join(
        f"[{s['start']:.1f}s] {s['text']}" for s in commentary_segments[:80]
    )
    crowd_text = ", ".join(f"{e['timestamp']:.1f}s" for e in crowd_events[:25])

    # Compact OCR timeline (score snapshots only, not full text)
    ocr_summary_lines = []
    for snap in ocr_timeline[:30]:
        parts = []
        if snap.get("score") is not None:
            parts.append(f"score={snap['score']}/{snap['wickets']} ({snap['overs']}ov)")
        for b in snap.get("batters", [])[:2]:
            parts.append(f"{b['name']} {b['runs']}({b['balls']})" + ("*" if b.get("striker") else ""))
        if parts:
            ocr_summary_lines.append(f"[{snap['timestamp_sec']:.0f}s] {' | '.join(parts)}")
    ocr_text = "\n".join(ocr_summary_lines)

    if cross_ref and (cross_ref.get("confirmed") or cross_ref.get("ocr_only")):
        confirmed_text = ", ".join(cross_ref.get("confirmed", [])[:20]) or "none"
        ocr_only_text  = ", ".join(cross_ref.get("ocr_only", [])[:20]) or "none"
        roster_text = (
            f"CONFIRMED (heard in commentary AND seen on scoreboard — prefer these): {confirmed_text}\n"
            f"SCOREBOARD-ONLY (seen on scoreboard, not yet confirmed by commentary): {ocr_only_text}"
        )
    else:
        roster_text = ", ".join(roster.get("all", [])[:30]) or "Names not extracted — use \"Unknown\""

    # Vision: analyse a selection of frames spread over video
    frame_analyses = []
    sample_count   = min(8, len(frames))
    indices        = [int(len(frames) * p / sample_count) for p in range(sample_count)]
    for idx in indices:
        if idx >= len(frames):
            continue
        try:
            b64  = _image_to_b64(frames[idx])
            msgs = [
                {"role": "system", "content": "You are a cricket video analyst. One-line description of the cricket event in this frame: event type, player if identifiable, action."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Describe the cricket event visible."},
                ]},
            ]
            desc = _llm(msgs, max_tokens=60)
            ts_approx = idx / max(len(frames), 1) * video_duration
            frame_analyses.append(f"[{ts_approx:.0f}s] {desc}")
        except Exception as ex:
            logger.debug("Frame %d vision skip: %s", idx, ex)

    frame_context = "\n".join(frame_analyses) or "Visual analysis unavailable."

    system_prompt = (
        "You are an expert cricket highlight detector for the FlashBoundary system running on AMD MI300X GPU.\n"
        "Analyze multi-modal signals to detect key cricket events.\n"
        "CRITICAL: Respond with ONLY a JSON array. Do NOT use <think> tags, reasoning, "
        "markdown fences, or any text before or after the array. "
        "The very first character of your response must be '['.\n"
        "Each object: event_type, timestamp_sec, player, confidence, caption, reasoning."
    )

    user_prompt = f"""VIDEO DURATION: {video_duration:.1f}s
REQUESTED EVENT TYPES: {', '.join(selected_events)}
MINIMUM CONFIDENCE: {confidence_threshold}

PLAYER NAMES (extracted live from this video's commentary audio + scoreboard OCR — these are the ONLY real names available, no hardcoded roster exists):
{roster_text}

COMMENTARY TRANSCRIPT (timestamped):
{commentary_text}

CROWD ENERGY PEAKS (seconds from start):
{crowd_text}

SCORECARD OCR TIMELINE (score / batting pairs at each timestamp):
{ocr_text}

VISUAL FRAME ANALYSIS:
{frame_context}

RULES:
1. Prefer CONFIRMED names for the "player" field. Use SCOREBOARD-ONLY names only if commentary at that timestamp clearly refers to that batter (e.g. matching score/over). If no name from either list fits, use "Unknown" — do NOT invent or guess a name not listed above.
2. Only detect events of types: {', '.join(selected_events)}.
3. Only include events with confidence >= {confidence_threshold}.
4. Events must be at least 10 seconds apart globally. Prefer strong multi-signal agreement.
5. For Wicket events: prefer scorecard delta evidence (wicket count increase, batter change) over commentary alone.
6. Aim for 6–16 events across the full {video_duration:.1f}s match.
7. caption must be ≤ 10 words, punchy, suitable for video overlay.

Return ONLY a JSON array:
[{{"event_type": "Six", "timestamp_sec": 45.2, "player": "<name from the lists above or Unknown>", "confidence": 0.92, "caption": "<punchy caption>", "reasoning": "Commentary + crowd peak"}}]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # This will raise RuntimeError if GPU is unavailable (intentional — no fallback)
    raw = _llm(messages, max_tokens=4000)
    events = _parse_llm_event_json(raw)

    if events is None:
        # Retry once with a stricter instruction — helps when the model
        # wrapped output in <think> tags or added commentary.
        logger.warning("First LLM response unparsable, retrying with stricter prompt")
        retry_messages = messages + [
            {"role": "assistant", "content": raw[:200]},
            {"role": "user", "content": (
                "Your previous response was not valid JSON. "
                "Reply with ONLY the JSON array, starting with '[' and ending with ']'. "
                "No <think> tags, no markdown, no commentary, no explanation."
            )},
        ]
        raw_retry = _llm(retry_messages, max_tokens=4000)
        events = _parse_llm_event_json(raw_retry)

    if events is None:
        logger.error("LLM event detection: could not parse JSON after retry. Raw (first 500 chars): %r", raw[:500])
        events = []

    logger.info("LLM detected %d raw events", len(events))
    return events


def _parse_llm_event_json(raw: str) -> Optional[list[dict]]:
    """
    Robustly parse a JSON array of events out of raw LLM text.
    Handles: <think>...</think> reasoning preambles, markdown fences,
    leading/trailing commentary, and truncated arrays (closes them off).
    Returns None if no array could be parsed.
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip <think>...</think> reasoning blocks (Qwen3 / reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Locate the first '[' — that's where the array should start
    start = text.find("[")
    if start == -1:
        return None

    candidate = text[start:]

    # Try direct parse first
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Bracket-matched extraction: walk the string tracking [ ] depth,
    # accounting for strings so brackets inside strings don't confuse us.
    depth = 0
    in_string = False
    escape = False
    end_idx = None
    for i, ch in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx is not None:
        try:
            parsed = json.loads(candidate[: end_idx + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Truncated array (max_tokens cut it off mid-object): trim back to the
    # last complete object "},\n" or "}" and close the array ourselves.
    last_close = candidate.rfind("}")
    if last_close != -1:
        truncated = candidate[: last_close + 1] + "]"
        try:
            parsed = json.loads(truncated)
            if isinstance(parsed, list):
                logger.warning("LLM JSON was truncated — recovered %d complete event(s)", len(parsed))
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────
# Event deduplication / gap enforcement
# ─────────────────────────────────────────────

def deduplicate_events(events: list[dict]) -> list[dict]:
    """
    Enforce minimum gap between events of the same type AND globally.
    Keeps higher-confidence events when two are too close together.
    Also prevents two different event types at the same timestamp (no combined events).
    """
    # Sort by confidence desc so we keep the best when colliding
    events_sorted = sorted(events, key=lambda x: float(x.get("confidence", 0)), reverse=True)

    kept: list[dict] = []

    for ev in events_sorted:
        ts      = float(ev.get("timestamp_sec", 0))
        ev_type = ev.get("event_type", "")
        min_gap_type   = MIN_EVENT_GAP_SEC.get(ev_type, 15.0)

        # Check global minimum gap (any event type)
        too_close_global = any(
            abs(ts - float(k.get("timestamp_sec", 0))) < GLOBAL_MIN_GAP_SEC
            for k in kept
        )
        # Check per-type minimum gap
        too_close_type = any(
            k.get("event_type") == ev_type and abs(ts - float(k.get("timestamp_sec", 0))) < min_gap_type
            for k in kept
        )

        if not too_close_global and not too_close_type:
            kept.append(ev)

    # Final sort by timestamp
    kept.sort(key=lambda x: float(x.get("timestamp_sec", 0)))
    logger.info("After dedup: %d events (from %d raw)", len(kept), len(events))
    return kept


# ─────────────────────────────────────────────
# Stage 6: Thumbnail generation
# ─────────────────────────────────────────────

def generate_thumbnail(frame_path: str, caption: str, out_path: str) -> str:
    if not _ffmpeg_available() or not frame_path or not Path(frame_path).exists():
        return frame_path or ""
    safe = caption.replace("'", "").replace(":", "-").replace("!", "")
    cmd = [
        "ffmpeg", "-i", frame_path,
        "-vf", (
            f"drawtext=text='{safe}':"
            f"fontcolor=white:fontsize={CAPTION_FONT_SIZE}:"
            "box=1:boxcolor=black@0.6:"
            "x=(w-text_w)/2:y=h-th-20"
        ),
        out_path, "-y",
    ]
    result = _run(cmd)
    return out_path if result.returncode == 0 else frame_path


# ─────────────────────────────────────────────
# Stage 7: Video segment cutting (event-duration-aware)
# ─────────────────────────────────────────────

def cut_segment(
    video_path:    str,
    timestamp:     float,
    event_type:    str,
    caption:       str,
    out_path:      str,
    video_duration: float,
) -> str:
    """
    Cut a clip for one event. Duration is determined by event type.
    Pre-roll and post-roll are added around the action duration.
    Returns "" on failure (logged via logger.warning — caller should
    surface this through its own log() if segment_paths ends up empty).
    """
    if not _ffmpeg_available():
        logger.warning("cut_segment: ffmpeg unavailable")
        return ""

    action_dur = EVENT_DURATIONS.get(event_type, DEFAULT_DURATION)
    start      = max(0.0, timestamp - PRE_ROLL_SEC)
    total_dur  = PRE_ROLL_SEC + action_dur + POST_ROLL_SEC
    # Don't exceed video
    total_dur  = min(total_dur, max(0.5, video_duration - start))

    if total_dur <= 0:
        logger.warning(
            "cut_segment: skip ts=%.1f — start=%.1f >= video_duration=%.1f",
            timestamp, start, video_duration
        )
        return ""

    safe = caption.replace("'", "").replace(":", "-").replace("!", "")
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(total_dur),
        "-vf", (
            f"drawtext=text='{safe}':"
            f"fontcolor=white:fontsize={CAPTION_FONT_SIZE}:"
            "box=1:boxcolor=black@0.55:"
            "x=(w-text_w)/2:y=h-th-30"
        ),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        out_path, "-y",
    ]
    result = _run(cmd)
    if result.returncode != 0 or not Path(out_path).exists() or Path(out_path).stat().st_size < 1000:
        logger.warning("cut_segment failed ts=%.1f type=%s rc=%s: %s",
                        timestamp, event_type, result.returncode, result.stderr[-600:])
        return ""
    return out_path


# ─────────────────────────────────────────────
# Stage 8: Stitch final reel
# ─────────────────────────────────────────────

def stitch_reel(segment_paths: list[str], out_path: str) -> str:
    """
    Concatenate segment clips into the final reel.
    Tries fast stream-copy concat first; falls back to re-encoding concat
    if the segments have incompatible parameters (common with URL-sourced
    videos that have unusual codecs/timestamps).
    """
    valid = [p for p in segment_paths if p and Path(p).exists() and Path(p).stat().st_size > 2000]
    if not valid:
        logger.warning("stitch_reel: no valid segment files to stitch")
        return ""
    if not _ffmpeg_available():
        logger.warning("stitch_reel: ffmpeg unavailable")
        return ""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in valid:
            f.write(f"file '{p}'\n")
        concat_list = f.name

    # Attempt 1: fast stream copy
    cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", out_path, "-y"]
    result = _run(cmd)
    if result.returncode == 0 and Path(out_path).exists() and Path(out_path).stat().st_size > 2000:
        Path(concat_list).unlink(missing_ok=True)
        return out_path

    logger.warning("stitch_reel: stream-copy concat failed (rc=%s): %s — retrying with re-encode",
                    result.returncode, result.stderr[-400:])

    # Attempt 2: re-encode concat (handles mismatched codec params/timestamps)
    cmd2 = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c:v", "libx264", "-c:a", "aac", "-preset", "fast", out_path, "-y"]
    result2 = _run(cmd2)
    Path(concat_list).unlink(missing_ok=True)

    if result2.returncode != 0 or not Path(out_path).exists() or Path(out_path).stat().st_size < 2000:
        logger.error("stitch_reel: re-encode concat also failed (rc=%s): %s",
                      result2.returncode, result2.stderr[-600:])
        return ""
    return out_path


# ─────────────────────────────────────────────
# Agent definitions
# ─────────────────────────────────────────────

AGENTS = [
    {"name": "Supervisor Agent",     "role": "Orchestrates pipeline, validates GPU availability"},
    {"name": "Vision Agent",         "role": "Extracts frames at 1fps via ffmpeg"},
    {"name": "Audio Agent",          "role": "Whisper transcription on AMD GPU (faster-whisper)"},
    {"name": "OCR Agent",            "role": "Dense scorecard OCR — every 5s of video"},
    {"name": "Roster Agent",         "role": "Builds player roster from OCR — no hardcoded names"},
    {"name": "Wicket Delta Agent",   "role": "Detects wickets via scorecard delta (score/batter change)"},
    {"name": "Crowd Signal Agent",   "role": "Audio RMS energy peak detection"},
    {"name": "Decision Agent",       "role": "Multi-modal LLM fusion on AMD MI300X vLLM"},
    {"name": "Dedup Agent",          "role": "Enforces event gaps, removes duplicate detections"},
    {"name": "Packaging Agent",      "role": "Event-duration-aware clip cutting + caption overlay"},
    {"name": "Thumbnail Agent",      "role": "Thumbnail generation per event"},
    {"name": "Reel Agent",           "role": "Concatenates clips into final highlight reel"},
]


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

# ── Pipeline stage weights for progress/ETA estimation ──
# Approximate relative cost of each stage (sums to 1.0). Used to compute
# elapsed-time-based ETA: eta = elapsed/done_weight * (1 - done_weight)
STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("Supervisor: GPU check",        0.01),
    ("Supervisor: load video",       0.02),
    ("Vision: frame extraction",     0.10),
    ("Audio: Whisper transcription", 0.22),
    ("Crowd signal analysis",        0.06),
    ("OCR: scorecard timeline",      0.19),
    ("Roster extraction",            0.02),
    ("Cross-reference players",      0.02),
    ("Wicket delta detection",       0.03),
    ("Decision: LLM fusion (GPU)",   0.19),
    ("Deduplication",                0.01),
    ("Packaging: clip cutting",      0.10),
    ("Thumbnails",                   0.01),
    ("Reel: stitching",              0.02),
]


def generate_highlights(
    source_input:         str,
    selected_events:      list[str],
    highlight_mode:       str = "Match Highlights",
    player_name:          Optional[str] = None,
    confidence_threshold: float = 0.70,
    progress_callback=None,
) -> tuple[dict, list[dict], list[str], pd.DataFrame]:
    """
    Full multi-modal pipeline.
    Raises RuntimeError immediately if GPU/vLLM is unreachable.
    Returns: (payload, agents, logs, analytics_df)

    progress_callback: optional callable(stage_label: str, fraction_done: float, elapsed: float)
    called after each stage completes. fraction_done is cumulative (0..1) based
    on STAGE_WEIGHTS, allowing the UI to estimate remaining time as:
        eta_sec = elapsed * (1 - fraction_done) / fraction_done
    """
    logs:   list[str]  = []
    agents: list[dict] = []
    _t_start = time.time()
    _cum_weight = 0.0
    _weight_iter = iter(STAGE_WEIGHTS)

    def _tick(label: str | None = None):
        """Advance to the next stage weight and report progress."""
        nonlocal _cum_weight
        try:
            default_label, w = next(_weight_iter)
        except StopIteration:
            w = 0.0
            default_label = label or "Finishing"
        _cum_weight = min(1.0, _cum_weight + w)
        if progress_callback:
            elapsed = time.time() - _t_start
            progress_callback(label or default_label, _cum_weight, elapsed)

    def log(msg: str):
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        logs.append(entry)
        logger.info(msg)

    def agent_ok(name: str, detail: str):
        agents.append({"name": f"✅ {name}", "detail": detail})
        log(f"AGENT {name}: {detail}")

    def agent_run(name: str, detail: str):
        agents.append({"name": f"⚙️ {name}", "detail": detail})
        log(f"AGENT {name}: {detail}")

    # ── GPU health check FIRST — fail fast ──
    online, gpu_msg = check_vllm_health()
    if not online:
        raise RuntimeError(
            f"AMD MI300X vLLM server is OFFLINE at {VLLM_BASE_URL}.\n"
            f"Reason: {gpu_msg}\n\n"
            f"Start the server with:\n"
            f"  VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve {VLLM_MODEL} \\\n"
            f"    --api-key abc-123 --port 8000 --enable-auto-tool-choice\n\n"
            "FlashBoundary requires the AMD GPU for multi-modal event detection."
        )
    agent_ok("Supervisor Agent", f"GPU online: {gpu_msg}")
    _tick("Supervisor: GPU check")

    workdir = tempfile.mkdtemp(prefix="flashboundary_")
    log(f"Workdir: {workdir} | Events: {selected_events} | Threshold: {confidence_threshold}")

    # ── Download if URL ──
    video_path = source_input
    if source_input.startswith("http"):
        agent_run("Supervisor Agent", "Downloading video from URL…")
        video_file = os.path.join(workdir, "input.mp4")
        with requests.get(source_input, stream=True, timeout=300, allow_redirects=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "text/html" in content_type or "application/json" in content_type:
                raise RuntimeError(
                    f"The URL did not return a video file (Content-Type: '{content_type}'). "
                    f"This usually means the link points to a webpage (e.g. a YouTube/Drive "
                    f"share page) rather than a direct, downloadable video file. "
                    f"Use a direct .mp4/.mov link, or upload the file instead."
                )
            with open(video_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    f.write(chunk)
        video_path = video_file

        size_bytes = Path(video_path).stat().st_size
        log(f"Downloaded: {video_path} ({size_bytes/1e6:.1f} MB, Content-Type: {content_type})")
        if size_bytes < 10_000:
            raise RuntimeError(
                f"Downloaded file is only {size_bytes} bytes — too small to be a real video. "
                f"Check that the URL points directly to the video file."
            )

    video_duration = _get_video_duration(video_path)
    if video_duration <= 0 or video_path == "" or not Path(video_path).exists():
        raise RuntimeError(
            f"Could not read a valid duration from the input video at '{video_path}'. "
            f"The file may be corrupted, unsupported, or not a video (ffprobe failed)."
        )
    log(f"Duration: {video_duration:.1f}s")
    agent_ok("Supervisor Agent", f"Video ready — {video_duration:.0f}s duration")
    _tick("Supervisor: load video")

    # ── Stage 1: Frame extraction ──
    agent_run("Vision Agent", "Extracting frames @ 1fps…")
    frames = extract_frames(video_path, workdir, fps=1.0)
    log(f"Frames: {len(frames)}")
    agent_ok("Vision Agent", f"Extracted {len(frames)} frames")
    _tick("Vision: frame extraction")

    # ── Stage 2: Audio transcription ──
    agent_run("Audio Agent", "Transcribing commentary via Whisper on AMD GPU…")
    commentary_segments = transcribe_audio(video_path, workdir)
    log(f"Commentary segments: {len(commentary_segments)}")
    agent_ok("Audio Agent", f"Transcribed {len(commentary_segments)} commentary segments")
    _tick("Audio: Whisper transcription")

    # ── Stage 3: Crowd energy ──
    agent_run("Crowd Signal Agent", "Analysing crowd audio energy…")
    crowd_events = detect_crowd_energy(video_path, workdir)
    log(f"Crowd peaks: {len(crowd_events)}")
    agent_ok("Crowd Signal Agent", f"Detected {len(crowd_events)} crowd energy peaks")
    _tick("Crowd signal analysis")

    # ── Stage 4a: Dense OCR timeline ──
    agent_run("OCR Agent", f"Running dense scorecard OCR across {len(frames)} frames…")
    ocr_timeline = ocr_frames_full(frames, video_duration)
    log(f"OCR snapshots: {len(ocr_timeline)}")
    agent_ok("OCR Agent", f"Built {len(ocr_timeline)} scorecard snapshots")
    _tick("OCR: scorecard timeline")

    # ── Stage 4b: Player roster extraction from scoreboard OCR ──
    agent_run("Roster Agent", "Extracting player names from scoreboard OCR…")
    ocr_roster = extract_player_roster(ocr_timeline)
    log(f"OCR roster: {ocr_roster['all']}")
    agent_ok("Roster Agent", f"OCR roster: {', '.join(ocr_roster['all'][:8])}{'…' if len(ocr_roster['all']) > 8 else ''}")
    _tick("Roster extraction")

    # ── Stage 4b-2: Cross-reference commentary audio with scoreboard OCR ──
    agent_run("Cross-Ref Agent", "Cross-referencing commentary names with scoreboard OCR…")
    cross_ref = cross_reference_players(commentary_segments, ocr_timeline, ocr_roster)
    roster = {"all": cross_ref["all"], "batting": cross_ref["all"], "bowling": []}
    log(f"Confirmed players (audio+scoreboard agree): {cross_ref['confirmed']}")
    log(f"OCR-only players (scoreboard, unconfirmed by audio): {cross_ref['ocr_only']}")
    agent_ok(
        "Cross-Ref Agent",
        f"Confirmed {len(cross_ref['confirmed'])} player(s) via audio+scoreboard match "
        f"({', '.join(cross_ref['confirmed'][:6])}{'…' if len(cross_ref['confirmed']) > 6 else ''})"
    )
    _tick("Cross-reference players")

    # ── Stage 4c: Wicket detection via scorecard delta ──
    ocr_wicket_events: list[dict] = []
    if "Wicket" in selected_events:
        agent_run("Wicket Delta Agent", "Detecting wickets via scorecard delta analysis…")
        ocr_wicket_events = detect_wicket_events_from_ocr(ocr_timeline)
        log(f"OCR-delta wicket events: {len(ocr_wicket_events)}")
        agent_ok("Wicket Delta Agent", f"Found {len(ocr_wicket_events)} wicket events from scorecard delta")
    _tick("Wicket delta detection")

    # ── Stage 5: LLM multi-modal fusion ──
    agent_run("Decision Agent", f"Fusing signals via {VLLM_MODEL} on AMD MI300X…")
    llm_events = detect_events_llm(
        commentary_segments=commentary_segments,
        crowd_events=crowd_events,
        ocr_timeline=ocr_timeline,
        roster=roster,
        frames=frames,
        selected_events=selected_events,
        confidence_threshold=confidence_threshold,
        video_duration=video_duration,
        cross_ref=cross_ref,
    )
    log(f"LLM events: {len(llm_events)}")

    # ── Merge LLM + OCR wicket events ──
    all_raw_events = llm_events + ocr_wicket_events

    # ── Filter by type + player + confidence ──
    filtered: list[dict] = []
    for ev in all_raw_events:
        et   = ev.get("event_type", "")
        conf = float(ev.get("confidence", 0))
        if et not in selected_events:
            continue
        if conf < confidence_threshold:
            continue
        if player_name and highlight_mode == "Highlights by Player":
            if player_name.lower() not in ev.get("player", "").lower():
                continue
        filtered.append(ev)

    agent_ok("Decision Agent", f"Raw: {len(all_raw_events)} events detected")
    _tick("Decision: LLM fusion (GPU)")

    # ── Deduplication & gap enforcement ──
    agent_run("Dedup Agent", "Removing duplicate / too-close events…")
    final_events = deduplicate_events(filtered)
    log(f"Final events after dedup: {len(final_events)}")
    agent_ok("Dedup Agent", f"{len(final_events)} events after gap enforcement")
    _tick("Deduplication")

    # ── Stage 6/7: Cut segments + thumbnails ──
    agent_run("Packaging Agent", "Cutting event-duration-aware clips with caption overlays…")
    segments_dir  = _ensure_dir(os.path.join(workdir, "segments"))
    thumbs_dir    = _ensure_dir(os.path.join(workdir, "thumbnails"))
    segment_paths = []
    segments_meta = []

    for i, ev in enumerate(final_events):
        ts       = float(ev.get("timestamp_sec", 0))
        ev_type  = ev.get("event_type", "")
        caption  = ev.get("caption", ev_type)
        seg_path = os.path.join(segments_dir, f"seg_{i:03d}.mp4")

        seg_out = cut_segment(
            video_path=video_path,
            timestamp=ts,
            event_type=ev_type,
            caption=caption,
            out_path=seg_path,
            video_duration=video_duration,
        )

        # Thumbnail: frame nearest to event timestamp
        frame_idx  = min(int(ts * len(frames) / max(video_duration, 1)), len(frames) - 1) if frames else -1
        frame_path = frames[frame_idx] if frame_idx >= 0 else ""
        thumb_path = os.path.join(thumbs_dir, f"thumb_{i:03d}.jpg")
        thumb_out  = generate_thumbnail(frame_path, caption, thumb_path) if frame_path else ""

        if seg_out:
            segment_paths.append(seg_out)
        else:
            log(f"⚠ Segment cut FAILED for event #{i} ({ev_type} @ {ts:.1f}s) — see server logs for ffmpeg error")

        clip_dur = PRE_ROLL_SEC + EVENT_DURATIONS.get(ev_type, DEFAULT_DURATION) + POST_ROLL_SEC
        segments_meta.append({
            "event_type":     ev_type,
            "timestamp":      f"{int(ts // 60)}:{int(ts % 60):02d}",
            "player":         ev.get("player", "Unknown"),
            "confidence":     f"{float(ev.get('confidence', 0)):.0%}",
            "caption":        caption,
            "clip_duration":  f"{clip_dur:.1f}s",
            "segment_path":   seg_out,
            "thumbnail_path": thumb_out,
            "reasoning":      ev.get("reasoning", ""),
            "source":         ev.get("source", "llm"),
        })

    log(f"Segments cut: {len(segment_paths)}")
    _tick("Packaging: clip cutting")
    agent_ok("Thumbnail Agent", f"Thumbnails generated for {len(segments_meta)} events")
    _tick("Thumbnails")

    # ── Stage 8: Stitch reel ──
    agent_run("Reel Agent", "Stitching final highlight reel…")
    reel_path  = os.path.join(workdir, "final_highlight_reel.mp4")

    if not final_events:
        final_reel = ""
        log("⚠ No events detected after filtering/dedup — no reel to generate. "
            "Try lowering the confidence threshold or selecting more event types.")
        agent_ok("Reel Agent", "No events detected — no reel generated")
    elif not segment_paths:
        final_reel = ""
        log(f"⚠ {len(final_events)} event(s) detected, but ALL {len(final_events)} "
            f"segment cuts failed (ffmpeg errors — check server logs for "
            f"'cut_segment failed'). No reel could be generated.")
        agent_ok("Reel Agent", "All segment cuts failed — no reel generated")
    else:
        final_reel = stitch_reel(segment_paths, reel_path)
        if not final_reel:
            log(f"⚠ {len(segment_paths)} segment(s) cut successfully, but stitching "
                f"into the final reel failed (ffmpeg concat error — check server logs "
                f"for 'stitch_reel'). Serving the first segment as a fallback preview.")
            final_reel = segment_paths[0]
            agent_ok("Reel Agent", f"Stitch failed — serving 1 of {len(segment_paths)} clips as preview")
        else:
            agent_ok("Reel Agent", f"Reel ready: {len(segments_meta)} events, {len(segment_paths)} clips compiled")

    log(f"Final reel: {final_reel or '(none)'}")
    _tick("Reel: stitching")

    # ── Analytics ──
    analytics_rows = []
    for ev in final_events:
        analytics_rows.append({
            "Event Type":        ev.get("event_type"),
            "Timestamp (s)":     round(float(ev.get("timestamp_sec", 0)), 1),
            "Player":            ev.get("player", "Unknown"),
            "Confidence":        round(float(ev.get("confidence", 0)), 2),
            "Clip Duration (s)": PRE_ROLL_SEC + EVENT_DURATIONS.get(ev.get("event_type", ""), DEFAULT_DURATION) + POST_ROLL_SEC,
            "Caption":           ev.get("caption", ""),
            "Source":            ev.get("source", "llm"),
        })
    analytics_df = pd.DataFrame(analytics_rows)

    payload = {
        "message":          f"✅ Generated reel with {len(segments_meta)} highlight events.",
        "final_video_path": final_reel,
        "segments":         segments_meta,
        "total_events":     len(segments_meta),
        "video_duration_sec": video_duration,
        "roster":           roster["all"],
        "roster_confirmed": cross_ref["confirmed"],
        "roster_ocr_only":  cross_ref["ocr_only"],
        "roster_unmatched_commentary": cross_ref["commentary_only"],
        "workdir":          workdir,
    }

    return payload, agents, logs, analytics_df
