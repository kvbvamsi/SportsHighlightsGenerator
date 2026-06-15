"""
app.py — FlashBoundary | Real-time Cricket Highlight Generator
AMD MI300X GPU required — no fallback, clear error if GPU unavailable
"""

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from backend import (
    generate_highlights,
    check_vllm_health,
    detect_local_gpu,
    EVENT_DURATIONS,
    MIN_EVENT_GAP_SEC,
    PRE_ROLL_SEC,
    POST_ROLL_SEC,
    VLLM_BASE_URL,
    VLLM_MODEL,
)


# ─────────────────────────────────────────────
# Debug helpers
# ─────────────────────────────────────────────

def debug_print(msg: str) -> None:
    print(f"[APP] {time.strftime('%H:%M:%S')} | {msg}", flush=True)


def get_env_diagnostics() -> dict:
    info = {
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "ffmpeg": False,
        "tesseract": False,
        "vllm_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL,
    }
    import shutil
    info["ffmpeg"]    = shutil.which("ffmpeg") is not None
    info["tesseract"] = shutil.which("tesseract") is not None
    for mod in ["faster_whisper", "whisper", "easyocr", "pytesseract", "soundfile", "scipy"]:
        try:
            __import__(mod)
            info[f"mod_{mod}"] = True
        except Exception:
            info[f"mod_{mod}"] = False
    return info


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False


def save_uploaded_file(uploaded_file) -> str:
    suffix   = Path(uploaded_file.name).suffix or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="fb_upload_"))
    fp       = temp_dir / f"video{suffix}"
    with open(fp, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_print(f"Saved {fp} ({fp.stat().st_size/1e6:.1f} MB)")
    return str(fp)


def reset_state() -> None:
    for k, v in [
        ("payload", {}), ("agents", []), ("logs", []),
        ("analytics", pd.DataFrame()), ("video_url", ""),
        ("last_error", ""), ("input_source", ""),
        ("last_traceback", ""), ("debug_info", {}),
    ]:
        st.session_state[k] = v


def render_segment_table(segments: list[dict]):
    if not segments:
        st.info("No event segments available.")
        return
    data = [{
        "Event":       s.get("event_type", ""),
        "Time":        s.get("timestamp", ""),
        "Player":      s.get("player", ""),
        "Conf":        s.get("confidence", ""),
        "Clip Length": s.get("clip_duration", ""),
        "Caption":     s.get("caption", ""),
        "Source":      s.get("source", ""),
    } for s in segments]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────

debug_print("Starting FlashBoundary")

st.set_page_config(
    page_title="FlashBoundary | Cricket AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(160deg, #0d0f14 0%, #111827 60%, #0a0d12 100%); }

  .hero {
    background: linear-gradient(135deg, #1a2332 0%, #0f1923 50%, #1a1a2e 100%);
    border: 1px solid #22c55e33;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #22c55e, #16a34a, #4ade80, #22c55e);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
  }
  @keyframes shimmer { to { background-position: 200% center; } }
  .hero-title { font-size: 2.4rem; font-weight: 800; color: #f0fdf4; margin: 0 0 0.3rem; letter-spacing: -0.02em; }
  .hero-sub   { font-size: 0.92rem; color: #86efac; opacity: 0.85; }

  /* GPU status banner */
  .gpu-online  { background:#052e16; border:1px solid #16a34a; border-radius:8px; padding:0.5rem 1rem; color:#4ade80; font-weight:600; font-size:0.88rem; }
  .gpu-offline { background:#450a0a; border:1px solid #dc2626; border-radius:8px; padding:0.5rem 1rem; color:#fca5a5; font-weight:600; font-size:0.88rem; }
  .gpu-error   { background:#1c0a00; border:1px solid #ea580c; border-radius:12px; padding:1.2rem 1.5rem; color:#fed7aa; font-size:0.92rem; line-height:1.6; }

  /* Event type badges */
  .badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; margin:2px; }
  .b-six  { background:#14532d; color:#4ade80; }
  .b-four { background:#1e3a5f; color:#60a5fa; }
  .b-wicket { background:#450a0a; color:#f87171; }
  .b-milestone { background:#451a03; color:#fb923c; }
  .b-celebration { background:#2e1065; color:#c084fc; }

  /* Duration chips */
  .dur-chip { display:inline-block; background:#1e293b; border:1px solid #334155; border-radius:6px;
              padding:3px 8px; font-size:0.72rem; color:#94a3b8; margin:2px; }

  /* Log */
  .log-box { background:#0a0d12; border:1px solid #1e293b; border-radius:8px; padding:1rem;
             font-family:'Courier New',monospace; font-size:0.76rem; color:#86efac;
             max-height:260px; overflow-y:auto; }

  /* Roster */
  .roster-chip { display:inline-block; background:#0f2231; border:1px solid #1e4060; border-radius:999px;
                 padding:3px 10px; font-size:0.76rem; color:#7dd3fc; margin:3px; }

  /* Tabs */
  .stTabs [data-baseweb="tab"]          { font-size:0.875rem; font-weight:600; color:#94a3b8; }
  .stTabs [aria-selected="true"]        { color:#22c55e !important; border-bottom-color:#22c55e !important; }

  section[data-testid="stSidebar"] { background:#0f1419 !important; border-right:1px solid #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# Session state init
for k, v in [
    ("payload", {}), ("agents", []), ("logs", []),
    ("analytics", pd.DataFrame()), ("video_url", ""),
    ("last_error", ""), ("input_source", ""),
    ("last_traceback", ""), ("debug_info", {}),
]:
    if k not in st.session_state:
        st.session_state[k] = v

st.session_state.debug_info["env"] = get_env_diagnostics()


# ─────────────────────────────────────────────
# GPU Health Check (cached 30s)
# ─────────────────────────────────────────────

@st.cache_data(ttl=30)
def _gpu_status():
    return check_vllm_health()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Config")

    gpu_online, gpu_msg = _gpu_status()
    if gpu_online:
        st.markdown(f'<div class="gpu-online">🟢 AMD MI300X Online<br><small>{gpu_msg}</small></div>', unsafe_allow_html=True)
    else:
        local_gpu = detect_local_gpu()
        st.markdown(
            f'<div class="gpu-offline">🔴 GPU Offline — Generation blocked<br>'
            f'<small>{gpu_msg[:100]}</small></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Server: `{VLLM_BASE_URL}`")
        st.caption(f"Detected hardware: **{local_gpu}**")

    st.divider()

    highlight_mode = st.radio(
        "Highlight Mode",
        ["Match Highlights", "Highlights by Player"],
        index=0,
    )

    selected_events = st.multiselect(
        "Event Types",
        ["Six", "Four", "Wicket", "Player Milestone", "Celebration"],
        default=["Six", "Four", "Wicket"],
    )

    player_name = None
    if highlight_mode == "Highlights by Player":
        # Show roster if available
        roster = (st.session_state.payload or {}).get("roster", [])
        if roster:
            st.markdown("**Players detected from scoreboard:**")
            chips = "".join(f'<span class="roster-chip">{p}</span>' for p in roster[:15])
            st.markdown(chips, unsafe_allow_html=True)
            player_name = st.selectbox("Select Player", [""] + roster)
        else:
            player_name = st.text_input("Player Name", placeholder="Run pipeline first to auto-detect players")

    confidence_threshold = st.slider("Min Confidence", 0.50, 1.00, 0.70, 0.05)

    st.divider()
    st.markdown("### 🕐 Clip Durations")
    for ev_type, dur in EVENT_DURATIONS.items():
        total = PRE_ROLL_SEC + dur + POST_ROLL_SEC
        st.markdown(
            f'<span class="dur-chip">{ev_type}: {total:.0f}s</span>',
            unsafe_allow_html=True,
        )
    st.caption(f"Pre-roll: {PRE_ROLL_SEC}s · Post-roll: {POST_ROLL_SEC}s")

    st.divider()
    st.markdown("### 🚫 Min Event Gaps")
    for ev_type, gap in MIN_EVENT_GAP_SEC.items():
        st.markdown(f'<span class="dur-chip">{ev_type}: ≥{gap:.0f}s apart</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🧠 Pipeline Stages")
    for icon, stage in [
        ("🎞️", "Frame Extraction (1fps)"),
        ("🎙️", "Whisper Commentary (AMD GPU)"),
        ("📢", "Crowd Energy (RMS)"),
        ("📊", "Dense Scorecard OCR (every 5s)"),
        ("👥", "Player Roster Extraction"),
        ("🎯", "Wicket Delta Detection"),
        ("🤖", f"LLM Fusion ({VLLM_MODEL})"),
        ("🔁", "Gap Deduplication"),
        ("✂️", "Event-aware Clip Cutting"),
        ("🎬", "Reel Compilation"),
    ]:
        st.markdown(f"<small>{icon} {stage}</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-title">🏏 FlashBoundary</div>
  <div class="hero-sub">Real-time Cricket Highlight Generator · Multi-modal AI · AMD MI300X GPU Required</div>
</div>
""", unsafe_allow_html=True)

# ── GPU error banner (persistent, above tabs) ──
if not gpu_online:
    st.markdown(f"""
<div class="gpu-error">
  <strong>⛔ AMD MI300X GPU Server Required</strong><br><br>
  FlashBoundary's multi-modal event detection runs exclusively on the AMD MI300X via vLLM.
  No cloud fallback is used — all processing is on-device.<br><br>
  <strong>Start the vLLM server:</strong><br>
  <code>VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve {VLLM_MODEL} \\<br>
  &nbsp;&nbsp;--served-model-name {VLLM_MODEL} \\<br>
  &nbsp;&nbsp;--api-key abc-123 --port 8000 \\<br>
  &nbsp;&nbsp;--enable-auto-tool-choice --tool-call-parser hermes</code><br><br>
  Server expected at: <code>{VLLM_BASE_URL}</code><br>
  Error: <code>{gpu_msg}</code><br>
  Detected hardware on this host: <code>{detect_local_gpu()}</code>
</div>
""", unsafe_allow_html=True)
    st.markdown("")


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

home_tab, reel_tab, thumbs_tab, player_tab, agent_tab, analytics_tab, export_tab = st.tabs([
    "🏠 Home", "🎬 Final Reel", "🖼️ Thumbnails",
    "👤 Player View", "🤖 Agent Insights", "📈 Analytics", "⬇️ Exports",
])


# ─────────────────────────────────────────────
# Home tab
# ─────────────────────────────────────────────

with home_tab:
    st.subheader("Upload Match Video")

    up_col, url_col = st.columns([1, 1])
    with up_col:
        uploaded_file = st.file_uploader("Upload match video", type=["mp4", "mov", "mkv", "avi"])
    with url_col:
        video_url = st.text_input(
            "OR enter direct video URL",
            value=st.session_state.video_url,
            placeholder="https://example.com/match.mp4",
        )
        st.session_state.video_url = video_url

    if uploaded_file is not None:
        try:
            src = save_uploaded_file(uploaded_file)
            st.session_state.input_source = src
            st.success(f"✅ {uploaded_file.name} ready ({uploaded_file.size / 1e6:.1f} MB)")
            st.video(src)
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.error(f"File save failed: {exc}")

    elif video_url:
        if is_valid_url(video_url):
            st.session_state.input_source = video_url
            st.success("✅ Valid URL")
            try:
                st.video(video_url)
            except Exception:
                st.info("Preview unavailable — URL will still be processed.")
        else:
            st.error("Invalid URL.")

    st.divider()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode",       highlight_mode.replace("Highlights ", ""))
    c2.metric("Events",     len(selected_events))
    c3.metric("Confidence", f"{confidence_threshold:.0%}")
    c4.metric("GPU",        "🟢 Online" if gpu_online else "🔴 Offline")

    if selected_events:
        cls_map = {"Six": "b-six", "Four": "b-four", "Wicket": "b-wicket",
                   "Player Milestone": "b-milestone", "Celebration": "b-celebration"}
        badges = " ".join(
            '<span class="badge {}">{}</span>'.format(cls_map.get(e, "b-six"), e)
            for e in selected_events
        )
        st.markdown(f"**Detecting:** {badges}", unsafe_allow_html=True)

    st.divider()

    btn_col, reset_col = st.columns([2, 1])
    with btn_col:
        gen_label    = "🚀 Generate Highlight Reel"
        gen_disabled = not st.session_state.input_source or not gpu_online
        gen_tooltip  = "Start the vLLM GPU server first" if not gpu_online else ""

        if st.button(gen_label, type="primary", use_container_width=True,
                     disabled=gen_disabled, help=gen_tooltip):
            st.session_state.last_error     = ""
            st.session_state.last_traceback = ""
            src = st.session_state.input_source

            if not src:
                st.error("Upload a video or enter a URL first.")
            elif highlight_mode == "Highlights by Player" and not player_name:
                st.error("Enter or select a player name.")
            elif not selected_events:
                st.error("Select at least one event type.")
            else:
                progress   = st.progress(0.0)
                status_box = st.empty()
                status_box.info("**🚀 Starting pipeline…**")

                # ── Real-time progress + ETA via callback ──
                def _on_progress(stage_label: str, frac_done: float, elapsed: float):
                    frac_done = max(min(frac_done, 0.999), 0.001)
                    progress.progress(frac_done)
                    if frac_done < 0.05:
                        eta_str = "estimating…"
                    else:
                        eta_sec = elapsed * (1 - frac_done) / frac_done
                        eta_str = f"~{eta_sec:0.0f}s remaining" if eta_sec >= 1 else "almost done"
                    status_box.info(
                        f"**{stage_label}…**  \n"
                        f"Elapsed: {elapsed:0.0f}s · {eta_str} · {frac_done:.0%} complete"
                    )

                try:
                    t0 = time.time()
                    payload, agents, logs, analytics = generate_highlights(
                        source_input=src,
                        selected_events=selected_events,
                        highlight_mode=highlight_mode,
                        player_name=player_name or None,
                        confidence_threshold=confidence_threshold,
                        progress_callback=_on_progress,
                    )
                    elapsed = time.time() - t0

                    st.session_state.payload   = payload
                    st.session_state.agents    = agents
                    st.session_state.logs      = logs
                    st.session_state.analytics = analytics
                    st.session_state.debug_info["elapsed_sec"] = round(elapsed, 2)

                    progress.progress(1.0)
                    status_box.success(
                        f"✅ {payload.get('message', 'Done!')} — {elapsed:.1f}s · "
                        f"Roster: {len(payload.get('roster', []))} players detected"
                    )

                except RuntimeError as exc:
                    # GPU unavailable, missing dependency, or no usable signal — show clear error, no fallback
                    tb = traceback.format_exc()
                    st.session_state.last_error     = str(exc)
                    st.session_state.last_traceback = tb
                    progress.empty()
                    status_box.empty()
                    st.markdown(f"""
<div class="gpu-error">
  <strong>⛔ Pipeline Stopped — Real Data Required</strong><br><br>
  {str(exc).replace(chr(10), '<br>')}
  <br><br>
  <small>FlashBoundary does not use synthetic/demo data — if a required
  component (Whisper, OCR, audio analysis, GPU) is unavailable, the
  pipeline stops here rather than producing fake highlights.</small>
</div>
""", unsafe_allow_html=True)

                except Exception as exc:
                    tb = traceback.format_exc()
                    st.session_state.last_error     = str(exc)
                    st.session_state.last_traceback = tb
                    progress.empty()
                    status_box.error(f"Pipeline error: {exc}")

    with reset_col:
        if st.button("🔄 Reset", use_container_width=True):
            reset_state()
            st.rerun()

    if st.session_state.last_error and "GPU Error" not in st.session_state.last_error:
        st.warning(st.session_state.last_error)
    if st.session_state.last_traceback:
        with st.expander("📋 Full Traceback"):
            st.code(st.session_state.last_traceback, language="python")

    with st.expander("🛠 Environment Info"):
        st.json(st.session_state.debug_info.get("env", {}))


# ─────────────────────────────────────────────
# Final Reel tab
# ─────────────────────────────────────────────

with reel_tab:
    st.subheader("Final Highlight Reel")
    payload   = st.session_state.payload or {}
    final_vid = payload.get("final_video_path")
    segments  = payload.get("segments", [])

    if not final_vid:
        if payload:
            # A pipeline run completed but produced no reel — show why.
            st.warning("⚠️ Pipeline ran but no reel was generated. See diagnostics below.")
            total = payload.get("total_events", 0)
            st.metric("Events detected after dedup", total)
            warn_logs = [l for l in st.session_state.logs if "⚠" in l]
            if warn_logs:
                st.markdown("**Diagnostic messages:**")
                for l in warn_logs:
                    st.code(l, language=None)
            with st.expander("Full execution log"):
                st.code("\n".join(st.session_state.logs), language=None)
        else:
            st.info("🎬 Your compiled reel will appear here after generation.")
        # Show duration table to set expectations
        st.markdown("#### Expected clip durations per event type")
        rows = []
        for ev_type, dur in EVENT_DURATIONS.items():
            rows.append({"Event Type": ev_type, "Action Duration": f"{dur}s",
                         "Total Clip (with pre/post roll)": f"{PRE_ROLL_SEC + dur + POST_ROLL_SEC:.1f}s"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        total = payload.get("total_events", len(segments))
        dur   = payload.get("video_duration_sec", 0)
        st.success(f"✅ {total} events · {dur:.0f}s source video")

        # Player roster from OCR
        roster = payload.get("roster", [])
        if roster:
            st.markdown("**Players detected from scoreboard:**")
            chips = "".join(f'<span class="roster-chip">{p}</span>' for p in roster[:20])
            st.markdown(chips, unsafe_allow_html=True)

        st.video(final_vid)
        st.markdown("### Included Segments")
        render_segment_table(segments)


# ─────────────────────────────────────────────
# Thumbnails tab
# ─────────────────────────────────────────────

with thumbs_tab:
    st.subheader("Event Thumbnails")
    payload  = st.session_state.payload or {}
    segments = payload.get("segments", [])

    if not segments:
        st.info("🖼️ Thumbnails will appear here after generation.")
    else:
        ev_filter = st.selectbox("Filter by event", ["All"] + sorted({s["event_type"] for s in segments}))
        show_segs = segments if ev_filter == "All" else [s for s in segments if s["event_type"] == ev_filter]

        cols = st.columns(4)
        icon_map = {"Six": "🏏", "Four": "🏏", "Wicket": "🎯", "Player Milestone": "⭐", "Celebration": "🎉"}
        for i, seg in enumerate(show_segs):
            with cols[i % 4]:
                thumb = seg.get("thumbnail_path", "")
                if thumb and Path(thumb).exists():
                    st.image(thumb, use_container_width=True)
                else:
                    icon = icon_map.get(seg["event_type"], "🏏")
                    st.markdown(f"""
<div style='background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;
            padding:18px;text-align:center;margin-bottom:6px;'>
  <div style='font-size:1.8rem'>{icon}</div>
  <div style='font-size:0.75rem;color:#86efac;font-weight:600;margin-top:6px'>{seg["event_type"]}</div>
  <div style='font-size:0.65rem;color:#64748b'>{seg["timestamp"]}</div>
</div>""", unsafe_allow_html=True)
                st.caption(f"{seg.get('player','')} · {seg.get('confidence','')} · {seg.get('clip_duration','')}")


# ─────────────────────────────────────────────
# Player tab
# ─────────────────────────────────────────────

with player_tab:
    st.subheader("Player View")
    payload  = st.session_state.payload or {}
    segments = payload.get("segments", [])
    roster   = payload.get("roster", [])

    if not segments:
        st.info("👤 Player stats will appear after generation.")
    else:
        # Prefer OCR roster; fallback to names seen in segments
        seg_players = sorted({s.get("player","") for s in segments if s.get("player") and s["player"] != "Unknown"})
        all_players = sorted(set(roster) | set(seg_players))

        if not all_players:
            st.warning("No player names could be extracted.")
        else:
            sel = st.selectbox("Select Player", all_players)
            p_segs = [s for s in segments if s.get("player") == sel]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Events", len(p_segs))
            m2.metric("Sixes",    len([s for s in p_segs if s["event_type"] == "Six"]))
            m3.metric("Fours",    len([s for s in p_segs if s["event_type"] == "Four"]))
            m4.metric("Wickets",  len([s for s in p_segs if s["event_type"] == "Wicket"]))

            st.markdown(f"**{sel}** — {len(p_segs)} highlight segments")
            render_segment_table(p_segs)

            if roster:
                with st.expander("Full roster from scoreboard OCR"):
                    chips = "".join(f'<span class="roster-chip">{p}</span>' for p in roster)
                    st.markdown(chips, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Agent Insights tab
# ─────────────────────────────────────────────

with agent_tab:
    st.subheader("Multi-modal Agent Pipeline")
    if not st.session_state.agents:
        st.info("🤖 Agent execution details will appear after generation.")
        st.markdown("""
**Agents in this pipeline:**
- **Supervisor** — GPU health check, goal orchestration
- **Vision** — ffmpeg frame extraction
- **Audio** — Whisper transcription on AMD GPU
- **OCR** — Dense scorecard OCR every 5s
- **Roster** — Player name extraction from scoreboard (no hardcoding)
- **Wicket Delta** — Scorecard delta: wicket count + batter change detection
- **Crowd Signal** — Audio RMS energy peaks
- **Decision** — Multi-modal LLM fusion via vLLM on MI300X
- **Dedup** — Gap enforcement (per-event-type + global minimum)
- **Packaging** — Event-duration-aware clip cutting
- **Thumbnail** — Per-event thumbnail generation
- **Reel** — Final concat compilation
        """)
    else:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Agent Status")
            for ag in st.session_state.agents:
                with st.container(border=True):
                    st.write(f"**{ag['name']}**")
                    st.caption(ag["detail"])
        with right:
            st.markdown("#### Execution Log")
            if st.session_state.logs:
                log_html = "<br>".join(st.session_state.logs)
                st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No logs.")


# ─────────────────────────────────────────────
# Analytics tab
# ─────────────────────────────────────────────

with analytics_tab:
    st.subheader("Match Analytics")
    df       = st.session_state.analytics
    payload  = st.session_state.payload or {}
    segments = payload.get("segments", [])

    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("📈 Analytics will appear after generation.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        if segments:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Events by Type")
                ev_df = pd.DataFrame(segments).groupby("event_type").size().reset_index(name="Count")
                st.bar_chart(ev_df.set_index("event_type"))

            with c2:
                st.markdown("#### Events by Player")
                pl_df = pd.DataFrame(segments)
                pl_df = pl_df[pl_df["player"] != "Unknown"].groupby("player").size().reset_index(name="Count")
                if not pl_df.empty:
                    st.bar_chart(pl_df.set_index("player"))
                else:
                    st.info("No named player events.")

            st.markdown("#### Source breakdown (LLM vs OCR-delta)")
            src_df = pd.DataFrame(segments).groupby("source").size().reset_index(name="Count")
            st.dataframe(src_df, hide_index=True)


# ─────────────────────────────────────────────
# Export tab
# ─────────────────────────────────────────────

with export_tab:
    st.subheader("Exports")
    payload = st.session_state.payload or {}
    if not payload.get("final_video_path"):
        st.info("⬇️ Export options appear after generation.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ Segment Metadata (CSV)",
                data=pd.DataFrame(payload.get("segments", [])).to_csv(index=False),
                file_name="flashboundary_segments.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇️ Agent Logs (TXT)",
                data="\n".join(st.session_state.logs),
                file_name="flashboundary_logs.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c3:
            df_csv = st.session_state.analytics.to_csv(index=False) if not st.session_state.analytics.empty else ""
            st.download_button(
                "⬇️ Analytics (CSV)",
                data=df_csv,
                file_name="flashboundary_analytics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        roster = payload.get("roster", [])
        if roster:
            st.markdown("#### Detected Player Roster")
            confirmed = payload.get("roster_confirmed", [])
            ocr_only  = payload.get("roster_ocr_only", [])
            if confirmed:
                st.markdown("**✅ Confirmed (commentary + scoreboard agree):**")
                chips = "".join(f'<span class="roster-chip">{p}</span>' for p in confirmed)
                st.markdown(chips, unsafe_allow_html=True)
            if ocr_only:
                st.markdown("**📊 Scoreboard-only (not yet heard in commentary):**")
                chips = "".join(f'<span class="roster-chip">{p}</span>' for p in ocr_only)
                st.markdown(chips, unsafe_allow_html=True)
            unmatched = payload.get("roster_unmatched_commentary", [])
            if unmatched:
                with st.expander(f"🎙️ Names heard in commentary but not matched to scoreboard ({len(unmatched)})"):
                    st.write(", ".join(unmatched))

        st.caption(f"Reel: `{payload.get('final_video_path')}`")
        with st.expander("Full payload JSON"):
            st.json({k: v for k, v in payload.items() if k != "segments"})

st.divider()
st.caption("FlashBoundary · AMD MI300X GPU · vLLM + Whisper + EasyOCR · No cloud fallback · AUM Hackathon 2025")
