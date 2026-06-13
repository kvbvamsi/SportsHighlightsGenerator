"""
app.py — FlashBoundary | Real-time Cricket Highlight Generator
Multi-modal pipeline: video frames + commentary audio + crowd signals → highlights + captions + thumbnails
Built for AMD MI300X GPU via vLLM + Pydantic AI (as per AUM GPU workshop notebooks)
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

from backend import generate_highlights, VLLM_BASE_URL, VLLM_MODEL


# ─────────────────────────────────────────────
# Debug helpers
# ─────────────────────────────────────────────

def debug_print(message: str) -> None:
    print(f"[APP DEBUG] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def get_environment_diagnostics() -> dict:
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "ffmpeg_in_path": False,
        "tesseract_in_path": False,
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL,
    }
    try:
        import shutil
        info["ffmpeg_in_path"] = shutil.which("ffmpeg") is not None
        info["tesseract_in_path"] = shutil.which("tesseract") is not None
    except Exception as exc:
        info["path_check_error"] = str(exc)

    for mod in ["pandas", "streamlit", "requests", "faster_whisper", "whisper", "easyocr", "pytesseract"]:
        try:
            __import__(mod)
            info[f"import_{mod}"] = True
        except Exception:
            info[f"import_{mod}"] = False
    return info


def check_vllm_server() -> tuple[bool, str]:
    """Ping the vLLM server to see if it's reachable."""
    try:
        import requests as req
        r = req.get(f"{VLLM_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'abc-123')}"},
                    timeout=3)
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            return True, f"Online — models: {', '.join(models)}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="flashboundary_upload_"))
    file_path = temp_dir / f"uploaded_video{suffix}"
    debug_print(f"Saving uploaded file to: {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    debug_print(f"Saved | size={file_path.stat().st_size} bytes")
    return str(file_path)


def reset_state() -> None:
    debug_print("Resetting session state")
    for key in ["payload", "agents", "logs", "analytics", "video_url", "last_error",
                "input_source", "last_traceback", "debug_info"]:
        defaults = {"payload": {}, "agents": [], "logs": [], "analytics": pd.DataFrame(),
                    "video_url": "", "last_error": "", "input_source": "",
                    "last_traceback": "", "debug_info": {}}
        st.session_state[key] = defaults[key]


def render_segment_table(segments):
    if not segments:
        st.info("No event segments available.")
        return
    data = [{
        "Event Type": x.get("event_type", ""),
        "Timestamp": x.get("timestamp", ""),
        "Player": x.get("player", ""),
        "Confidence": x.get("confidence", ""),
        "Caption": x.get("caption", ""),
        "Segment Path": x.get("segment_path", ""),
    } for x in segments]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────

debug_print("Starting FlashBoundary Streamlit app")

st.set_page_config(
    page_title="FlashBoundary | Cricket Highlight Generator",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_bar="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0d0f14; }
  
  .stApp { background: linear-gradient(160deg, #0d0f14 0%, #111827 60%, #0a0d12 100%); }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #1a2332 0%, #0f1923 50%, #1a1a2e 100%);
    border: 1px solid #22c55e33;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #22c55e, #16a34a, #4ade80, #22c55e);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
  }
  @keyframes shimmer { to { background-position: 200% center; } }

  .hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #f0fdf4;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #86efac;
    opacity: 0.8;
  }

  /* Event badges */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
  }
  .badge-six { background: #14532d; color: #4ade80; }
  .badge-four { background: #1e3a5f; color: #60a5fa; }
  .badge-wicket { background: #450a0a; color: #f87171; }
  .badge-milestone { background: #451a03; color: #fb923c; }
  .badge-celebration { background: #2e1065; color: #c084fc; }

  /* Agent cards */
  .agent-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
  }

  /* GPU status */
  .gpu-online { color: #22c55e; font-weight: 600; }
  .gpu-offline { color: #ef4444; font-weight: 600; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] {
    font-size: 0.875rem;
    font-weight: 600;
    color: #94a3b8;
  }
  .stTabs [aria-selected="true"] {
    color: #22c55e !important;
    border-bottom-color: #22c55e !important;
  }

  /* Log box */
  .log-box {
    background: #0a0d12;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #86efac;
    max-height: 280px;
    overflow-y: auto;
  }

  /* Thumbnail grid */
  .thumb-grid { display: flex; flex-wrap: wrap; gap: 10px; }
  .thumb-item {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 6px;
    text-align: center;
    width: 160px;
  }
  .thumb-caption { font-size: 0.7rem; color: #94a3b8; margin-top: 4px; }

  /* Progress */
  .step-progress {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 0.5rem 0;
    font-size: 0.82rem;
    color: #64748b;
  }
  .step-active { color: #22c55e; font-weight: 600; }
  .step-done { color: #4ade80; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1419 !important;
    border-right: 1px solid #1e293b !important;
  }
</style>
""", unsafe_allow_html=True)

# Session state init
for key, default in [
    ("payload", {}), ("agents", []), ("logs", []), ("analytics", pd.DataFrame()),
    ("video_url", ""), ("last_error", ""), ("input_source", ""),
    ("last_traceback", ""), ("debug_info", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

env_info = get_environment_diagnostics()
st.session_state.debug_info["environment"] = env_info


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Config")

    # GPU status
    gpu_online, gpu_msg = check_vllm_server()
    if gpu_online:
        st.markdown(f'<p class="gpu-online">🟢 AMD MI300X vLLM Online</p>', unsafe_allow_html=True)
        st.caption(gpu_msg)
    else:
        st.markdown(f'<p class="gpu-offline">🔴 vLLM Offline — Claude API fallback</p>', unsafe_allow_html=True)
        st.caption(f"Reason: {gpu_msg[:80]}")
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("Set ANTHROPIC_API_KEY for fallback mode")

    st.divider()

    highlight_mode = st.radio(
        "Highlight Mode",
        ["Match Highlights", "Highlights by Player"],
        index=0,
    )

    selected_events = st.multiselect(
        "Event Types to Detect",
        ["Six", "Four", "Wicket", "Player Milestone", "Celebration"],
        default=["Six", "Four", "Wicket"],
    )

    player_name = None
    if highlight_mode == "Highlights by Player":
        player_name = st.text_input("Player Name", placeholder="Rohit Sharma")

    confidence_threshold = st.slider("Min Confidence", 0.50, 1.00, 0.70, 0.05)

    st.divider()
    st.markdown("### 🧠 Multi-modal Pipeline")
    steps_info = [
        ("🎞️", "Frame Extraction", "1 fps via ffmpeg"),
        ("🎙️", "Commentary Transcription", "Whisper on AMD GPU"),
        ("📢", "Crowd Signal Analysis", "Audio RMS peaks"),
        ("📊", "Scorecard OCR", "EasyOCR / Tesseract"),
        ("🤖", "Event Detection", f"LLM: {VLLM_MODEL}"),
        ("✂️", "Clip Cutting", "ffmpeg + caption overlay"),
        ("🎬", "Reel Compilation", "Concat + encode"),
    ]
    for icon, title, detail in steps_info:
        st.markdown(f"**{icon} {title}**  \n<small style='color:#64748b'>{detail}</small>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📡 Server")
    st.code(f"vLLM: {VLLM_BASE_URL}\nModel: {VLLM_MODEL}", language=None)


# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🏏 FlashBoundary</div>
  <div class="hero-sub">Real-time Cricket Highlight Generator · Multi-modal AI Pipeline · AMD MI300X GPU</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

home_tab, reel_tab, thumbs_tab, player_tab, agent_tab, analytics_tab, export_tab = st.tabs([
    "🏠 Home",
    "🎬 Final Reel",
    "🖼️ Thumbnails",
    "👤 Player View",
    "🤖 Agent Insights",
    "📈 Analytics",
    "⬇️ Exports",
])


# ─────────────────────────────────────────────
# Home tab
# ─────────────────────────────────────────────

with home_tab:
    st.subheader("Upload Match Video")

    upload_col, url_col = st.columns([1, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload match video",
            type=["mp4", "mov", "mkv", "avi"],
            help="Supports MP4, MOV, MKV, AVI formats",
        )

    with url_col:
        video_url = st.text_input(
            "OR enter direct video URL",
            value=st.session_state.video_url,
            placeholder="https://example.com/cricket-match.mp4",
        )
        st.session_state.video_url = video_url

    if uploaded_file is not None:
        debug_print(f"File received: {uploaded_file.name}, {uploaded_file.size} bytes")
        try:
            source_input = save_uploaded_file(uploaded_file)
            st.session_state.input_source = source_input
            st.success(f"✅ {uploaded_file.name} ready ({uploaded_file.size / 1e6:.1f} MB)")
            st.video(source_input)
        except Exception as exc:
            tb = traceback.format_exc()
            st.session_state.last_error = str(exc)
            st.session_state.last_traceback = tb
            st.error(f"Failed to save file: {exc}")

    elif video_url:
        if is_valid_url(video_url):
            st.session_state.input_source = video_url
            st.success("✅ Valid video URL")
            try:
                st.video(video_url)
            except Exception:
                st.info("Preview unavailable, but URL will be processed.")
        else:
            st.error("Invalid URL — please enter a valid direct video link.")

    # Execution summary
    st.divider()
    st.markdown("#### Pipeline Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mode", highlight_mode.replace("Highlights ", ""))
    col2.metric("Events", len(selected_events))
    col3.metric("Confidence", f"{confidence_threshold:.0%}")
    col4.metric("Player Filter", player_name or "All players")

    if selected_events:
        badge_classes = {"Six": "six", "Four": "four", "Wicket": "wicket",
                         "Player Milestone": "milestone", "Celebration": "celebration"}
        badges = " ".join(
            '<span class="badge badge-{}">{}</span>'.format(badge_classes.get(e, "six"), e)
            for e in selected_events
        )
        st.markdown(f"**Detecting:** {badges}", unsafe_allow_html=True)

    st.divider()

    # ── Generate button ──
    btn_col, reset_col = st.columns([2, 1])

    with btn_col:
        gen_disabled = not st.session_state.input_source
        if st.button("🚀 Generate Highlight Reel", type="primary", use_container_width=True, disabled=gen_disabled):
            st.session_state.last_error = ""
            st.session_state.last_traceback = ""

            source_input = st.session_state.input_source

            if not source_input:
                st.error("Upload a video or enter a URL first.")
            elif highlight_mode == "Highlights by Player" and not player_name:
                st.error("Enter a player name for player-mode highlights.")
            elif not selected_events:
                st.error("Select at least one event type.")
            else:
                pipeline_steps = [
                    ("🎞️ Extracting frames", 0.14),
                    ("🎙️ Transcribing commentary", 0.28),
                    ("📢 Analysing crowd signals", 0.42),
                    ("📊 Running scorecard OCR", 0.55),
                    ("🤖 Detecting events via LLM", 0.72),
                    ("✂️ Cutting highlight clips", 0.87),
                    ("🎬 Stitching final reel", 1.00),
                ]
                progress = st.progress(0.0)
                status_box = st.empty()

                for msg, pct in pipeline_steps[:-1]:
                    status_box.info(f"**{msg}…**")
                    progress.progress(pct)
                    time.sleep(0.08)

                status_box.info("**🤖 Calling AMD MI300X vLLM / Claude fallback for multi-modal event detection…**")

                try:
                    start_t = time.time()
                    payload, agents, logs, analytics = generate_highlights(
                        source_input=source_input,
                        selected_events=selected_events,
                        highlight_mode=highlight_mode,
                        player_name=player_name,
                        confidence_threshold=confidence_threshold,
                    )
                    elapsed = time.time() - start_t

                    st.session_state.payload = payload
                    st.session_state.agents = agents
                    st.session_state.logs = logs
                    st.session_state.analytics = analytics
                    st.session_state.debug_info["elapsed_sec"] = round(elapsed, 2)

                    progress.progress(1.0)
                    status_box.success(f"✅ {payload.get('message', 'Reel generated!')} — {elapsed:.1f}s")

                except Exception as exc:
                    tb = traceback.format_exc()
                    st.session_state.last_error = str(exc)
                    st.session_state.last_traceback = tb
                    progress.empty()
                    status_box.error(f"Pipeline failed: {exc}")
                    debug_print(tb)

    with reset_col:
        if st.button("🔄 Reset", use_container_width=True):
            reset_state()
            st.rerun()

    if st.session_state.last_error:
        st.warning(f"⚠ {st.session_state.last_error}")
    if st.session_state.last_traceback:
        with st.expander("📋 Full Traceback", expanded=False):
            st.code(st.session_state.last_traceback, language="python")

    # Debug info
    with st.expander("🛠 Debug / Environment Info", expanded=False):
        st.json(st.session_state.debug_info.get("environment", {}))


# ─────────────────────────────────────────────
# Final reel tab
# ─────────────────────────────────────────────

with reel_tab:
    st.subheader("Final Highlight Reel")
    payload = st.session_state.payload or {}
    final_video_path = payload.get("final_video_path")
    segments = payload.get("segments", [])

    if not final_video_path:
        st.info("🎬 Your compiled highlight reel will appear here after generation.")
        st.markdown("""
        **What FlashBoundary produces:**
        - One continuous highlight reel video
        - Each clip has a caption overlay burned in
        - Events ordered by match timestamp
        - Only selected event types included
        """)
    else:
        total = payload.get("total_events", len(segments))
        st.success(f"✅ Reel compiled — **{total} events** detected across {payload.get('video_duration_sec', 0):.0f}s match.")
        st.video(final_video_path)
        st.markdown("### 📋 Included Segments")
        render_segment_table(segments)


# ─────────────────────────────────────────────
# Thumbnails tab
# ─────────────────────────────────────────────

with thumbs_tab:
    st.subheader("Event Thumbnails")
    payload = st.session_state.payload or {}
    segments = payload.get("segments", [])

    if not segments:
        st.info("🖼️ Event thumbnails will appear here after generation.")
    else:
        event_filter = st.selectbox("Filter by event", ["All"] + list({s["event_type"] for s in segments}))
        filtered = segments if event_filter == "All" else [s for s in segments if s["event_type"] == event_filter]

        cols = st.columns(4)
        for i, seg in enumerate(filtered):
            with cols[i % 4]:
                thumb = seg.get("thumbnail_path", "")
                if thumb and Path(thumb).exists():
                    st.image(thumb, caption=seg.get("caption", ""), use_container_width=True)
                else:
                    badge_map = {"Six": "🏏", "Four": "🏏", "Wicket": "🎯",
                                 "Player Milestone": "⭐", "Celebration": "🎉"}
                    icon = badge_map.get(seg["event_type"], "🏏")
                    st.markdown(f"""
                    <div style='background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;
                                padding:20px;text-align:center;margin-bottom:8px;'>
                      <div style='font-size:2rem'>{icon}</div>
                      <div style='font-size:0.75rem;color:#86efac;margin-top:6px;font-weight:600'>{seg['event_type']}</div>
                      <div style='font-size:0.65rem;color:#64748b;margin-top:2px'>{seg['timestamp']}</div>
                      <div style='font-size:0.65rem;color:#94a3b8;margin-top:4px'>{seg.get('caption','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption(f"{seg.get('player', '')} · {seg.get('confidence', '')} conf.")


# ─────────────────────────────────────────────
# Player tab
# ─────────────────────────────────────────────

with player_tab:
    st.subheader("Player View")
    payload = st.session_state.payload or {}
    segments = payload.get("segments", [])

    if not segments:
        st.info("👤 Player-specific analysis will appear here after generation.")
    else:
        players = sorted({x.get("player", "") for x in segments if x.get("player") and x["player"] != "Unknown"})
        if not players:
            players = ["Unknown"]
        selected_player = st.selectbox("Select Player", players)
        player_segs = [x for x in segments if x.get("player") == selected_player]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Events", len(player_segs))
        m2.metric("Sixes", len([s for s in player_segs if s["event_type"] == "Six"]))
        m3.metric("Fours", len([s for s in player_segs if s["event_type"] == "Four"]))

        st.markdown(f"**{selected_player}** appears in **{len(player_segs)}** highlight segments.")
        render_segment_table(player_segs)


# ─────────────────────────────────────────────
# Agent tab
# ─────────────────────────────────────────────

with agent_tab:
    st.subheader("Multi-modal Agent Pipeline")
    if not st.session_state.agents:
        st.info("🤖 Agent execution details will appear here after generation.")
        st.markdown("""
        **Pipeline Agents:**
        - **Supervisor Agent** — Orchestrates goal execution
        - **Vision Agent** — Frame extraction & visual analysis
        - **Audio Agent** — Whisper transcription on AMD GPU
        - **OCR Agent** — Scorecard text extraction
        - **Crowd Signal Agent** — Audio energy peak detection
        - **Decision Agent** — Multi-modal LLM fusion (vLLM/Claude)
        - **Packaging Agent** — Clip cutting, caption overlay, reel stitch
        - **Thumbnail Agent** — Event thumbnail generation
        """)
    else:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Agent Status")
            for agent in st.session_state.agents:
                with st.container(border=True):
                    st.write(f"**{agent['name']}**")
                    st.caption(agent["detail"])

        with right:
            st.markdown("#### Execution Logs")
            if st.session_state.logs:
                log_text = "\n".join(st.session_state.logs)
                st.markdown(f'<div class="log-box">{log_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            else:
                st.info("No logs available.")


# ─────────────────────────────────────────────
# Analytics tab
# ─────────────────────────────────────────────

with analytics_tab:
    st.subheader("Match Analytics")
    if st.session_state.analytics is None or (hasattr(st.session_state.analytics, "empty") and st.session_state.analytics.empty):
        st.info("📈 Analytics will appear here after generation.")
    else:
        df = st.session_state.analytics
        st.dataframe(df, use_container_width=True, hide_index=True)

        segments = (st.session_state.payload or {}).get("segments", [])
        if segments:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Events by Type")
                event_df = pd.DataFrame(segments).groupby("event_type").size().reset_index(name="Count")
                st.bar_chart(event_df.set_index("event_type"), use_container_width=True)

            with col2:
                st.markdown("#### Confidence Distribution")
                conf_data = pd.DataFrame(segments)
                conf_data["conf_float"] = conf_data["confidence"].str.rstrip("%").astype(float, errors="ignore")
                try:
                    st.bar_chart(conf_data["conf_float"].value_counts().sort_index())
                except Exception:
                    st.info("Confidence chart unavailable.")


# ─────────────────────────────────────────────
# Export tab
# ─────────────────────────────────────────────

with export_tab:
    st.subheader("Exports")
    payload = st.session_state.payload or {}
    if not payload.get("final_video_path"):
        st.info("⬇️ Export options will appear here after generation.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            segs_csv = pd.DataFrame(payload.get("segments", [])).to_csv(index=False)
            st.download_button(
                label="⬇️ Segment Metadata CSV",
                data=segs_csv,
                file_name="flashboundary_segments.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                label="⬇️ Agent Execution Logs",
                data="\n".join(st.session_state.logs),
                file_name="flashboundary_agent_logs.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col3:
            analytics_csv = st.session_state.analytics.to_csv(index=False) if not st.session_state.analytics.empty else "No data"
            st.download_button(
                label="⬇️ Analytics CSV",
                data=analytics_csv,
                file_name="flashboundary_analytics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.caption(f"Final reel: `{payload.get('final_video_path')}`")
        with st.expander("Preview full payload JSON"):
            safe_payload = {k: v for k, v in payload.items() if k != "segments"}
            st.json(safe_payload)

st.divider()
st.caption("FlashBoundary · AMD MI300X GPU · vLLM + Pydantic AI + Whisper + EasyOCR · AUM Hackathon 2025")
