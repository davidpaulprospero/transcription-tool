"""Streamlit demo for the transcription pipeline.

Upload an audio or video file, transcribe it, and optionally trim the silence
first and watch the timestamps get remapped back onto the original timeline.

Run with:
    streamlit run demo.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Allow "streamlit run demo.py" from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transcription.silence_removal import remove_voiceover_silence
from src.transcription.utils import remap_segments_to_original_time
from src.transcription.whisper_client import WhisperClient

st.set_page_config(
    page_title="Transcription Pipeline",
    page_icon="🎙️",
    layout="wide",
)

MODEL_CHOICES = ["tiny", "base", "small", "medium"]
MODEL_NOTES = {
    "tiny": "fastest, roughest — good for trying the pipeline",
    "base": "fast and usable",
    "small": "noticeably better, slower",
    "medium": "best accuracy here, slow on CPU",
}


@st.cache_resource(show_spinner="Loading model…")
def get_client(model_name: str) -> WhisperClient:
    """Build a WhisperClient. Model weights load lazily on first use."""
    return WhisperClient(model_name=model_name)


def unwrap_transcribe_result(result) -> tuple[list[dict], dict]:
    """Split WhisperClient.transcribe() output into (segments, quality_metrics).

    The return shape is path-dependent: the main path returns
    ``(segments, quality_metrics)``, the CPU-fallback path returns a bare
    segment list, and a total failure returns ``[]``.
    """
    metrics: dict = {}
    if isinstance(result, tuple):
        segments = result[0] if result else []
        if len(result) > 1 and isinstance(result[1], dict):
            metrics = result[1]
    else:
        segments = result
    return list(segments) if segments else [], metrics


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    """Render segments as SRT."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.get("start", 0.0))
        end = format_timestamp(seg.get("end", 0.0))
        blocks.append(f"{i}\n{start} --> {end}\n{seg.get('text', '').strip()}\n")
    return "\n".join(blocks)


st.title("🎙️ Transcription Pipeline")
st.caption(
    "Speech-to-text with silence trimming that keeps your timestamps honest. "
    "Self-initiated project."
)

with st.expander("What am I looking at?"):
    st.markdown(
        """
Trimming silence before transcription is a real speedup — the model reads less
audio. It also quietly breaks the output: the model reports timestamps against
the shortened file, so subtitles drift earlier and earlier against your original
recording.

This tool keeps the mapping explicit. The trimmer returns the speech regions it
found, and every timestamp is shifted back onto the original timeline using those
regions as anchors.

Tick **Trim silence** and compare the two timestamp columns.
"""
    )

st.divider()

uploaded = st.file_uploader(
    "Audio or video file",
    type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "mov", "webm", "mkv"],
    help="Short clips work best — transcription runs on CPU unless a GPU is present.",
)

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox(
        "Model",
        MODEL_CHOICES,
        index=1,
        format_func=lambda m: f"{m} — {MODEL_NOTES[m]}",
    )
with col2:
    language = st.text_input("Language (blank = auto-detect)", value="")

trim = st.checkbox(
    "Trim silence before transcribing",
    value=True,
    help="Removes long silences, then remaps timestamps back to the original timeline.",
)

with st.expander("Trimming settings"):
    a, b, c, d = st.columns(4)
    min_silence = a.slider("Min silence (ms)", 200, 3000, 700, 100)
    threshold = b.slider("Threshold (dBFS)", -60, -10, -35, 1)
    keep_silence = c.slider("Padding (ms)", 0, 1000, 250, 25)
    crossfade = d.slider("Crossfade (ms)", 0, 500, 50, 10)

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

# Save to a temp dir so the trimmer can write alongside it.
tmp_dir = Path(st.session_state.setdefault("_tmp_dir", tempfile.mkdtemp(prefix="transcription-demo-")))
tmp_dir.mkdir(parents=True, exist_ok=True)
source_path = tmp_dir / uploaded.name
source_path.write_bytes(uploaded.getvalue())

st.caption(f"{uploaded.name} — {len(uploaded.getvalue()) / 1_048_576:.1f} MB")

if not st.button("Transcribe", type="primary"):
    st.stop()

progress = st.progress(0, text="Preparing…")

speech_regions: list[tuple[int, int]] = []
audio_for_model = str(source_path)

with st.spinner("Trimming silence…"):
    try:
        trim_result = remove_voiceover_silence(
            str(source_path),
            min_silence_len_ms=min_silence,
            silence_thresh_dbfs=threshold,
            keep_silence_ms=keep_silence,
            crossfade_ms=crossfade,
        )
    except Exception as exc:  # noqa: BLE001 - surface the real reason in the UI
        st.error(f"Silence removal failed: {exc}")
        st.stop()

    if trim_result is not None and trim_result.was_trimmed:
        audio_for_model = trim_result.trimmed_path
        speech_regions = trim_result.speech_regions_ms
        progress.progress(0.2, text=f"Trimmed into {len(speech_regions)} speech regions.")
    else:
        progress.progress(0.2, text="Nothing to trim.")

progress.progress(0.4, text="Loading model and transcribing…")
started = time.time()

client = get_client(model_name)

try:
    raw_segments, quality_metrics = unwrap_transcribe_result(
        client.transcribe(
            audio_for_model,
            language=language or None,
        )
    )
except FileNotFoundError as exc:
    progress.empty()
    st.error(
        "ffmpeg was not found on this system. It is required to decode the "
        "uploaded media into audio. Install it and retry:\n\n"
        "- **Windows:** `winget install Gyan.FFmpeg`\n"
        "- **macOS:** `brew install ffmpeg`\n"
        "- **Debian/Ubuntu:** `sudo apt install ffmpeg`"
    )
    st.stop()
except Exception as exc:  # noqa: BLE001 - surface the real reason in the UI
    progress.empty()
    st.error(f"Transcription failed: {exc}")
    st.stop()

elapsed = time.time() - started
progress.progress(1.0, text=f"Done in {elapsed:.1f}s")
progress.empty()

if not raw_segments:
    st.warning("No speech detected in that file.")
    st.stop()

# Keep an unmapped copy so the drift can be shown when trimming was on.
segments = remap_segments_to_original_time(
    raw_segments, speech_regions, crossfade_ms=crossfade
) if speech_regions else list(raw_segments)

words = sum(len(seg.get("text", "").split()) for seg in segments)
st.success(
    f"{len(segments)} segments, {words} words in {elapsed:.1f}s "
    f"using the {model_name} model."
)

# avg_word_confidence is only meaningful when word-level timestamps are on, which
# they are not by default — it comes back as 0.0. min_segment_confidence is real.
min_conf = quality_metrics.get("min_segment_confidence")
if min_conf:
    st.caption(
        f"Weakest segment confidence: {min_conf:.0%} "
        "(word-level confidence needs `word_timestamps=True`, so it is not shown)."
    )

if speech_regions:
    st.info(
        f"Trimmed into **{len(speech_regions)} speech regions**. "
        "Timestamps below are mapped back to the original timeline."
    )

st.divider()

# Transcript table
show_raw = bool(speech_regions) and st.checkbox(
    "Show the un-remapped timestamps too (the drift trimming would cause)",
    value=True,
)

if show_raw:
    rows = []
    for i, (raw, mapped) in enumerate(zip(raw_segments, segments), start=1):
        raw_start = raw.get("start", 0.0)
        mapped_start = mapped.get("start", 0.0)
        rows.append(
            {
                "#": i,
                "Start (trimmed)": format_timestamp(raw_start),
                "Start (corrected)": format_timestamp(mapped_start),
                "Drift": f"{mapped_start - raw_start:+.2f}s",
                "Text": mapped.get("text", "").strip(),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    rows = [
        {
            "#": i,
            "Start": format_timestamp(seg.get("start", 0.0)),
            "End": format_timestamp(seg.get("end", 0.0)),
            "Text": seg.get("text", "").strip(),
        }
        for i, seg in enumerate(segments, start=1)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.download_button(
    "Download SRT",
    data=to_srt(segments),
    file_name=f"{source_path.stem}.srt",
    mime="text/plain",
)

if speech_regions:
    with st.expander("Speech regions found"):
        st.dataframe(
            {
                "Region": list(range(1, len(speech_regions) + 1)),
                "Start (original)": [format_timestamp(r[0] / 1000) for r in speech_regions],
                "End (original)": [format_timestamp(r[1] / 1000) for r in speech_regions],
            },
            use_container_width=True,
            hide_index=True,
        )