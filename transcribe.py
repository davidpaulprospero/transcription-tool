#!/usr/bin/env python3
"""Transcribe an audio or video file, with optional silence trimming.

The point of this tool is the `--trim-silence` path: it removes silence from
the audio before transcription and then remaps every returned timestamp back
onto the original timeline, so subtitles still line up with the source.

Examples:
    python transcribe.py talk.mp3
    python transcribe.py talk.mp3 --trim-silence --srt talk.srt
    python transcribe.py talk.mp3 --trim-silence --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file from any directory without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transcription.silence_removal import remove_voiceover_silence
from src.transcription.utils import remap_segments_to_original_time


def unwrap_transcribe_result(result) -> list[dict]:
    """Return the segment list from whatever WhisperClient.transcribe() gave back.

    The return shape is path-dependent: the main path returns
    ``(segments, quality_metrics)``, the CPU-fallback path returns a bare
    segment list, and a total failure returns ``[]``. Normalising here keeps the
    rest of the file working against one type.
    """
    if isinstance(result, tuple):
        result = result[0] if result else []
    return result if result else []


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp (HH:MM:SS,mmm)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], out_path: Path) -> None:
    """Write segments to an SRT file."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.get("start", 0.0))
        end = format_timestamp(seg.get("end", 0.0))
        blocks.append(f"{i}\n{start} --> {end}\n{seg.get('text', '').strip()}\n")
    out_path.write_text("\n".join(blocks), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a media file, optionally trimming silence first.",
    )
    parser.add_argument("input", type=Path, help="Audio or video file")
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--language", default=None, help="Force a language code, e.g. en")
    parser.add_argument("--srt", type=Path, default=None, help="Write an SRT file here")
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Remove silence before transcribing, then remap timestamps to the original timeline",
    )
    parser.add_argument(
        "--min-silence-ms", type=int, default=700,
        help="Minimum silence length to remove (default: 700)",
    )
    parser.add_argument(
        "--silence-threshold-dbfs", type=int, default=-35,
        help="Absolute silence threshold in dBFS (default: -35)",
    )
    parser.add_argument(
        "--keep-silence-ms", type=int, default=250,
        help="Padding kept either side of speech (default: 250)",
    )
    parser.add_argument(
        "--crossfade-ms", type=int, default=50,
        help="Crossfade between joined regions (default: 50)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="With --trim-silence, print the timestamp drift that remapping corrects",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1

    # Import lazily so --help works without the heavy dependencies installed.
    try:
        from src.transcription.whisper_client import WhisperClient
    except ImportError as exc:
        print(f"error: missing dependency ({exc}). Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    audio_for_model = str(args.input)
    speech_regions_ms: list[tuple[int, int]] = []
    crossfade_ms = args.crossfade_ms

    if args.trim_silence:
        result = remove_voiceover_silence(
            str(args.input),
            min_silence_len_ms=args.min_silence_ms,
            silence_thresh_dbfs=args.silence_threshold_dbfs,
            keep_silence_ms=args.keep_silence_ms,
            crossfade_ms=args.crossfade_ms,
        )
        if result is None:
            print("error: silence removal failed", file=sys.stderr)
            return 1
        if result.was_trimmed:
            audio_for_model = result.trimmed_path
            speech_regions_ms = result.speech_regions_ms
            removed = (args.input.stat().st_size - Path(result.trimmed_path).stat().st_size) / 1024
            print(
                f"Trimmed {len(speech_regions_ms)} speech regions "
                f"(~{removed:.0f} KB removed), remapping timestamps to the original timeline."
            )
        else:
            print("Nothing to trim; transcribing the original audio.")

    client = WhisperClient(model_name=args.model)
    segments = unwrap_transcribe_result(
        client.transcribe(audio_for_model, language=args.language)
    )

    if speech_regions_ms:
        raw_segments = [dict(s) for s in segments]
        segments = remap_segments_to_original_time(
            segments, speech_regions_ms, crossfade_ms=crossfade_ms
        )
        if args.compare and raw_segments:
            first_raw = raw_segments[0].get("start", 0.0)
            first_mapped = segments[0].get("start", 0.0)
            print(
                f"First segment: {first_raw:.2f}s on the trimmed timeline, "
                f"{first_mapped:.2f}s on the original — "
                f"{first_mapped - first_raw:+.2f}s of drift corrected."
            )

    if not segments:
        print("No speech detected.")
        return 0

    for seg in segments:
        print(f"[{seg.get('start', 0):7.2f} -> {seg.get('end', 0):7.2f}] {seg.get('text', '').strip()}")

    if args.srt:
        write_srt(segments, args.srt)
        print(f"\nWrote {len(segments)} segments to {args.srt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())