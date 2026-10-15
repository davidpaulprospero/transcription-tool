"""Core data types for the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TranscriptSegment:
    """A single transcript segment produced by transcription.

    Times are in seconds. `start_time`/`end_time` are on whichever timeline the
    segment's producer used - see `transcription.utils.remap_segments_to_original_time`
    for moving them between the trimmed and original audio timelines.
    """

    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)