"""Configuration objects for the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SegmentPostProcessingConfig:
    """Controls what happens to segments after transcription.

    - Merge consecutive segments that belong to the same speaker.
    - Split over-long segments at natural language boundaries.
    """

    # Enable/disable all post-processing
    enabled: bool = True

    # Merge segments with the same speaker (if speaker labels available)
    merge_same_speaker: bool = True

    # Maximum gap between segments to consider merging (seconds)
    merge_max_gap_seconds: float = 0.5

    # Minimum duration for a merged segment (seconds)
    merge_min_duration: float = 1.0

    # Split segments at natural language boundaries (punctuation)
    split_at_punctuation: bool = True

    # Punctuation that triggers a split, in order of precedence
    split_punctuation: str = ".!?"

    # Don't split on abbreviations ("Mr.", "Dr.", "U.S.")
    split_ignore_abbreviations: bool = True

    # Abbreviations to ignore when splitting
    abbreviations: List[str] = None

    # Minimum text length after punctuation before starting a new segment
    split_min_after_text: int = 2

    # Minimum duration for split segments (seconds)
    split_min_duration: float = 0.5

    # Merge by detected speaker change (from VAD) rather than explicit labels
    enable_speaker_aware_merging: bool = False

    # Confidence required to call a boundary a speaker change (0.0-1.0)
    speaker_change_confidence_threshold: float = 0.7

    def __post_init__(self):
        if self.abbreviations is None:
            self.abbreviations = [
                "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
                "vs", "etc", "eg", "ie", "al",
                "us", "usa", "uk", "eu", "un", "nato",
                "a", "b", "c", "d", "e", "f", "g", "h",
            ]


@dataclass
class TranscriptionConfig:
    """Settings controlling how transcription runs."""

    provider: str = "faster-whisper"
    model: str = "base"          # tiny, base, small, medium, large
    model_version: Optional[str] = None
    language: str = "en"
    use_gpu: bool = True
    compute_type: str = "auto"   # auto, float16, int8, float32

    # GPU memory
    minimum_gpu_memory_mb: int = 2000
    auto_downgrade_model: bool = True

    # Parallel processing
    max_workers: int = 4
    batch_size: int = 50
    batch_wait_seconds: int = 5

    # Silence removal
    min_silence_len_ms: int = 700    # only silences >= this are removed
    silence_thresh_dbfs: int = -35   # absolute threshold
    keep_silence_ms: int = 250       # padding either side of speech
    crossfade_ms: int = 50           # smooth transitions between regions

    # Caching
    cache_transcriptions: bool = True
    cache_dir: str = "transcriptions"
    cache_max_age_days: int = 30

    # Timeouts
    audio_extraction_timeout: int = 60
    gpu_transcription_timeout: int = 300

    # Retries for transient errors such as GPU OOM
    max_retries: int = 2

    # Whisper threading
    whisper_num_workers: int = 1
    cpu_threads: int = 4

    def __post_init__(self):
        if self.max_retries < 0 or self.max_retries > 5:
            raise ValueError(
                f"TranscriptionConfig.max_retries={self.max_retries} "
                f"must be >= 0 and <= 5."
            )
        if self.cache_max_age_days < 1:
            raise ValueError(
                f"TranscriptionConfig.cache_max_age_days={self.cache_max_age_days} "
                f"must be >= 1."
            )