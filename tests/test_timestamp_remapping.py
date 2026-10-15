"""Tests for remapping transcript timestamps onto the original timeline.

When silence is trimmed out, the model reports timestamps against the shortened
audio. These tests pin down how `remap_segments_to_original_time` moves them
back using the speech regions the trimmer found.

Note on boundaries: adjacent regions overlap by the crossfade length, and a
timestamp landing exactly on a region's end satisfies that region's `t <= end`
test. The first match wins, so such a timestamp resolves to the region that is
ending. Tests here place timestamps clearly inside or clearly past a region
rather than on the seam.
"""

from src.transcription.utils import remap_segments_to_original_time


def test_single_region_offset_is_applied():
    """With one speech region, every timestamp shifts by the removed lead-in."""
    # Original: 5s of silence, then speech from 5s to 12s.
    # Trimmed: speech starts at 0.
    segments = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    regions = [(5000, 12000)]

    mapped = remap_segments_to_original_time(segments, regions, crossfade_ms=0)

    assert mapped[0]["start"] == 5.0
    assert mapped[0]["end"] == 7.0
    assert mapped[0]["text"] == "hello"


def test_two_regions_shift_the_second_one():
    """A segment past the first region is offset by that region's start."""
    # Region A: 0-5s.  Region B: 13-18s.  In trimmed time B starts at 5000ms.
    # A segment at 6-8s trimmed is 1s into region B -> 14-16s original.
    segments = [
        {"start": 0.0, "end": 3.0, "text": "first"},
        {"start": 6.0, "end": 8.0, "text": "second"},
    ]
    regions = [(0, 5000), (13000, 18000)]

    mapped = remap_segments_to_original_time(segments, regions, crossfade_ms=0)

    assert mapped[0]["start"] == 0.0
    assert mapped[0]["end"] == 3.0
    assert mapped[1]["start"] == 14.0
    assert mapped[1]["end"] == 16.0


def test_crossfade_advances_the_second_region():
    """A crossfade starts the next region slightly early in trimmed time."""
    # Region A: 0-3s, region B: 5-8s, joined with a 500ms crossfade.
    # In trimmed time region B starts at 2500ms, not 3000ms.
    segments = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 3.5, "end": 5.0, "text": "b"},
    ]
    regions = [(0, 3000), (5000, 8000)]

    mapped = remap_segments_to_original_time(segments, regions, crossfade_ms=500)

    assert mapped[0]["start"] == 0.0
    # 3500ms trimmed is 1000ms into region B -> 5000 + 1000 = 6000ms.
    assert mapped[1]["start"] == 6.0
    assert mapped[1]["end"] == 7.5


def test_past_all_regions_clamps_to_the_last_end():
    """A timestamp beyond the final region clamps to that region's end."""
    segments = [{"start": 99.0, "end": 120.0, "text": "late"}]
    regions = [(0, 3000), (5000, 8000)]

    mapped = remap_segments_to_original_time(segments, regions, crossfade_ms=0)

    assert mapped[0]["start"] == 8.0
    assert mapped[0]["end"] == 8.0


def test_empty_inputs_pass_through_untouched():
    """No segments or no regions means there is nothing to remap."""
    assert remap_segments_to_original_time([], [(0, 1000)]) == []
    segments = [{"start": 1.0, "end": 2.0, "text": "x"}]
    assert remap_segments_to_original_time(segments, []) == segments


def test_other_segment_fields_are_preserved():
    """Remapping touches timing only."""
    segments = [{"start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}]
    mapped = remap_segments_to_original_time(segments, [(0, 5000)], 50)
    assert mapped[0]["text"] == "hi"
    assert mapped[0]["confidence"] == 0.9