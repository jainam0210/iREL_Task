"""
Tests for src/utils.py — the Phase 1 transcript integrity gate.

Run with:  pytest tests/test_utils.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import test_transcript_integrity as check_transcript_integrity  # noqa: E402


def test_valid_transcript_passes(tmp_path):
    """A normal, non-empty transcript (>= 50 words) should pass and return True."""
    text_path = tmp_path / "transcript.txt"
    text_path.write_text(" ".join(["word"] * 100), encoding="utf-8")

    assert check_transcript_integrity(str(text_path)) is True


def test_missing_file_raises_file_not_found_error(tmp_path):
    """A transcript path that doesn't exist should raise FileNotFoundError with a helpful message."""
    missing_path = tmp_path / "does_not_exist.txt"

    with pytest.raises(FileNotFoundError, match="Transcript not found"):
        check_transcript_integrity(str(missing_path))


def test_empty_file_raises_assertion_error(tmp_path):
    """A 0-byte transcript file should raise an AssertionError, not silently pass through."""
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(AssertionError, match="0 bytes"):
        check_transcript_integrity(str(empty_path))


def test_short_transcript_under_50_words_has_a_bug(tmp_path):
    """
    KNOWN BUG: when the transcript has fewer than 50 words, `start_index` is
    never assigned (it's only set inside the `if word_count >= chunk_size`
    branch), but the final print statement references it unconditionally.
    This currently raises a NameError instead of printing the short sample.

    This test documents the current (buggy) behavior. Once utils.py is fixed
    to handle short transcripts gracefully, update this test to assert
    `check_transcript_integrity(...) is True` instead.
    """
    short_path = tmp_path / "short.txt"
    short_path.write_text(" ".join(["word"] * 10), encoding="utf-8")

    with pytest.raises(NameError, match="start_index"):
        check_transcript_integrity(str(short_path))


def test_exactly_50_words_does_not_hit_the_short_path(tmp_path):
    """Boundary check: exactly 50 words should take the >= branch and pass cleanly."""
    boundary_path = tmp_path / "boundary.txt"
    boundary_path.write_text(" ".join(["word"] * 50), encoding="utf-8")

    assert check_transcript_integrity(str(boundary_path)) is True