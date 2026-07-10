"""
Tests for src/extract_pedagogy.py — Phase 2 Gemini extraction engine.

These tests mock the Gemini client so they run offline, with no API key and
no network calls. They verify our own logic (env var check, JSON parsing,
error wrapping) rather than Gemini's behavior itself.

Run with:  pytest tests/test_extract_pedagogy.py -v
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import extract_pedagogy  # noqa: E402


def test_missing_api_key_raises_environment_error(monkeypatch):
    """If GEMINI_API_KEY isn't set, _get_client should fail fast with a clear error."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        extract_pedagogy._get_client()


def test_process_transcript_returns_parsed_json(tmp_path, monkeypatch):
    """A well-formed JSON response from Gemini should come back as a Python dict."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    transcript_path = tmp_path / "sample.txt"
    transcript_path.write_text("array wala concept ke baad pointers aate hain", encoding="utf-8")

    fake_response = MagicMock()
    fake_response.text = json.dumps({
        "concepts": ["Arrays", "Pointers"],
        "translations": [
            {"original_term": "array wala concept", "standardized_term": "Array Data Structure"}
        ],
        "dependencies": [{"source": "Arrays", "target": "Pointers"}],
    })

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    with patch("extract_pedagogy._get_client", return_value=fake_client), \
         patch("extract_pedagogy.time.sleep"):  # skip the real 4-second rate-limit guard
        result = extract_pedagogy.process_transcript(str(transcript_path))

    assert result["concepts"] == ["Arrays", "Pointers"]
    assert result["dependencies"] == [{"source": "Arrays", "target": "Pointers"}]


def test_non_json_response_raises_value_error(tmp_path, monkeypatch):
    """If Gemini ever returns non-JSON text, we should get a clear ValueError, not a raw crash."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    transcript_path = tmp_path / "sample.txt"
    transcript_path.write_text("some transcript text", encoding="utf-8")

    fake_response = MagicMock()
    fake_response.text = "Sorry, I can't help with that."  # not valid JSON

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    with patch("extract_pedagogy._get_client", return_value=fake_client), \
         patch("extract_pedagogy.time.sleep"):
        with pytest.raises(ValueError, match="non-JSON response"):
            extract_pedagogy.process_transcript(str(transcript_path))


def test_api_failure_raises_runtime_error(tmp_path, monkeypatch):
    """If the Gemini call itself throws (network error, quota, etc.), we should wrap it in RuntimeError."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    transcript_path = tmp_path / "sample.txt"
    transcript_path.write_text("some transcript text", encoding="utf-8")

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = Exception("connection reset")

    with patch("extract_pedagogy._get_client", return_value=fake_client), \
         patch("extract_pedagogy.time.sleep"):
        with pytest.raises(RuntimeError, match="Gemini API call failed"):
            extract_pedagogy.process_transcript(str(transcript_path))