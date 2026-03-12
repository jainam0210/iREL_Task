"""Module 3 — Test 1: assert transcript existence, non-emptiness, word count, and a random sample."""

import os
import random
from pathlib import Path


def test_transcript_integrity(text_path: str) -> bool:
    """Validate transcript existence, non-emptiness, and print a random 50-word sample."""
    path = Path(text_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"[integrity] Transcript not found at '{path}'. "
            "Ensure Module 2 (transcribe_audio) ran without errors."
        )

    file_size_bytes: int = os.path.getsize(path)
    assert file_size_bytes > 0, (
        f"[integrity] FAIL — '{path}' exists but is 0 bytes."
    )

    content: str = path.read_text(encoding="utf-8")
    char_count: int = len(content)
    assert char_count > 0, (
        f"[integrity] FAIL — '{path}' has {file_size_bytes} bytes but "
        "decoded to an empty string (possible encoding issue)."
    )

    words: list[str] = content.split()
    word_count: int = len(words)
    print(f"[integrity] File      : '{path.name}'")
    print(f"[integrity] Size      : {file_size_bytes:,} bytes")
    print(f"[integrity] Characters: {char_count:,}")
    print(f"[integrity] Word count: {word_count:,}")

    chunk_size: int = 50
    if word_count >= chunk_size:
        start_index: int = random.randint(0, word_count - chunk_size)
        chunk: str = " ".join(words[start_index : start_index + chunk_size])
    else:
        chunk = content.strip()
        print(
            f"[integrity] Note: transcript has only {word_count} words "
            f"(< {chunk_size}); printing full content as sample."
        )

    print(
        f"\n[integrity] ── Random 50-word sample (words {start_index}–"
        f"{start_index + chunk_size - 1 if word_count >= chunk_size else word_count - 1}) ──\n"
        f"{chunk}\n"
        "─────────────────────────────────────────────────────────────────\n"
    )

    return True