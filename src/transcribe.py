"""Module 2 — Transcription: transcribe a 16 kHz mono WAV to a .txt file using faster-whisper."""

from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

# Avoids reloading large model weights when processing multiple videos in a loop
_MODEL_CACHE: dict[str, WhisperModel] = {}


def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """Return a cached WhisperModel, loading it on first use."""
    cache_key = f"{model_size}:{device}:{compute_type}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _MODEL_CACHE[cache_key]


def transcribe_audio(
    audio_path: str,
    output_text_path: str,
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    language: Optional[str] = None,
) -> None:
    """Transcribe a WAV file with faster-whisper and write the transcript to disk."""
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(
            f"Audio file not found: '{audio}'. "
            "Ensure Module 1 (download_audio) completed successfully."
        )

    output = Path(output_text_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = _get_model(model_size, device, compute_type)

    try:
        segments, info = model.transcribe(
            str(audio),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=True,
            condition_on_previous_text=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"faster-whisper inference failed for '{audio}': {exc}"
        ) from exc

    detected_lang = info.language if info.language else "unknown"
    print(
        f"[transcribe] Detected language: '{detected_lang}' "
        f"(probability: {info.language_probability:.2f})"
    )

    # Lazy iteration avoids holding the full transcript in memory for long recordings
    with output.open("w", encoding="utf-8") as fh:
        for segment in segments:
            fh.write(segment.text.strip() + "\n")

    print(f"[transcribe] Transcript saved → '{output}'")