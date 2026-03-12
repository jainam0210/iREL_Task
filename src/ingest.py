"""Module 1 — Data Ingestion: download a YouTube audio stream as a 16 kHz mono WAV."""

import subprocess
from pathlib import Path


def download_audio(video_url: str, output_dir: str) -> str:
    """Download and convert a YouTube video's audio to a 16 kHz mono WAV, returning its path."""
    if not isinstance(video_url, str) or not video_url.strip():
        raise ValueError(f"Invalid video_url received: {video_url!r}")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # %(id)s ensures deterministic, collision-free filenames across multiple videos
    output_template = str(output_path / "%(id)s.%(ext)s")

    command: list[str] = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args",
        "ffmpeg:-ar 16000 -ac 1",
        "--output", output_template,
        "--quiet",
        "--no-warnings",
        video_url,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise subprocess.CalledProcessError(
            exc.returncode,
            exc.cmd,
            output=exc.stdout,
            stderr=f"yt-dlp failed for {video_url!r}.\nstderr: {exc.stderr.strip()}",
        ) from exc

    wav_files = sorted(output_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"Expected a .wav file in '{output_path}' after yt-dlp ran, "
            "but none was found. Verify that ffmpeg is installed and in PATH."
        )

    # max by mtime handles re-runs where a previous .wav already exists
    latest_wav = max(wav_files, key=lambda p: p.stat().st_mtime)
    return str(latest_wav)