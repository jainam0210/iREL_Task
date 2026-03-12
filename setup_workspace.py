"""Scaffold the local project directory structure for the Phase 2/3 NLP pipeline."""

import os
from pathlib import Path

REQUIREMENTS: str = """\
networkx
matplotlib
google-genai
openai
python-dotenv
"""


def scaffold_project(base_path: str) -> None:
    """Create all directories, stub files, and config files under base_path."""
    base = Path(base_path)

    dirs: list[Path] = [
        base / "data" / "input_transcripts",
        base / "data" / "output_graphs",
        base / "src",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    src_stubs: list[Path] = [
        base / "src" / "extract_pedagogy.py",
        base / "src" / "visualize_pedagogy.py",
        base / "src" / "pipeline.py",
    ]
    for stub in src_stubs:
        stub.touch(exist_ok=True)

    root_stubs: list[Path] = [
        base / "test_pipeline.py",
        base / "README.md",
        base / ".env",
    ]
    for stub in root_stubs:
        stub.touch(exist_ok=True)

    req_file = base / "requirements.txt"
    if not req_file.exists():
        req_file.write_text(REQUIREMENTS, encoding="utf-8")


if __name__ == "__main__":
    cwd = os.getcwd()
    scaffold_project(cwd)
    print(f"[setup] Workspace scaffolded at '{cwd}' — ready for source code.")
