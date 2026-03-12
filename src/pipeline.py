"""Local Phase 2 + 3 orchestrator: extract pedagogy from a transcript and visualise the DAG."""
import time
import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _run_phase(label: str, cmd: list[str]) -> None:
    """Execute a subprocess command, forward its output, and abort on non-zero exit."""
    log.info("▶ %s", label)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        log.info(result.stdout.strip())
    if result.stderr.strip():
        log.warning(result.stderr.strip())
    if result.returncode != 0:
        log.critical("✗ %s failed (exit %d). Aborting.", label, result.returncode)
        sys.exit(1)
    log.info("✓ %s complete.", label)


def run_local_pipeline(target_name: str) -> None:
    """Run Phase 2 then Phase 3 for the transcript identified by target_name."""
    input_txt  = PROJECT_ROOT / "data" / "input_transcripts" / f"{target_name}.txt"
    output_dir = PROJECT_ROOT / "data" / "output_graphs"
    output_json = output_dir / f"{target_name}_pedagogy.json"
    output_png  = output_dir / f"{target_name}_dag.png"

    if not input_txt.exists():
        log.critical("Input transcript not found: '%s'", input_txt)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    _run_phase(
        "Phase 2 — Pedagogy Extraction",
        [sys.executable, str(PROJECT_ROOT / "src" / "extract_pedagogy.py"),
         str(input_txt), str(output_json)],
    )

    _run_phase(
        "Phase 3 — DAG Visualisation",
        [sys.executable, str(PROJECT_ROOT / "src" / "visualize_pedagogy.py"),
         str(output_json), str(output_png)],
    )

    log.info("Pipeline complete.")
    log.info("  Pedagogy JSON → %s", output_json)
    log.info("  DAG image     → %s", output_png)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run the local Phase 2 + 3 NLP pipeline for a single transcript."
    )
    parser.add_argument(
        "--name", required=True,
        help="Base name of the transcript file in data/input_transcripts/ (without .txt)."
    )
    args = parser.parse_args()

    run_local_pipeline(args.name)
