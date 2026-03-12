# Code-Mixed Pedagogical Flow Extractor

> **Name:** Jainam Modi &nbsp;|&nbsp; **Roll No.:** 2024101057

An end-to-end NLP pipeline that ingests Hindi-English (*Hinglish*) computer-science
lecture videos from YouTube, transcribes them, extracts their pedagogical structure
using a large language model, and renders the resulting concept-prerequisite graph as
a Directed Acyclic Graph (DAG).

---

## Demonstration

> ### ▶ [Insert Demo Video Link Here]

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why Python?](#2-why-python)
3. [Architecture](#3-architecture)
4. [Why This Flow?](#4-why-this-flow)
5. [Why Phase 1 on Google Colab?](#5-why-phase-1-on-google-colab)
6. [Why a DAG? (And Why Not Something Else?)](#6-why-a-dag-and-why-not-something-else)
7. [Repository Layout](#7-repository-layout)
8. [File-by-File Reference](#8-file-by-file-reference)
9. [Data Flow](#9-data-flow)
10. [Setup & Installation](#10-setup--installation)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [Hardware Notes](#12-hardware-notes)
13. [Output Schema](#13-output-schema)

---

## 1. Project Overview

The task targets **five Hinglish CS lecture videos**  from YouTube. The pipeline is split into two execution environments:

| Environment | What runs there | Why |
|---|---|---|
| **Google Colab (T4 GPU)** | Phase 1 — download + transcribe | `whisper large-v3` requires float16 GPU inference; running it on CPU takes ~10× longer |
| **Local machine** | Phase 2 + Phase 3 — extraction + visualisation | Only API calls and lightweight graph rendering; no GPU needed |

The five source videos:

| # | URL | Channel |
|---|---|---|
| 1 | <https://youtu.be/XV-lIaO00H8> | Apna College |
| 2 | <https://youtu.be/MZdVAVMgNpA> | Gate Smashers |
| 3 | <https://youtu.be/rWFH6PLOIEI> | Gate Smashers |
| 4 | <https://youtu.be/zVjxEIy33Fs> | Gate Smashers |
| 5 | <https://youtu.be/ziyiakWUbaQ> | Gate Smashers |

---

## 2. Why Python?

Python was the natural and only practical choice for this pipeline for several concrete reasons:

**Ecosystem coverage** — every tool this pipeline relies on has a first-class Python
binding: `yt-dlp` (audio download), `faster-whisper` (ASR), `google-genai` (Gemini
API), `networkx` (graph algorithms), and `matplotlib` (rendering). Equivalent
libraries simply do not exist at the same maturity level in any other language.

**ML/AI tooling** — the entire machine learning ecosystem — Whisper, CTranslate2,
Hugging Face, and all LLM SDKs — is Python-first. Using any other language would
mean wrapping Python binaries via subprocess, which eliminates any supposed benefit.

**Rapid iteration** — the pipeline needed to be built and debugged quickly. Python's
interactive REPL and Jupyter notebook support (used for Phase 1 in Colab) allow
immediate inspection of intermediate outputs (audio paths, raw transcript text,
parsed JSON) at each stage without a compile/build step.

**Colab compatibility** — Google Colab natively runs Python notebooks. Phase 1 runs on Colab for GPU access for efficient and quicker processing, so the entire project using Python ensures zero friction when moving code between local development and Colab execution.

**Subprocess orchestration** — Python's `subprocess`, `argparse`, `pathlib`, and
`logging` standard library modules are sufficient to build a production-quality
pipeline orchestrator without any external framework.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1 — Google Colab (T4 GPU)                 │
│                                                                     │
│  YouTube URL                                                        │
│      │                                                              │
│      ▼                                                              │
│  src/ingest.py  ─── yt-dlp + ffmpeg ──►  data/audio/<id>.wav        │
│      │                                                              │
│      ▼                                                              │
│  src/transcribe.py ─ faster-whisper ──►  data/transcripts/<id>.txt  │
│      │               large-v3 float16                               │
│      ▼                                                              │
│  src/utils.py  ─── integrity check ───►  PASS / FAIL                │
│                                                                     │
│  Entry point: src/notebooks/Ingest&transcribe.ipynb                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                 Copy .txt to data/input_transcripts/
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 2+3 — Local Machine                       │
│                                                                     │
│  data/input_transcripts/<name>.txt                                  │
│      │                                                              │
│      ▼                                                              │
│  src/extract_pedagogy.py ── Gemini 2.5-flash ──►  <name>_pedagogy.json │
│      │                       JSON Mode                              │
│      ▼                                                              │
│  src/visualize_pedagogy.py ─ NetworkX + matplotlib ► <name>_dag.png │
│                                                                     │
│  Entry point: python src/pipeline.py --name <name>                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Why This Flow?

The three-phase sequence — **Ingest → Transcribe → Extract → Visualise** — is not
arbitrary. Each step is a hard dependency on the previous one, and the ordering
reflects the minimum viable transformation at each stage:

**Phase 1 must come first because the source is audio, not text.**
LLMs cannot meaningfully process a raw audio waveform. The input must be converted
to natural-language text before any semantic analysis is possible. `ingest.py`
produces a normalised 16 kHz mono WAV (the exact format `faster-whisper` requires),
and `transcribe.py` converts it to a plain `.txt` file that any subsequent step can
read with a single `open()` call. The integrity check in `utils.py` acts as a
hard gate — if the transcript is empty or corrupt, running Phase 2 on it would
waste an API call and produce meaningless output.

**Phase 2 exists because the transcripts are noisy and code-mixed.**
A raw Hinglish transcript contains colloquialisms, mid-sentence language switches,
filler words, and informal analogies that do not map directly to academic CS
terminology. Rule-based NLP (regex, keyword lists, dependency parsing) would fail
on this kind of text. A large language model with a well-crafted system prompt is
the correct tool: it understands the semantic intent of phrases like
*"array wala concept"* and maps them to *"Array Data Structure"* while
simultaneously inferring the pedagogical sequence from the teacher's explanation
order. The structured output (JSON Mode with a typed `response_schema`) ensures the
LLM's answer is machine-parseable without post-processing heuristics.

**Phase 3 comes last because the graph can only be built once the nodes and edges
are known.**
`visualize_pedagogy.py` is a pure consumer of the Phase 2 JSON. It builds the
`DiGraph`, validates it, and renders the PNG. Separating rendering from extraction
means the graph can be re-rendered (different layouts, colours, styles) without
re-calling the API.

**The two-environment split (Colab / local) is a hardware constraint, not a design
preference**.

---

## 5. Why Phase 1 on Google Colab?

Phase 1 (transcription with `whisper large-v3`) was run on Google Colab rather than
locally for the following reasons:

**GPU requirement.** `whisper large-v3` is a 1.5 billion parameter model. At
`float16` precision it occupies ~3 GB of VRAM and runs in roughly 2 minutes per
lecture video on a T4 GPU. Running the same model on a CPU at `int8` quantisation
takes approximately significantly longer time per video and produces slightly lower accuracy
because CTranslate2's INT8 kernel is a lossy approximation of the full weights.
And since Colab provides a free T4, Colab was chose here.

**Memory.** The model alone uses ~6 GB of RAM when loaded (CPU) or ~3 GB VRAM (GPU).
On a machine with other processes running, CPU inference can trigger OOM errors on
transcripts longer than ~30 minutes.

**Colab-specific configuration used in the notebook:**

| Setting | Value | Reason |
|---|---|---|
| `device` | `"cuda"` | Use the T4 GPU |
| `compute_type` | `"float16"` | Full precision on GPU; fits within 16 GB VRAM |
| `condition_on_previous_text` | `False` | Prevents hallucination loops on Hinglish language switches |
| Post-processing | WAV deleted after transcription | Saves Colab disk quota (15 GB cap) |

The transcripts produced in Colab are then downloaded and placed in
`data/input_transcripts/` locally. This is the only manual handoff in the
entire pipeline — everything else is automated.

---

## 6. Why a DAG? (And Why Not Something Else?)

A **Directed Acyclic Graph** is the natural structure for modelling pedagogical
prerequisites because:

* **Directionality** — an edge `A → B` encodes *"A must be understood before B"*,
  which is an inherently asymmetric relationship.
* **Multi-dependency** — a single concept can depend on several prerequisites
  simultaneously (in-degree > 1), which a simple list or tree cannot represent.
* **No cycles** — learning dependencies are never circular; a cycle
  (`A → B → A`) would imply that you need to know A before B *and* B before A,
  which is logically impossible. The DAG constraint enforces this at the data level:
  - The system prompt explicitly instructs Gemini to produce a cycle-free graph.
  - `networkx.is_directed_acyclic_graph()` validates the constraint at runtime and
    raises an error if a cycle slips through.
* **Topological order** — a DAG admits a topological sort, which can directly
  generate a *recommended study sequence* from the extracted graph.

### Why not the other obvious alternatives?

| Alternative | Why it was rejected |
|---|---|
| **Ordered list / sequence** | A linear sequence forces a single chain: concept 1 → 2 → 3 → …. It cannot represent a concept that has *two* prerequisites, e.g., *"Binary Search Trees require both Arrays and Recursion"*. The real prerequisite structure of CS topics is not linear. |
| **Tree** | A tree is a DAG but with the additional constraint that every node has *exactly one parent*. This rules out shared prerequisites (a concept depending on multiple others), which are common in CS curricula. A tree is strictly less expressive. |
| **Undirected graph** | Removing direction loses the asymmetry *"A before B"* vs *"B before A"*. An undirected edge only says the two concepts are *related*, not which one must come first. This makes topological sorting impossible. |
| **Knowledge graph (full RDF/OWL ontology)** | A full ontology supports typed relations, class hierarchies, and inference rules. This is far more than needed: the task only requires *one* relation type ("is a prerequisite of"). A DAG is the minimal correct model; adding ontology machinery adds complexity with no benefit for this use case. |
| **Concept map (cyclic graph allowed)** | Concept maps deliberately allow cycles to represent mutual relationships. That is useful for encyclopaedic knowledge but wrong for pedagogical sequencing — a curriculum cannot have a circular dependency without being unlearnable. |
| **Flat JSON / CSV** | Tabular formats cannot natively represent graph topology. Prerequisite chains of depth > 1 cannot be expressed without either redundancy or a custom traversal convention. NetworkX's `DiGraph` gives topological sort, cycle detection, and layout algorithms for free. |

The DAG is the **minimum structure that is both necessary and sufficient**: directed
(to encode ordering), acyclic (to enforce learnability), and multi-parent (to encode
shared prerequisites).

---

## 7. Repository Layout

```
iREL_Task/
│
├── main.py                          # Phase 1 batch runner (CPU-safe, 5 videos)
├── pipeline.py                      # Full 3-phase orchestrator (--url + --output-dir)
├── setup_workspace.py               # One-time scaffold script
├── requirements.txt                 # All Python dependencies
├── links.txt                        # Raw list of the 5 YouTube URLs
│
├── src/
│   ├── ingest.py                    # Module 1 — yt-dlp audio download
│   ├── transcribe.py                # Module 2 — faster-whisper transcription
│   ├── utils.py                     # Module 3 — transcript integrity assertion
│   ├── extract_pedagogy.py          # Phase 2 — Gemini 2.5-flash extraction
│   ├── visualize_pedagogy.py        # Phase 3 — NetworkX DAG renderer
│   ├── pipeline.py                  # Phase 2+3 local orchestrator
│   │
│   ├── notebooks/
│   │   └── Ingest&transcribe.ipynb  # Colab notebook for Phase 1 (GPU)
│
└── data/
    ├── input_transcripts/           # Drop zone for src/pipeline.py
    └── output_graphs/               # Phase 2+3 output — JSON + PNG
```

---

## 8. File-by-File Reference

### Active source files

| File | Role in the pipeline flow | 
|---|---|
| `src/notebooks/Ingest&transcribe.ipynb` | **Phase 1 entry point (Colab).** Cell 1 installs `yt-dlp` + `faster-whisper`. Cell 2 loops over all 5 URLs: downloads each as a WAV, transcribes with `large-v3` on T4 GPU (`float16`, `condition_on_previous_text=False`), writes `transcripts/video_N.txt`, then deletes the WAV to save Colab disk. | 
| `src/ingest.py` | **Phase 1, Module 1.** `download_audio(video_url, output_dir) → str`. Calls `yt-dlp` with ffmpeg post-processing to produce a 16 kHz mono WAV. Uses `%(id)s` output template for deterministic filenames. Returns the most recently modified WAV (safe for re-runs). | 
| `src/transcribe.py` | **Phase 1, Module 2.** `transcribe_audio(audio_path, output_text_path, ...) → None`. Loads `faster-whisper` large-v3 via a module-level `_MODEL_CACHE` dict to avoid reloading weights across a batch. Writes one line per segment with `vad_filter=True`. | 
| `src/utils.py` | **Phase 1, Module 3 / integrity gate.** `test_transcript_integrity(text_path) → bool`. Asserts file exists, is non-zero bytes, and has non-empty content. Prints word count and a random 50-word Hinglish sample as a human-readable sanity check before Phase 2 consumes the file. | 
| `src/extract_pedagogy.py` | **Phase 2 engine.** `process_transcript(file_path) → dict`. Reads the `.txt` transcript, sends it to **Gemini 2.5-flash** with a typed `response_schema` enforcing JSON Mode (three keys: `concepts`, `translations`, `dependencies`). Includes a 4-second `time.sleep()` as a rate-limit guard between consecutive calls. Raises `ValueError` on non-JSON responses, `RuntimeError` on API failure. | 
| `src/visualize_pedagogy.py` | **Phase 3 renderer.** `build_and_visualize_dag(data, output_image_path) → None`. Reads the Phase 2 JSON, builds a `networkx.DiGraph`, calls `is_directed_acyclic_graph()` to validate, tries `planar_layout` then falls back to `spring_layout(seed=42, k=2.5)` for non-planar graphs, and saves a 150 DPI PNG. | 
| `src/pipeline.py` | **Phase 2+3 local entry point.** `python src/pipeline.py --name <stem>`. Resolves `data/input_transcripts/<stem>.txt` as input and `data/output_graphs/<stem>_pedagogy.json` / `<stem>_dag.png` as outputs. Chains Phase 2 → Phase 3 via `_run_phase()` subprocess calls with structured logging. | 
| `requirements.txt` | Pinned Python dependencies for the full pipeline. | 
| `links.txt` | Plain-text list of the 5 YouTube URLs. No code; reference only. | 



## 9. Data Flow

```
[Google Colab — Phase 1]
  src/notebooks/Ingest&transcribe.ipynb
      │  downloads + transcribes 5 videos
      ▼
  transcripts/video_1.txt ... video_5.txt
      │
      │  ← manually copy to local machine →
      ▼
  data/input_transcripts/video_1.txt ... video_5.txt

[Local Machine — Phase 2 + 3]
  python src/pipeline.py --name video_1
      │
      ├─ src/extract_pedagogy.py  →  data/output_graphs/video_1_pedagogy.json
      │
      └─ src/visualize_pedagogy.py →  data/output_graphs/video_1_dag.png
```

Repeat `--name video_2` through `--name video_5` for the remaining transcripts.

---

## 10. Setup & Installation

### Prerequisites

```bash
# System dependency — required by yt-dlp for audio conversion
sudo apt install ffmpeg         
```

### Python environment

```bash
git clone <URL/SSH key>
cd iREL_Task

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### API key

```bash
export GEMINI_API_KEY="your_key_here"

# To persist across sessions:
echo 'export GEMINI_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Setup data directories

```bash
python setup_workspace.py
```

---

## 11. Running the Pipeline

### Phase 1 — Transcription (Google Colab, recommended)

1. Open `src/notebooks/Ingest&transcribe.ipynb` in Google Colab.
2. Set the runtime to **T4 GPU** (*Runtime → Change runtime type → T4 GPU*).
3. Run Cell 1 to install dependencies.
4. Run Cell 2 — it downloads all 5 videos and saves transcripts to `transcripts/`.
5. Download the `.txt` files and place them in `data/input_transcripts/` locally.

### Phase 2 + 3 — Extraction and Visualisation

```bash
# Process a single transcript by its file stem
python src/pipeline.py --name video_1

# Outputs:
#   data/output_graphs/video_1_pedagogy.json
#   data/output_graphs/video_1_dag.png
```

---

## 12. Hardware Notes

| Phase | Recommended hardware | Minimum hardware | Approximate time per video |
|---|---|---|---|
| Phase 1 (transcription) | Colab T4 GPU, `float16` | Any CPU, `int8` | ~2 min (GPU) / ~12 min (CPU) |
| Phase 2 (Gemini API) | Any machine with internet | Same | ~10–20 sec (API round-trip + 4 sec rate-limit guard) |
| Phase 3 (DAG render) | Any machine | Same | < 2 sec |

**Why `float16` in the notebook and not `int8`?**  
The Colab notebook runs `faster-whisper` with `device="cuda", compute_type="float16"`
to exploit the 16 GB VRAM of the T4 GPU for maximum accuracy and speed.

**Why `condition_on_previous_text=False` in the notebook?**  
Hinglish audio produces noisy, code-mixed segments. Conditioning on previous text
can cause Whisper to hallucinate repetitive loops once it gets confused by a
language switch. Disabling it resets the context at each segment boundary.

---

## 13. Output Schema

`data/output_graphs/<name>_pedagogy.json`

```json
{
    "concepts": [
        "Arrays",
        "Pointers",
        "Dynamic Memory Allocation"
    ],
    "translations": [
        {
            "original_term": "array wala concept",
            "standardized_term": "Array Data Structure"
        }
    ],
    "dependencies": [
        { "source": "Arrays",   "target": "Pointers" },
        { "source": "Pointers", "target": "Dynamic Memory Allocation" }
    ]
}
```

| Key | Type | Description |
|---|---|---|
| `concepts` | `string[]` | Standardised academic concept names extracted from the lecture |
| `translations` | `{original_term, standardized_term}[]` | Hinglish / colloquial → formal English mappings |
| `dependencies` | `{source, target}[]` | DAG edges encoding `source` is a prerequisite of `target` |

The `dependencies` array is validated as a proper DAG (no cycles) by both:
1. The Gemini system prompt (*"CRITICAL: … MUST form a valid DAG"*)
2. `networkx.is_directed_acyclic_graph()` at render time

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `yt-dlp` | ≥ 2024.1.1 | YouTube audio download |
| `faster-whisper` | ≥ 1.0.0 | CTranslate2-backed Whisper ASR |
| `google-genai` | ≥ 1.0.0 | Gemini API client (structured output) |
| `networkx` | ≥ 3.2.0 | DAG modeling and cycle detection |
| `matplotlib` | ≥ 3.8.0 | Graph rendering and PNG export |
| `ctranslate2` | ≥ 4.0.0 | Backend engine for faster-whisper |
| `huggingface-hub` | ≥ 0.20.0 | Model weight downloads |
| `tokenizers` | ≥ 0.15.0 | Rust tokenizer used by faster-whisper |

System dependency: **ffmpeg** (not in `requirements.txt`; install via OS package manager).
