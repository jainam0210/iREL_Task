"""Phase 2 — Pedagogy Extraction: parse a transcript with Gemini and return structured JSON."""
import time
import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

_SYSTEM_PROMPT = (
    "You are an expert computational linguist and computer science professor. "
    "Your task is to analyze a noisy, heavily code-mixed (Hinglish) educational "
    "transcript and extract its pedagogical structure into a strict JSON format. "
    "You must perform three operations:\n\n"
    "Concept Extraction: Identify the core technical concepts taught.\n\n"
    "Linguistic Standardization: Map any colloquial Indic terms, analogies, or "
    "code-mixed phrasing back to standard English academic terminology.\n\n"
    "Prerequisite Mapping: Establish dependencies based strictly on the pedagogical "
    "flow presented by the teacher (i.e., identifying which concepts must be "
    "understood before moving to the next).\n\n"
    "Your output MUST be a JSON object with exactly three keys:\n"
    '    "concepts": A list of strings representing the standardized academic concepts.\n'
    '    "translations": A list of dictionaries mapping the original Hinglish/colloquial '
    'terms to standardized academic terms, formatted as {"original_term": "...", "standardized_term": "..."}.\n'
    '    "dependencies": A list of dictionaries representing a Directed Acyclic Graph '
    '(DAG) of prerequisites, formatted as {"source": "Prerequisite Concept", '
    '"target": "Dependent Concept"}.\n\n'
    "CRITICAL: The relationships in the 'dependencies' list MUST form a valid "
    "Directed Acyclic Graph (DAG). There cannot be any circular or cyclical dependencies "
    "(e.g., A depends on B, and B depends on A). You must strictly verify that no loops exist before responding.\n"
)

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["concepts", "translations", "dependencies"],
    properties={
        "concepts": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
        ),
        "translations": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["original_term", "standardized_term"],
                properties={
                    "original_term": types.Schema(type=types.Type.STRING),
                    "standardized_term": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        "dependencies": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["source", "target"],
                properties={
                    "source": types.Schema(type=types.Type.STRING),
                    "target": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
    },
)


def _get_client() -> genai.Client:
    """Return a Gemini client authenticated via the GEMINI_API_KEY env variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def process_transcript(file_path: str) -> dict:
    """Read a transcript file, call Gemini with structured output, and return parsed JSON."""
    transcript = Path(file_path).read_text(encoding="utf-8")

    client = _get_client()

    try:
        time.sleep(4)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=transcript,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON response: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_pedagogy.py <input_transcript.txt> <output.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    result: dict = process_transcript(input_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=4), encoding="utf-8")

    print(f"[extract_pedagogy] Output written → '{out}'")