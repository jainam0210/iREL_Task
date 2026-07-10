"""
Tests for src/visualize_pedagogy.py — Phase 3 DAG builder + renderer.

Run with:  pytest tests/test_visualize_pedagogy.py -v
"""
import sys
from pathlib import Path

import pytest
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visualize_pedagogy import build_and_visualize_dag, load_graph_data  # noqa: E402


VALID_DATA = {
    "concepts": ["Arrays", "Pointers", "Dynamic Memory Allocation"],
    "translations": [
        {"original_term": "array wala concept", "standardized_term": "Array Data Structure"}
    ],
    "dependencies": [
        {"source": "Arrays", "target": "Pointers"},
        {"source": "Pointers", "target": "Dynamic Memory Allocation"},
    ],
}

CYCLIC_DATA = {
    "concepts": ["A", "B", "C"],
    "translations": [],
    "dependencies": [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
        {"source": "C", "target": "A"},  # closes the loop -> not a DAG
    ],
}


def test_valid_dag_renders_a_png(tmp_path):
    """A well-formed acyclic dependency list should render a non-empty PNG file."""
    output_path = tmp_path / "graph.png"

    build_and_visualize_dag(VALID_DATA, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cyclic_dependencies_raise_value_error(tmp_path):
    """A cyclic dependency list (A->B->C->A) must be rejected before any rendering happens."""
    output_path = tmp_path / "should_not_exist.png"

    with pytest.raises(ValueError, match="not a valid DAG"):
        build_and_visualize_dag(CYCLIC_DATA, str(output_path))

    assert not output_path.exists()


def test_single_node_no_edges_still_renders(tmp_path):
    """Edge case: a transcript with one concept and no dependencies should still render fine."""
    data = {"concepts": ["Solo Concept"], "translations": [], "dependencies": []}
    output_path = tmp_path / "solo.png"

    build_and_visualize_dag(data, str(output_path))

    assert output_path.exists()


def test_load_graph_data_reads_json_file(tmp_path):
    """load_graph_data should correctly parse a JSON file back into a dict."""
    import json

    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(VALID_DATA), encoding="utf-8")

    result = load_graph_data(str(json_path))

    assert result == VALID_DATA


@pytest.mark.parametrize("sample_name", ["video_1", "video_2", "video_3", "video_4", "video_5"])
def test_existing_sample_outputs_are_valid_dags(sample_name):
    """
    Regression check: every pedagogy JSON already committed under
    data/output_graphs/ should still form a valid DAG. This guards against a
    prompt or schema change silently breaking previously-generated outputs.
    """
    sample_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "output_graphs"
        / f"{sample_name}_pedagogy.json"
    )
    if not sample_path.exists():
        pytest.skip(f"{sample_path.name} not present in this checkout")

    data = load_graph_data(str(sample_path))

    graph = nx.DiGraph()
    for concept in data["concepts"]:
        graph.add_node(concept)
    for dep in data["dependencies"]:
        graph.add_edge(dep["source"], dep["target"])

    assert nx.is_directed_acyclic_graph(graph), (
        f"{sample_path.name} contains a cyclic dependency"
    )