"""Phase 3 — DAG Visualization: build and render a prerequisite graph from extracted JSON."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_graph_data(file_path: str) -> dict:
    """Read and parse the Phase 2 JSON output, returning it as a dictionary."""
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def build_and_visualize_dag(data: dict, output_image_path: str) -> None:
    """Build a DAG from concepts and dependencies, validate it, and save a plot."""
    graph = nx.DiGraph()

    for concept in data["concepts"]:
        graph.add_node(concept)

    for dep in data["dependencies"]:
        graph.add_edge(dep["source"], dep["target"])

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "Cyclical dependencies detected in extracted data — the prerequisite "
            "graph is not a valid DAG."
        )

    try:
        pos = nx.planar_layout(graph)
    except nx.NetworkXException:
        # planar_layout raises if the graph is non-planar; spring_layout always works
        pos = nx.spring_layout(graph, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(16, 10))

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=3000, node_color="#4A90D9", alpha=0.9)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="#333333",
        width=1.5,
        connectionstyle="arc3,rad=0.1",
    )

    ax.set_title("Prerequisite Dependency Graph (DAG)", fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    fig.tight_layout()

    out = Path(output_image_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[visualize_pedagogy] Graph saved → '{out}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualize_pedagogy.py <input.json> <output.png>")
        sys.exit(1)

    data = load_graph_data(sys.argv[1])
    build_and_visualize_dag(data, sys.argv[2])