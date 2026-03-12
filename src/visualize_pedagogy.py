"""Phase 3 — DAG Visualization: build and render a prerequisite graph from extracted JSON."""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

# ── Visual constants ──────────────────────────────────────────────────────────
_BG_COLOR    = "#FFFFFF"
_EDGE_COLOR  = "#94A3B8"     # slate-400 — subtle but visible
_ARROW_COLOR = "#475569"     # slate-600 — slightly darker arrow heads
_NODE_FILL   = "#EFF6FF"     # blue-50  — very light blue box fill
_NODE_BORDER = "#3B82F6"     # blue-500 — clean blue border
_TEXT_COLOR  = "#1E293B"     # slate-800 — near-black text
_TITLE_COLOR = "#0F172A"     # slate-900
_FONT        = "DejaVu Sans"

_FONT_SIZE   = 8
_WRAP_WIDTH  = 20            # characters per line inside a box
_PAD_X       = 12            # horizontal padding in points
_PAD_Y       = 6             # vertical padding in points
_BOX_STYLE   = "round,pad=0.3,rounding_size=0.15"


def load_graph_data(file_path: str) -> dict:
    """Read and parse the Phase 2 JSON output, returning it as a dictionary."""
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def _wrap(label: str) -> str:
    """Wrap a concept label so it fits inside a compact box."""
    return "\n".join(textwrap.wrap(label, width=_WRAP_WIDTH))


def _hierarchical_layout(graph: nx.DiGraph) -> dict:
    """
    Compute a top-to-bottom hierarchical layout.

    Prefers Graphviz 'dot' (best DAG layout engine). Falls back to
    kamada_kawai → spring if Graphviz is unavailable.
    """
    try:
        pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")
        # Graphviz returns positions in arbitrary pixel space — normalise
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_range = (x_max - x_min) or 1
            y_range = (y_max - y_min) or 1
            pos = {
                n: ((p[0] - x_min) / x_range, (p[1] - y_min) / y_range)
                for n, p in pos.items()
            }
            return pos
    except Exception:
        pass

    try:
        return nx.kamada_kawai_layout(graph)
    except Exception:
        return nx.spring_layout(graph, seed=42, k=4.0, iterations=100)


def build_and_visualize_dag(data: dict, output_image_path: str) -> None:
    """Build a DAG from concepts and dependencies, validate it, and save a clean plot."""
    graph = nx.DiGraph()

    for concept in data["concepts"]:
        graph.add_node(concept)
    for dep in data["dependencies"]:
        graph.add_edge(dep["source"], dep["target"])

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "Cyclical dependencies detected — the prerequisite graph "
            "is not a valid DAG."
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    pos = _hierarchical_layout(graph)

    # ── Figure sizing — scale with number of nodes ────────────────────────────
    n = len(graph.nodes())
    width  = max(16, min(36, n * 0.7))
    height = max(10, min(48, n * 0.55))
    fig, ax = plt.subplots(figsize=(width, height), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)
    ax.axis("off")

    # ── Draw edges ────────────────────────────────────────────────────────────
    for src, dst in graph.edges():
        sx, sy = pos[src]
        dx, dy = pos[dst]
        ax.annotate(
            "",
            xy=(dx, dy), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=_ARROW_COLOR,
                lw=1.2,
                shrinkA=15, shrinkB=15,
                mutation_scale=14,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=1,
        )

    # ── Draw nodes as text inside auto-sized boxes ────────────────────────────
    for node in graph.nodes():
        x, y = pos[node]
        label = _wrap(node)

        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=_FONT_SIZE, fontfamily=_FONT,
            fontweight="semibold", color=_TEXT_COLOR,
            multialignment="center",
            bbox=dict(
                boxstyle=_BOX_STYLE,
                facecolor=_NODE_FILL,
                edgecolor=_NODE_BORDER,
                linewidth=1.4,
                alpha=0.95,
                pad=0.4,
            ),
            zorder=2,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Prerequisite Dependency Graph (DAG)",
        fontsize=15, fontweight="bold",
        color=_TITLE_COLOR, fontfamily=_FONT,
        pad=16,
    )
    fig.text(
        0.5, 0.005,
        "A → B  means  \"A is a prerequisite of B\"",
        ha="center", fontsize=8,
        color="#64748B", fontfamily=_FONT, style="italic",
    )

    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.98])

    out = Path(output_image_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)

    print(f"[visualize_pedagogy] Graph saved → '{out}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualize_pedagogy.py <input.json> <output.png>")
        sys.exit(1)

    data = load_graph_data(sys.argv[1])
    build_and_visualize_dag(data, sys.argv[2])