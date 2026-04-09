"""
Generate scatter plots from benchmark results.
One dot per model for each metric pair.

Plots:
  1. Accuracy (%) vs Peak Memory (GB)
  2. Time to First Token (s) vs Peak Memory (GB)
  3. Tokens per Second vs Peak Memory (GB)

Usage:
    python scripts/plot_benchmark.py [results/benchmark.jsonl] [--output results/plots.png]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else math.nan


def summarise(records: list[dict]) -> dict:
    stats = defaultdict(lambda: {
        "correct": 0, "total": 0,
        "ttft": [], "tps": [], "mem": [],
    })
    for r in records:
        m = r["model"]
        stats[m]["total"]   += 1
        stats[m]["correct"] += bool(r.get("is_correct"))
        if r.get("ttft")           is not None: stats[m]["ttft"].append(r["ttft"])
        if r.get("tps")            is not None: stats[m]["tps"].append(r["tps"])
        if r.get("peak_memory_gb") is not None: stats[m]["mem"].append(r["peak_memory_gb"])
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default="results/benchmark.jsonl")
    parser.add_argument("--output-dir", default="results/plots",
                        help="Directory to write one file per plot")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError:
        print("matplotlib not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    path = Path(args.file)
    if not path.exists():
        print(f"No results file at {path}. Run benchmark.py first.", file=sys.stderr)
        sys.exit(1)

    records = load_records(path)
    if not records:
        print("Results file is empty.", file=sys.stderr)
        sys.exit(1)

    stats  = summarise(records)
    models = sorted(stats.keys())

    # ── Per-model aggregates ──────────────────────────────────────────────
    data = {}
    for m in models:
        s = stats[m]
        data[m] = {
            "acc":  s["correct"] / s["total"] * 100 if s["total"] else math.nan,
            "ttft": avg(s["ttft"]),
            "tps":  avg(s["tps"]),
            "mem":  avg(s["mem"]),
            "n":    s["total"],
        }

    # ── Colour + marker per model ─────────────────────────────────────────
    palette = plt.cm.tab10.colors
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]
    style = {
        m: {"color": palette[i % len(palette)], "marker": markers[i % len(markers)]}
        for i, m in enumerate(models)
    }

    # ── Plot definitions ──────────────────────────────────────────────────
    PLOTS = [
        ("accuracy_vs_memory",  "mem", "acc",  "Peak Memory (GB)", "Accuracy (%)",            "Accuracy vs Memory"),
        ("ttft_vs_memory",      "mem", "ttft", "Peak Memory (GB)", "Time to First Token (s)", "TTFT vs Memory"),
        ("tps_vs_memory",       "mem", "tps",  "Peak Memory (GB)", "Tokens / second",         "Tokens per Second vs Memory"),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legend handles (shared across all plots)
    legend_handles = [
        plt.scatter([], [],
                    color=style[m]["color"], marker=style[m]["marker"],
                    s=80, label=m)
        for m in models
    ]

    for slug, xk, yk, xlabel, ylabel, title in PLOTS:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle("MMLU-Pro Benchmark — model comparison",
                     fontsize=11, fontweight="bold")

        for m in models:
            x = data[m][xk]
            y = data[m][yk]
            if math.isnan(x) or math.isnan(y):
                continue
            ax.scatter(x, y,
                       color=style[m]["color"], marker=style[m]["marker"],
                       s=140, zorder=4)
            txt = ax.annotate(
                m, (x, y),
                textcoords="offset points", xytext=(7, 5),
                fontsize=8.5, color=style[m]["color"],
            )
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=legend_handles, fontsize=8.5,
                  loc="best", frameon=True, framealpha=0.8)

        plt.tight_layout()
        out = out_dir / f"{slug}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
