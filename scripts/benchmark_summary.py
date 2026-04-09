"""
Print a summary table from benchmark results.

Usage:
    python scripts/benchmark_summary.py [results/benchmark.jsonl] [--by-category]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def avg(lst):
    return sum(lst) / len(lst) if lst else math.nan


def fmt(val, fmt_str):
    return format(val, fmt_str) if not math.isnan(val) else "—"


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


def summarise(records: list[dict]) -> dict:
    stats = defaultdict(lambda: {
        "correct": 0, "total": 0, "parse_failures": 0,
        "ttft": [], "tps": [], "mem": [], "gen_tok": [],
    })
    for r in records:
        m = r["model"]
        stats[m]["total"]          += 1
        stats[m]["correct"]        += bool(r.get("is_correct"))
        stats[m]["parse_failures"] += bool(r.get("parse_failed"))
        if r.get("ttft")              is not None: stats[m]["ttft"].append(r["ttft"])
        if r.get("tps")               is not None: stats[m]["tps"].append(r["tps"])
        if r.get("peak_memory_gb")    is not None: stats[m]["mem"].append(r["peak_memory_gb"])
        if r.get("generation_tokens") is not None: stats[m]["gen_tok"].append(r["generation_tokens"])
    return stats


def print_table(stats: dict):
    HDR = f"{'Model':<25} {'N':>5} {'Acc %':>6} {'Parse?':>7} {'TTFT s':>7} {'TPS':>7} {'MemGB':>7} {'GenTok':>7}"
    SEP = "─" * len(HDR)
    print(f"\n{HDR}")
    print(SEP)
    for model, s in sorted(stats.items(), key=lambda x: -x[1]["correct"] / max(x[1]["total"], 1)):
        acc  = s["correct"] / s["total"] * 100 if s["total"] else 0
        pf   = s["parse_failures"]
        pf_s = format(pf, ">7") if pf else "      —"
        print(
            f"{model:<25} {s['total']:>5} {acc:>6.1f}"
            f" {pf_s}"
            f" {fmt(avg(s['ttft']), '>7.2f')}"
            f" {fmt(avg(s['tps']),  '>7.1f')}"
            f" {fmt(avg(s['mem']),  '>7.2f')}"
            f" {fmt(avg(s['gen_tok']), '>7.1f')}"
        )
    print()


def print_by_category(records: list[dict]):
    # Group by (model, category)
    cat_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in records:
        cat_stats[r["model"]][r.get("category", "unknown")]["total"]   += 1
        cat_stats[r["model"]][r.get("category", "unknown")]["correct"] += bool(r.get("is_correct"))

    for model in sorted(cat_stats):
        print(f"\n  {model}")
        rows = sorted(cat_stats[model].items(), key=lambda x: -x[1]["correct"] / max(x[1]["total"], 1))
        for cat, s in rows:
            acc = s["correct"] / s["total"] * 100 if s["total"] else 0
            bar_len = int(acc / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {cat:<30} {s['correct']:>3}/{s['total']:<3}  {acc:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default="results/benchmark.jsonl")
    parser.add_argument("--by-category", action="store_true")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"No results file at {path}. Run benchmark.py first.", file=sys.stderr)
        sys.exit(1)

    records = load_records(path)
    if not records:
        print("File is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} result(s) from {path}")
    stats = summarise(records)
    print_table(stats)

    if args.by_category:
        print("Per-category breakdown:")
        print_by_category(records)
        print()


if __name__ == "__main__":
    main()
