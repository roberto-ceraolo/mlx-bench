#!/bin/sh
# Benchmark MLX models on multiple-choice datasets (Apple Silicon only).
# Results are written incrementally — safe to Ctrl-C and resume.
#
# Usage:
#   ./scripts/run_benchmark.sh [options]
#
# Common options:
#   --dataset    mmlu-pro (default), mmlu, arc-challenge, arc-easy
#   --models     bonsai-1.7B, bonsai-4B, bonsai-8B,
#                gemma-4-e2b, ministral-3b, qwen3-4b
#   --n          Number of questions (default: 200)
#
# Examples:
#   ./scripts/run_benchmark.sh --n 50
#   ./scripts/run_benchmark.sh --dataset arc-challenge --models qwen3-4b,ministral-3b
#
# Pass --help to see all options.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"
DEMO_DIR="$(resolve_demo_dir)"
cd "$DEMO_DIR"

if [ "$(uname -s)" != "Darwin" ]; then
    err "MLX only runs on Apple Silicon (macOS)."
    exit 1
fi

ensure_venv "$DEMO_DIR"

# Sync deps (adds datasets if not yet installed)
if ! python -c "import datasets" 2>/dev/null; then
    step "Syncing dependencies (one-time for datasets) ..."
    uv sync
    info "Dependencies up to date."
fi

python "$SCRIPT_DIR/benchmark.py" --demo-dir "$DEMO_DIR" "$@"
