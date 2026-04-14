#!/bin/sh
# Interactive multi-turn chat with Bonsai MLX model (Apple Silicon only)
# Usage: BONSAI_MODEL=1.7B ./scripts/chat_mlx.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"
assert_valid_model
DEMO_DIR="$(resolve_demo_dir)"
cd "$DEMO_DIR"

if [ "$(uname -s)" != "Darwin" ]; then
    err "MLX only runs on Apple Silicon (macOS). Use ./scripts/run_llama.sh instead."
    exit 1
fi

assert_mlx_downloaded
ensure_venv "$DEMO_DIR"

export BONSAI_MLX_MODEL="$MLX_MODEL_DIR"

# Optional overrides via flags
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--max-tokens) export BONSAI_MAX_TOKENS="$2"; shift 2 ;;
        --temp)          export BONSAI_TEMP="$2"; shift 2 ;;
        --top-p)         export BONSAI_TOP_P="$2"; shift 2 ;;
        *) shift ;;
    esac
done

"$DEMO_DIR/.venv/bin/python3" "$SCRIPT_DIR/mlx_chat.py"
