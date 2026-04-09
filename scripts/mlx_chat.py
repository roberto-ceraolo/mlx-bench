"""Interactive multi-turn chat with Bonsai MLX model."""
import os
import sys

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import make_sampler

_SIZE = os.environ.get("BONSAI_MODEL", "8B")
_DEFAULT_MODEL = f"models/Bonsai-{_SIZE}-mlx"

CYAN = "\033[36m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

MODEL_DIR = os.environ.get("BONSAI_MLX_MODEL", _DEFAULT_MODEL)
MAX_TOKENS = int(os.environ.get("BONSAI_MAX_TOKENS", "512"))
TEMP = float(os.environ.get("BONSAI_TEMP", "0.5"))
TOP_P = float(os.environ.get("BONSAI_TOP_P", "0.85"))


def main():
    print(f"\n{BOLD}Loading {MODEL_DIR}...{RESET}", flush=True)
    model, tokenizer = load(MODEL_DIR)
    sampler = make_sampler(temp=TEMP, top_p=TOP_P)

    print(f"{BOLD}Bonsai chat — type {CYAN}/exit{RESET}{BOLD} or Ctrl-C to quit, {CYAN}/reset{RESET}{BOLD} to clear history.{RESET}\n")

    history = []

    while True:
        try:
            user_input = input(f"{CYAN}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            print(f"{DIM}Bye!{RESET}")
            break
        if user_input.lower() == "/reset":
            history = []
            print(f"{DIM}History cleared.{RESET}\n")
            continue

        history.append({"role": "user", "content": user_input})

        chat_prompt = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )

        print(f"{GREEN}Bonsai: {RESET}", end="", flush=True)

        assistant_reply = []
        last = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=chat_prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
        ):
            sys.stdout.write(response.text)
            sys.stdout.flush()
            assistant_reply.append(response.text)
            last = response

        print("\n")

        history.append({"role": "assistant", "content": "".join(assistant_reply)})

        if last:
            peak_gb = mx.get_peak_memory() / (1024**3)
            print(
                f"{DIM}[{last.generation_tokens} tokens · {last.generation_tps:.1f} t/s · peak {peak_gb:.2f} GB]{RESET}\n"
            )


if __name__ == "__main__":
    main()
