"""
Benchmark MLX models on multiple-choice datasets (Apple Silicon only).

Writes results incrementally — safe to stop (Ctrl-C) and resume at any time.
Already-completed (model, question_id) pairs are skipped on restart.

Usage:
    python scripts/benchmark.py [options]

Options:
    --dataset  Benchmark dataset (default: mmlu-pro)
               Built-in: mmlu-pro, mmlu, arc-challenge, arc-easy
               To add your own, edit the DATASETS dict below.
    --models   Comma-separated model keys to run (default: bonsai-8B,gemma-4-e2b,ministral-3b,qwen3-4b)
               Built-in keys: bonsai-1.7B, bonsai-4B, bonsai-8B, gemma-4-e2b, ministral-3b, qwen3-4b
               To add your own, edit the MODELS dict below.
    --n        Number of questions to sample (default: 200)
    --seed     Random seed for reproducible sampling (default: 42)
    --max-tokens  Max tokens per response (default: 1024)
    --temp     Override sampling temperature for all models (default: use each model's recommended temp)
    --output   Output JSONL file (default: results/benchmark.jsonl)
    --questions-cache  Cache file for sampled questions (default: results/<dataset>_sample.json)
    --demo-dir Path to the repo root (default: parent of this script's directory)
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import make_sampler

# ── Model registry ────────────────────────────────────────────────────────────
# Add any MLX-compatible model here — either a HuggingFace repo ID or a local
# path relative to the repo root (must start with "models/").
#
# HuggingFace models are downloaded automatically on first run.
# Local models (Bonsai) require ./setup.sh to have been run first.
#
# To add your own model:
#   "my-model": {"path": "mlx-community/Llama-3.2-3B-Instruct-4bit", "temp": 0.0},
#
# temp: creator-recommended sampling temperature (0.0 = greedy / deterministic).

MODELS = {
    # ── Bonsai (local — requires ./setup.sh) ──────────────────────────────
    "bonsai-1.7B":  {"path": "models/Bonsai-1.7B-mlx",                    "temp": 0.5},
    "bonsai-4B":    {"path": "models/Bonsai-4B-mlx",                       "temp": 0.5},
    "bonsai-8B":    {"path": "models/Bonsai-8B-mlx",                       "temp": 0.5},
    # ── Community models (downloaded automatically from HuggingFace) ──────
    "gemma-4-e2b":  {"path": "unsloth/gemma-4-E2B-it-UD-MLX-4bit",         "temp": 1.0},
    "ministral-3b": {"path": "mlx-community/Ministral-3-3B-Instruct-2512-4bit", "temp": 0.1},
    "qwen3-4b":     {"path": "Qwen/Qwen3-4B-MLX-4bit",                     "temp": 0.7},
}

OPTION_LETTERS = "ABCDEFGHIJ"

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each adapter receives a raw HuggingFace row (dict) and its 0-based index and
# must return a normalised dict with these keys:
#   question_id : str   — stable unique ID (used for resume + seeding)
#   category    : str   — subject/topic (used for stratified sampling)
#   question    : str   — question text
#   options     : list  — answer choices (will be labelled A, B, C, …)
#   answer      : str   — correct letter (A–J) matching position in options
#
# To add a new dataset, write an adapter and add an entry to DATASETS below.

def _adapt_mmlu_pro(row: dict, idx: int) -> dict:
    return {
        "question_id": str(row["question_id"]),
        "category":    row["category"],
        "question":    row["question"],
        "options":     row["options"],
        "answer":      row["answer"],
    }

def _adapt_mmlu(row: dict, idx: int) -> dict:
    return {
        "question_id": str(idx),
        "category":    row["subject"].replace("_", " "),
        "question":    row["question"],
        "options":     row["choices"],
        "answer":      OPTION_LETTERS[int(row["answer"])],
    }

def _adapt_arc(row: dict, idx: int) -> dict:
    labels     = row["choices"]["label"]
    texts      = row["choices"]["text"]
    answer_key = row["answerKey"]
    # answerKey may be "A"–"D" or "1"–"4" depending on the question
    if answer_key.isdigit():
        answer_idx = int(answer_key) - 1
    else:
        answer_idx = labels.index(answer_key) if answer_key in labels else 0
    return {
        "question_id": row["id"],
        "category":    "science",
        "question":    row["question"],
        "options":     texts,
        "answer":      OPTION_LETTERS[answer_idx],
    }

DATASETS: dict[str, dict] = {
    "mmlu-pro": {
        "hf_id":   "TIGER-Lab/MMLU-Pro",
        "config":  None,
        "split":   "test",
        "adapter": _adapt_mmlu_pro,
        "about":   "MMLU-Pro — expert-level questions across 14 categories",
    },
    "mmlu": {
        "hf_id":   "cais/mmlu",
        "config":  "all",
        "split":   "test",
        "adapter": _adapt_mmlu,
        "about":   "MMLU — multiple-choice questions across 57 subjects",
    },
    "arc-challenge": {
        "hf_id":   "allenai/ai2_arc",
        "config":  "ARC-Challenge",
        "split":   "test",
        "adapter": _adapt_arc,
        "about":   "ARC-Challenge — grade-school science questions (hard subset)",
    },
    "arc-easy": {
        "hf_id":   "allenai/ai2_arc",
        "config":  "ARC-Easy",
        "split":   "test",
        "adapter": _adapt_arc,
        "about":   "ARC-Easy — grade-school science questions (easy subset)",
    },
}

# System prompt: dataset-agnostic (works for any multiple-choice benchmark).
SYSTEM_PROMPT = (
    "You are an expert at answering multiple-choice questions. "
    "Think step by step, then provide your final answer in the format "
    '"The answer is (X)" where X is the letter of the correct option.'
)

# ── ANSI colours ─────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
DIM   = "\033[2m"
RESET = "\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stable_int(s: str) -> int:
    """Deterministic int from any string (for per-question seeding)."""
    try:
        return int(s)
    except ValueError:
        return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def format_prompt(question: str, options: list[str]) -> str:
    """Standard multiple-choice format: question + lettered options."""
    opts = "\n".join(f"{OPTION_LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"Question: {question}\n\nOptions:\n{opts}"


def extract_answer(text: str) -> str | None:
    """
    Parse the model's answer (CoT format).
    Priority:
      1. 'The answer is (X)' — canonical CoT format
      2. 'The answer is X'
      3. First standalone A-J letter (fallback)
    """
    t = text.strip()
    m = re.search(r"[Tt]he answer is\s*\(([A-J])\)", t)
    if m:
        return m.group(1)
    m = re.search(r"[Tt]he answer is\s+([A-J])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-J])\b", t)
    return m.group(1) if m else None


def _stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n rows preserving the category distribution."""
    import math
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)

    total = len(rows)
    rng = random.Random(seed)

    raw    = {cat: n * len(items) / total for cat, items in by_cat.items()}
    floors = {cat: math.floor(v) for cat, v in raw.items()}
    remainder = n - sum(floors.values())
    order  = sorted(raw, key=lambda c: raw[c] - floors[c], reverse=True)
    quotas = {cat: floors[cat] + (1 if i < remainder else 0)
              for i, cat in enumerate(order)}

    sampled: list[dict] = []
    for cat, items in by_cat.items():
        k = min(quotas[cat], len(items))
        sampled.extend(rng.sample(items, k))

    rng.shuffle(sampled)
    return sampled


def load_questions(n: int, seed: int, cache_file: Path, dataset_key: str) -> list[dict]:
    """
    Download the requested dataset once (cached by HuggingFace locally),
    normalise rows via the dataset adapter, then apply stratified sampling.
    """
    if cache_file.exists():
        print(f"{DIM}Using cached questions from {cache_file}{RESET}")
        with open(cache_file) as f:
            return json.load(f)

    from datasets import load_dataset

    ds_cfg  = DATASETS[dataset_key]
    adapter = ds_cfg["adapter"]
    print(f"Loading {ds_cfg['about']} (downloaded once, cached locally)...")

    load_kwargs: dict = {"split": ds_cfg["split"]}
    if ds_cfg["config"]:
        ds = load_dataset(ds_cfg["hf_id"], ds_cfg["config"], **load_kwargs)
    else:
        ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)

    rows = [adapter(dict(row), idx) for idx, row in enumerate(ds)]
    questions = _stratified_sample(rows, min(n, len(rows)), seed)

    from collections import Counter
    cat_counts = Counter(q["category"] for q in questions)
    print(f"{GREEN}Sampled {len(questions)} questions across "
          f"{len(cat_counts)} categories (stratified, seed={seed}){RESET}")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:<35} {cnt:>3}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"{DIM}Cached → {cache_file}{RESET}")
    return questions


def load_done(
    output_file: Path,
    questions: list[dict] | None = None,
    model_keys: list[str] | None = None,
) -> set[tuple[str, str]]:
    """
    Return already-completed (model_key, question_id) pairs.

    If questions and model_keys are supplied, also scans every other *.jsonl
    file in the same directory and imports matching records into output_file.
    This lets a larger run (N=200) reuse results from a smaller prior run
    (N=100) without re-running any shared questions.
    """
    done: set[tuple[str, str]] = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], str(r["question_id"])))
                except Exception:
                    pass

    if not questions or not model_keys:
        return done

    question_ids = {str(q["question_id"]) for q in questions}
    model_set    = set(model_keys)
    to_import: dict[tuple[str, str], str] = {}  # key → raw JSON line

    for jsonl_file in sorted(output_file.parent.glob("*.jsonl")):
        if jsonl_file.resolve() == output_file.resolve():
            continue
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        continue
                    try:
                        r   = json.loads(line)
                        key = (r["model"], str(r["question_id"]))
                        if (r.get("model") in model_set
                                and str(r.get("question_id")) in question_ids
                                and key not in done
                                and key not in to_import):
                            to_import[key] = line
                    except Exception:
                        pass
        except Exception:
            pass

    if to_import:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            for line in to_import.values():
                f.write(line + "\n")
        done |= to_import.keys()
        print(f"{GREEN}Imported {len(to_import)} result(s) from previous run(s) — "
              f"skipping those questions.{RESET}")

    return done


def clear_mlx_cache():
    """Release pooled Metal buffers back to the system and run Python GC."""
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    """
    Apply the model's chat template with two fallback levels:
      1. Try with enable_thinking=False (Bonsai / Qwen3).
      2. If that raises TypeError, try without it.
      3. If either raises because system role is unsupported (Gemma),
         merge the system message into the first user message and retry.
    """
    kwargs = {"add_generation_prompt": True, "tokenize": False}

    def _apply(msgs):
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, **kwargs)

    try:
        return _apply(messages)
    except Exception:
        # Fallback: prepend system content to first user message
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            sys_text = msgs.pop(0)["content"]
            msgs[0] = {"role": "user",
                       "content": sys_text + "\n\n" + msgs[0]["content"]}
        return _apply(msgs)


def run_inference(model, tokenizer, sampler, prompt_text: str, max_tokens: int) -> dict:
    """Stream one completion and return metrics."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_text},
    ]
    chat_prompt = apply_chat_template(tokenizer, messages)

    start       = time.perf_counter()
    ttft        = None
    chunks: list[str] = []
    last        = None

    for response in stream_generate(
        model, tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if ttft is None:
            ttft = time.perf_counter() - start
        chunks.append(response.text)
        last = response

    elapsed = time.perf_counter() - start
    peak_gb = mx.get_peak_memory() / (1024 ** 3)

    return {
        "response":          "".join(chunks),
        "ttft":              ttft,
        "tps":               last.generation_tps if last else None,
        "prompt_tokens":     last.prompt_tokens if last else None,
        "generation_tokens": last.generation_tokens if last else None,
        "peak_memory_gb":    peak_gb,
        "total_time":        elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_models = "bonsai-8B,gemma-4-e2b,ministral-3b,qwen3-4b"
    parser.add_argument("--dataset",         default="mmlu-pro",
                        choices=list(DATASETS),
                        help="Benchmark dataset (default: mmlu-pro)")
    parser.add_argument("--models",          default=_default_models)
    parser.add_argument("--n",               type=int,   default=200)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--max-tokens",      type=int,   default=1024)
    parser.add_argument("--temp",            type=float, default=None,
                        help="Override temperature for all models "
                             "(default: use each model's recommended temp)")
    parser.add_argument("--output",          default="results/benchmark.jsonl")
    parser.add_argument("--questions-cache", default=None,
                        help="Cache file for sampled questions "
                             "(default: results/<dataset>_sample.json)")
    parser.add_argument("--demo-dir",        default=None,
                        help="Root of the Bonsai demo repo (default: parent of this script)")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir) if args.demo_dir else Path(__file__).parent.parent
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cache_file = (Path(args.questions_cache) if args.questions_cache
                  else Path(f"results/{args.dataset}_sample.json"))

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]

    # ── Load questions ─────────────────────────────────────────────────────
    questions = load_questions(args.n, args.seed, cache_file, args.dataset)
    print(f"Dataset: {args.dataset}  |  {len(questions)} questions  |  "
          f"models: {', '.join(model_keys)}\n")

    # ── Resume state (+ import from previous runs) ─────────────────────────
    done = load_done(output_file, questions, model_keys)
    if done:
        print(f"{DIM}{len(done)} question/model pair(s) already done — skipping those.{RESET}\n")

    run_start = time.perf_counter()
    model_timings: dict[str, dict] = {}   # model_key → {load, inference, n_questions}

    # ── Per-model loop ─────────────────────────────────────────────────────
    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"{RED}Unknown model key '{model_key}' — skipping.{RESET}")
            print(f"  Valid keys: {', '.join(MODELS)}")
            continue

        cfg = MODELS[model_key]
        model_path_raw = cfg["path"]
        temp = args.temp if args.temp is not None else cfg["temp"]

        # Resolve local paths relative to demo_dir
        if model_path_raw.startswith("models/"):
            local_path = demo_dir / model_path_raw
            if not local_path.exists():
                print(f"{RED}  Local model not found: {local_path}{RESET}")
                print(f"  Bonsai models require setup.sh to be run first:")
                print(f"    ./setup.sh")
                print(f"  Or download just the models:")
                print(f"    ./scripts/download_models.sh")
                continue
            model_path = str(local_path)
        else:
            model_path = model_path_raw  # HuggingFace repo ID — downloaded automatically

        remaining = [q for q in questions
                     if (model_key, str(q["question_id"])) not in done]

        if not remaining:
            print(f"[{model_key}] All {len(questions)} questions done — skipping.\n")
            continue

        print(f"{BOLD}{'─'*60}{RESET}")
        print(f"{BOLD}[{model_key}]{RESET}  {len(remaining)}/{len(questions)} questions to run")
        print(f"  path: {model_path}  temp: {temp}")

        t_load_start = time.perf_counter()
        try:
            import warnings
            import transformers
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transformers.logging.set_verbosity_error()
                model, tokenizer = load(model_path)
        except Exception as e:
            print(f"{RED}  Failed to load: {e}{RESET}\n")
            continue
        t_load_end = time.perf_counter()

        sampler = make_sampler(temp=temp)
        correct        = 0
        parse_failures = 0
        t_inference_start = time.perf_counter()

        try:
            with open(output_file, "a") as out_f:
                for i, q in enumerate(remaining):
                    qid = str(q["question_id"])
                    prompt_text = format_prompt(q["question"], q["options"])

                    # Per-question seed: stable across resumes since it's
                    # derived from the fixed question_id, not loop position.
                    mx.random.seed(
                        (args.seed * 100003 + _stable_int(qid)) % (2 ** 32)
                    )

                    try:
                        metrics = run_inference(model, tokenizer, sampler,
                                                prompt_text, args.max_tokens)
                    except KeyboardInterrupt:
                        print(f"\n{CYAN}Interrupted — progress saved to {output_file}{RESET}")
                        sys.exit(0)
                    except Exception as e:
                        print(f"  {RED}Error on q{qid}: {e}{RESET}")
                        continue

                    predicted  = extract_answer(metrics["response"])
                    parse_failed = predicted is None
                    is_correct   = (predicted == q["answer"]) if not parse_failed else False
                    if is_correct:
                        correct += 1
                    elif parse_failed:
                        parse_failures += 1

                    record = {
                        "model":             model_key,
                        "question_id":       qid,
                        "temp":              temp,
                        "seed":              args.seed,
                        "category":          q.get("category", ""),
                        "correct_answer":    q["answer"],
                        "predicted_answer":  predicted,
                        "is_correct":        is_correct,
                        "parse_failed":      parse_failed,
                        "response":          metrics["response"],
                        "ttft":              metrics["ttft"],
                        "tps":               metrics["tps"],
                        "prompt_tokens":     metrics["prompt_tokens"],
                        "generation_tokens": metrics["generation_tokens"],
                        "peak_memory_gb":    metrics["peak_memory_gb"],
                        "total_time":        metrics["total_time"],
                        "timestamp":         time.time(),
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    # Release KV-cache buffers from this question before the next
                    clear_mlx_cache()

                    done_count = i + 1
                    acc = correct / done_count * 100
                    if parse_failed:
                        mark = CYAN + "?" + RESET
                    elif is_correct:
                        mark = GREEN + "✓" + RESET
                    else:
                        mark = RED + "✗" + RESET
                    ttft_s = f"{metrics['ttft']:.2f}s" if metrics["ttft"] else "—"
                    tps_s  = f"{metrics['tps']:.1f}" if metrics["tps"] else "—"
                    mem_s  = f"{metrics['peak_memory_gb']:.2f}GB"
                    fail_s = f"  {CYAN}parse_fail={parse_failures}{RESET}" if parse_failures else ""
                    print(
                        f"  {mark} [{done_count:>3}/{len(remaining)}] "
                        f"acc={acc:5.1f}%  TTFT={ttft_s:>6}  "
                        f"TPS={tps_s:>6}  mem={mem_s}{fail_s}"
                    )

        except KeyboardInterrupt:
            print(f"\n{CYAN}Interrupted — progress saved to {output_file}{RESET}")
            sys.exit(0)

        total_done = sum(1 for q in questions
                         if (model_key, str(q["question_id"])) in done
                         or (model_key, str(q["question_id"])) in
                         {(model_key, str(qq["question_id"])) for qq in remaining})
        t_inference_end = time.perf_counter()

        n_done         = len(remaining)
        load_secs      = t_load_end      - t_load_start
        inference_secs = t_inference_end - t_inference_start
        per_q_secs     = inference_secs  / n_done if n_done else 0

        model_timings[model_key] = {
            "load_s":      load_secs,
            "inference_s": inference_secs,
            "per_q_s":     per_q_secs,
            "n_questions": n_done,
        }

        pf_s = f"  {parse_failures} parse failures" if parse_failures else ""
        print(f"\n  {BOLD}[{model_key}] finished — {correct}/{n_done} correct "
              f"({correct/n_done*100:.1f}%){pf_s}{RESET}")
        print(f"  load: {load_secs:.1f}s  |  inference: {inference_secs:.1f}s  "
              f"|  {per_q_secs:.1f}s/q\n")

        # Unload model fully before loading the next one
        del model, tokenizer, sampler
        clear_mlx_cache()

    total_secs = time.perf_counter() - run_start

    # ── Category breakdown ────────────────────────────────────────────────
    from collections import Counter
    cat_counts = Counter(q["category"] for q in questions)

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  RECAP{RESET}")
    print(f"{'═'*60}")
    print(f"  Total wall time : {total_secs/60:.1f} min  ({total_secs:.0f}s)")
    print(f"  Questions       : {len(questions)}")
    print(f"  Models run      : {len(model_timings)}")

    print(f"\n  {BOLD}Questions per category:{RESET}")
    for cat, cnt in sorted(cat_counts.items()):
        bar = "█" * cnt
        print(f"    {cat:<32} {cnt:>3}  {bar}")

    if model_timings:
        print(f"\n  {BOLD}Time per model:{RESET}")
        print(f"    {'Model':<25} {'Load':>7} {'Total':>8} {'Per Q':>7}")
        print(f"    {'─'*25} {'─'*7} {'─'*8} {'─'*7}")
        for mk, t in model_timings.items():
            print(f"    {mk:<25} {t['load_s']:>6.1f}s "
                  f"{t['inference_s']:>7.1f}s "
                  f"{t['per_q_s']:>6.1f}s")

    print(f"\n{GREEN}  Results → {output_file}{RESET}")
    print(f"  python scripts/benchmark_summary.py {output_file}")
    print(f"{'═'*60}\n")

    # ── Auto-generate plots ───────────────────────────────────────────────
    plot_script = Path(__file__).parent / "plot_benchmark.py"
    plots_dir   = output_file.parent / "plots"
    print(f"{BOLD}Generating plots → {plots_dir}/{RESET}")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(plot_script),
         str(output_file), "--output-dir", str(plots_dir)],
        check=False,
    )
    if result.returncode != 0:
        print(f"{RED}Plotting failed (exit {result.returncode}) — "
              f"run manually: python {plot_script} {output_file}{RESET}")


if __name__ == "__main__":
    main()
