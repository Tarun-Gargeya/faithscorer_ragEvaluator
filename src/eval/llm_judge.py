"""LLM-as-judge baseline over any OpenAI-compatible endpoint (local Ollama, Groq, Cerebras, ...).

The judge sees only (question, context, answer) -- never the gold answer -- and returns a binary
faithful/unfaithful verdict. It is a *baseline* to compare against, not the source of labels.

Judges are configured in JUDGES below; API keys come from .env (GROQ_API_KEY).
Sampling is stratified by condition so small free-tier budgets still cover every condition.
Resumable: results are appended to data/judge/<split>/<judge>.json and existing ids are skipped.

Usage:
  python src/eval/llm_judge.py --split test --judge qwen7b --per-condition 100
  python src/eval/llm_judge.py --split ragtruth_qa --judge gptoss120b --limit 300
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scorers import load_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

JUDGES = {
    # name: (base_url, api_key_env or None, model, min seconds between calls, extra request kwargs)
    "qwen7b": ("http://localhost:11434/v1", None, "qwen2.5:7b-instruct", 0.0, {"max_tokens": 5}),
    # Groq free tier: 1000 requests/day, 8000 tokens/min -> ~8s between ~700-token calls (6s still tripped the per-minute cap).
    # gpt-oss-120b is a reasoning model: low effort, and room for its hidden reasoning tokens.
    "gptoss120b": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", 8.0,
                   {"max_tokens": 512, "reasoning_effort": "low"}),
    # Smaller GPT-class judge; separate per-model daily quota on the same Groq account.
    "gptoss20b": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-20b", 8.0,
                  {"max_tokens": 512, "reasoning_effort": "low"}),
}

PROMPT = """You are checking whether an answer is FAITHFUL to the retrieved context.

An answer is faithful only if every claim in it is supported by the context. It is UNFAITHFUL if any \
part of it contradicts the context, or adds information that the context does not state -- even if \
that information is true in the real world. Judge only against the context, not your own knowledge.

Context:
{context}

Question: {question}

Answer: {answer}

Reply with exactly one word: faithful or unfaithful."""


def parse(text):
    t = text.strip().lower()
    if re.search(r"\bunfaithful\b", t):
        return 0
    if re.search(r"\bfaithful\b", t):
        return 1
    return None


def retry_after(message):
    """Parse Groq's 'Please try again in 1h2m3.4s' (any part optional); default 60s."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", message)
    if not m or not any(m.groups()):
        return 60.0
    h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
    return 3600 * h + 60 * mi + sec


def stratified_sample(rows, per_condition, limit, seed):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    if per_condition:
        by_cond, picked = defaultdict(list), []
        for i in idx:
            if len(by_cond[rows[i]["condition"]]) < per_condition:
                by_cond[rows[i]["condition"]].append(i)
        picked = sorted(i for v in by_cond.values() for i in v)
        return picked
    return sorted(idx[:limit]) if limit else list(range(len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--judge", choices=JUDGES, required=True)
    ap.add_argument("--per-condition", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="random sample of this size")
    ap.add_argument("--first", type=int, default=None, help="judge the first N rows in file order")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_url, key_env, model, pause, extra = JUDGES[args.judge]
    api_key = os.environ.get(key_env) if key_env else "ollama"
    if not api_key:
        sys.exit(f"Set {key_env} in .env to use judge '{args.judge}'.")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)

    rows = load_split(args.split)
    if args.first:
        indices = list(range(min(args.first, len(rows))))
    else:
        indices = stratified_sample(rows, args.per_condition, args.limit, args.seed)
    out_path = REPO_ROOT / "data" / "judge" / args.split / f"{args.judge}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["index"] for r in results}
    todo = [i for i in indices if i not in done]
    print(f"{args.judge} on {args.split}: {len(done)} done, {len(todo)} to go")

    for n, i in enumerate(todo, 1):
        r = rows[i]
        prompt = PROMPT.format(context=r["context"], question=r["question"], answer=r["answer"])
        failures = 0
        while True:
            try:
                t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], temperature=0, **extra)
                secs = time.perf_counter() - t0
                break
            except RateLimitError as e:
                # Groq's free tier caps tokens per minute AND per day (200k/day for gpt-oss-120b); the daily
                # budget refills continuously, so wait exactly as long as the server says and carry on.
                wait = retry_after(str(e)) + 3
                print(f"  rate-limited on {i}; waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
            except Exception as e:  # transient errors: back off, give up after 5
                failures += 1
                if failures >= 5:
                    break
                print(f"  error on {i} ({type(e).__name__}: {str(e)[:400]}); retrying", flush=True)
                time.sleep(15 * 2 ** failures)
        if failures >= 5:
            print("  giving up for now; re-run later to resume")
            break
        text = resp.choices[0].message.content or ""
        results.append({"index": i, "pred": parse(text), "raw": text, "seconds": secs,
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None)})
        if n % 5 == 0 or n == len(todo):
            out_path.write_text(json.dumps(results))
            print(f"  [{n}/{len(todo)}]", flush=True)
        time.sleep(pause)
    out_path.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
