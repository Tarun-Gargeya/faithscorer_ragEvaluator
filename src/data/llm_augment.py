"""Stage B: use a local LLM (Ollama) to write the natural-language pieces we need.

Per question the LLM returns:
  grounded_answer    1-2 sentences, only from the supporting facts, containing the gold answer
  wrong_answer       a plausible same-type replacement for the gold answer (for counterfactual edits)
  supported_extra    one extra sentence that IS stated in the gold paragraphs
  unsupported_extra  one plausible, specific sentence NOT stated anywhere in the gold paragraphs

Every output is validated in code (see validate()); invalid rows are retried once, then dropped.
The LLM never decides a label -- labels are assigned by construction in construct.py.

Resumable: re-running skips ids already present in the output file.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "data" / "base"
OUT_DIR = REPO_ROOT / "data" / "augmented"
MODEL = "qwen2.5:7b-instruct"
# num_predict caps runaway JSON-mode generations; the timeout stops one stuck request from stalling the run.
CLIENT = ollama.Client(timeout=90)

PROMPT = """You are helping build a dataset. Read the paragraphs and answer in JSON.

Paragraphs:
{paragraphs}

Supporting facts (the sentences needed to answer):
{supporting}

Question: {question}
Gold answer: {answer}

Return a JSON object with exactly these keys:
- "grounded_answer": 1-2 sentences answering the question using ONLY the supporting facts. It MUST contain the exact text "{answer}" and should also state the linking fact from the other supporting sentence. Write it as a plain answer (max 40 words): never mention "the paragraphs", "the supporting facts", "the context", or quote them.
- "wrong_answer": a plausible but WRONG replacement for "{answer}" of the same type (a person for a person, a year for a year, a place for a place, etc.). It must not appear anywhere in the paragraphs.
- "supported_extra": one short sentence stating a different fact that IS explicitly written in the paragraphs but is not already in grounded_answer.
- "unsupported_extra": one short, specific, plausible-sounding sentence about the same people/things that is NOT stated anywhere in the paragraphs (e.g. an invented date, award, location or number).
"""


META_RE = re.compile(r"supporting (fact|sentence)|the (paragraph|context|passage)|explicitly (stated|mentioned)", re.I)


def fmt_paragraphs(row):
    return "\n\n".join(f"[{p['title']}] " + " ".join(p["sentences"]) for p in row["gold"])


def fmt_supporting(row):
    paras = {p["title"]: p["sentences"] for p in row["gold"]}
    return "\n".join(f"- {paras[t][i]}" for t, i in row["supporting"])


def contains(haystack, needle):
    return needle.lower() in haystack.lower()


def validate(row, out):
    keys = ["grounded_answer", "wrong_answer", "supported_extra", "unsupported_extra"]
    if not all(isinstance(out.get(k), str) and out[k].strip() for k in keys):
        return "missing key"
    out = {k: out[k].strip() for k in keys}
    gold_text = " ".join(s for p in row["gold"] for s in p["sentences"])
    if not contains(out["grounded_answer"], row["answer"]):
        return "grounded_answer lacks gold answer"
    if len(out["grounded_answer"].split()) > 50:
        return "grounded_answer too long"
    for k in ("grounded_answer", "supported_extra", "unsupported_extra"):
        if META_RE.search(out[k]):
            return f"{k} has meta-commentary"
    wa = out["wrong_answer"]
    if contains(wa, row["answer"]) or contains(row["answer"], wa) or contains(gold_text, wa):
        return "wrong_answer overlaps gold/context"
    if contains(out["supported_extra"], row["answer"]) and contains(out["grounded_answer"], out["supported_extra"]):
        return "supported_extra duplicates answer"
    if out["supported_extra"] == out["unsupported_extra"]:
        return "extras identical"
    return out


def augment(row):
    prompt = PROMPT.format(paragraphs=fmt_paragraphs(row), supporting=fmt_supporting(row),
                           question=row["question"], answer=row["answer"])
    last_err = None
    for attempt in range(2):
        try:
            resp = CLIENT.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                               format="json", options={"temperature": 0.3 + 0.4 * attempt, "seed": attempt,
                                                                  "num_predict": 400})
            result = validate(row, json.loads(resp["message"]["content"]))
        except Exception as e:  # malformed JSON, server hiccup
            result = f"error: {e}"
        if isinstance(result, dict):
            return {"id": row["id"], **result}
        last_err = result
    return {"id": row["id"], "failed": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = json.loads((BASE_DIR / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.split}.json"
    done = {}
    if out_path.exists():
        done = {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go")

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(augment, r) for r in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            done[res["id"]] = res
            if n % 50 == 0 or n == len(todo):
                out_path.write_text(json.dumps(list(done.values()), indent=1), encoding="utf-8")
                n_ok = sum("failed" not in r for r in done.values())
                print(f"[{n}/{len(todo)}] ok so far: {n_ok}/{len(done)}", flush=True)

    out_path.write_text(json.dumps(list(done.values()), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
