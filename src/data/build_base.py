"""Stage A: sample HotpotQA (distractor setting) questions into a compact base file.

Train questions come from HotpotQA train, test questions from HotpotQA validation,
so no question (or its gold paragraphs) is shared between our train and test sets.

Keeps only questions whose answer is a span (not yes/no) that appears verbatim in
exactly one supporting sentence -- that sentence is the one we later edit/remove.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "base"


def clean(s: str) -> str:
    return s.strip()


def convert(ex):
    answer = ex["answer"].strip()
    if answer.lower() in {"yes", "no"} or len(answer) < 2:
        return None

    titles = ex["context"]["title"]
    paras = {t: [clean(s) for s in sents] for t, sents in zip(titles, ex["context"]["sentences"])}
    sf = list(zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]))
    gold_titles = list(dict.fromkeys(t for t, _ in sf))
    if len(gold_titles) != 2 or any(t not in paras for t in gold_titles):
        return None
    if any(i >= len(paras[t]) for t, i in sf):
        return None

    answer_sents = [(t, i) for t, i in sf if answer.lower() in paras[t][i].lower()]
    if len(answer_sents) != 1:
        return None
    # The answer must not also appear in non-supporting gold sentences, otherwise
    # deleting the answer sentence wouldn't make the answer unsupported.
    n_mentions = sum(s.lower().count(answer.lower()) for t in gold_titles for s in paras[t])
    if n_mentions != 1:
        return None

    distractors = [{"title": t, "sentences": paras[t]} for t in titles if t not in gold_titles]
    if len(distractors) < 3:
        return None

    return {
        "id": ex["id"],
        "question": ex["question"].strip(),
        "answer": answer,
        "type": ex["type"],
        "gold": [{"title": t, "sentences": paras[t]} for t in gold_titles],
        "supporting": [[t, i] for t, i in sf],
        "answer_sent": list(answer_sents[0]),
        "distractors": distractors,
    }


def sample(split, n, seed):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        row = convert(ds[i])
        if row:
            out.append(row)
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split, n in [("train", "train", args.n_train), ("test", "validation", args.n_test)]:
        rows = sample(split, n, args.seed)
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"{name}: {len(rows)} questions -> {path}")


if __name__ == "__main__":
    main()
