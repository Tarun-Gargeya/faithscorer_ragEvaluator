"""Stage C: assemble (context, answer, label) examples whose label is known by construction.

Faithfulness is a property of the (context, answer) pair only -- it never looks at whether
the answer is factually correct. Conditions per question:

  label 1 (faithful)
    clean               gold paragraphs                       + grounded answer
    noisy               gold + hard-negative distractors      + grounded answer
    cf_followed         gold with answer swapped to a wrong one + answer using the wrong value
                        (faithful to context even though factually wrong)
    supported_extra     gold                                  + grounded answer + supported sentence

  label 0 (unfaithful)
    cf_contradicted     gold with answer swapped              + original grounded answer  (contradiction)
    incomplete          gold minus the answer-bearing sentence + grounded answer           (unsupported)
    irrelevant          related-but-non-answering distractors only + grounded answer       (parametric)
    unsupported_extra   gold                                  + grounded answer + invented sentence (addition)

Distractors are added to every non-'irrelevant' condition with p=0.5 (always for 'noisy'),
so "distractors present" can't be used as a shortcut. Paragraph order is shuffled.

Rows that failed the independent NLI check in verify.py are dropped (whole question if the
grounded answer isn't entailed; just the affected *_extra condition otherwise).
"""
import argparse
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "data" / "base"
AUG_DIR = REPO_ROOT / "data" / "augmented"
OUT_DIR = REPO_ROOT / "data" / "constructed"

VERIFY_THRESHOLD = 0.5  # same as verify.py
FAITHFUL = {"clean", "noisy", "cf_followed", "supported_extra"}
CONDITIONS = ["clean", "noisy", "cf_followed", "supported_extra",
              "cf_contradicted", "incomplete", "irrelevant", "unsupported_extra"]


def replace_ci(text, old, new):
    return re.sub(re.escape(old), lambda _: new, text, flags=re.I)


def render(paragraphs):
    return "\n\n".join(" ".join(p["sentences"]) for p in paragraphs)


def build(base, aug, rng):
    ans_title, ans_idx = base["answer_sent"]
    gold = base["gold"]
    distractors = base["distractors"]
    grounded = aug["grounded_answer"]

    def with_noise(paras, force=False):
        paras = list(paras)
        if force or rng.random() < 0.5:
            paras += rng.sample(distractors, k=min(len(distractors), rng.randint(2, 3)))
        rng.shuffle(paras)
        return paras

    def edit_answer_sentence(fn):
        out = []
        for p in gold:
            sents = list(p["sentences"])
            if p["title"] == ans_title:
                sents = fn(sents)
            out.append({"title": p["title"], "sentences": sents})
        return out

    def swap(sents):
        sents[ans_idx] = replace_ci(sents[ans_idx], base["answer"], aug["wrong_answer"])
        return sents

    cf_gold = edit_answer_sentence(swap)
    incomplete_gold = [p for p in edit_answer_sentence(lambda s: s[:ans_idx] + s[ans_idx + 1:]) if p["sentences"]]
    cf_answer = replace_ci(grounded, base["answer"], aug["wrong_answer"])

    variants = {
        "clean": (with_noise(gold), grounded),
        "noisy": (with_noise(gold, force=True), grounded),
        "cf_followed": (with_noise(cf_gold), cf_answer),
        "supported_extra": (with_noise(gold), f"{grounded} {aug['supported_extra']}"),
        "cf_contradicted": (with_noise(cf_gold), grounded),
        "incomplete": (with_noise(incomplete_gold), grounded),
        "irrelevant": (rng.sample(distractors, k=min(len(distractors), 3)), grounded),
        "unsupported_extra": (with_noise(gold), f"{grounded} {aug['unsupported_extra']}"),
    }
    if aug["nli_unsupported"] >= VERIFY_THRESHOLD:
        del variants["unsupported_extra"]
    if aug["nli_supported"] < VERIFY_THRESHOLD:
        del variants["supported_extra"]
    rows = []
    for cond in CONDITIONS:
        if cond not in variants:
            continue
        paras, answer = variants[cond]
        context = render(paras)
        # Sanity: removing/swapping the answer sentence must actually remove the gold answer.
        if cond in {"cf_contradicted", "cf_followed", "incomplete", "irrelevant"} and \
                base["answer"].lower() in context.lower():
            continue
        rows.append({
            "qid": base["id"], "condition": cond, "label": int(cond in FAITHFUL),
            "question": base["question"], "context": context, "answer": answer,
            "gold_answer": base["answer"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        # Independent RNG per split, so building one split never changes another
        # (test keeps the plain seed it was first built and scored with).
        rng = random.Random(args.seed if split == "test" else args.seed + 1)
        base = {r["id"]: r for r in json.loads((BASE_DIR / f"{split}.json").read_text(encoding="utf-8"))}
        aug = [r for r in json.loads((AUG_DIR / f"{split}.json").read_text(encoding="utf-8")) if "failed" not in r]
        if any("nli_grounded" not in r for r in aug):
            raise SystemExit(f"Run src/data/verify.py --split {split} first.")
        aug = [r for r in aug if r["nli_grounded"] >= VERIFY_THRESHOLD]
        rows = [row for a in aug for row in build(base[a["id"]], a, rng)]

        if split == "train":
            # Dev split by question so no question leaks between train and dev.
            qids = sorted({r["qid"] for r in rows})
            rng.shuffle(qids)
            dev_q = set(qids[: int(len(qids) * args.dev_frac)])
            parts = {"train": [r for r in rows if r["qid"] not in dev_q],
                     "dev": [r for r in rows if r["qid"] in dev_q]}
        else:
            parts = {"test": rows}

        for name, part in parts.items():
            (OUT_DIR / f"{name}.json").write_text(json.dumps(part, indent=1), encoding="utf-8")
            pos = sum(r["label"] for r in part)
            print(f"{name}: {len(part)} examples from {len({r['qid'] for r in part})} questions "
                  f"({pos} faithful / {len(part) - pos} unfaithful)")


if __name__ == "__main__":
    main()
