"""Robustness checks behind the claims in the README (writes data/results/extra_checks.json).

  bootstrap       paired bootstrap over RAGTruth source questions: AUROC(faithscorer_sent) - AUROC(alignscore)
  lexical         training-free baseline: fraction of the answer's content words that occur in the context
                  (threshold tuned on constructed dev) -- how much of each test set surface overlap solves
  overlap         test questions whose gold paragraph also occurs in the training data, and the test
                  results with those questions removed
  sent_baselines  RAGTruth AUROC / balanced accuracy of baselines given the same sentence-min aggregation
                  (scores cached in data/scores/<split>/<name>_sent.json; needs the GPU, ~1 h with MiniCheck)

Usage:
  python src/eval/extra_checks.py                    # cheap checks only (uses cached scores)
  python src/eval/extra_checks.py --sent-baselines   # also score the sentence-min baselines
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_scores, tune_threshold  # noqa: E402
from run_scorers import load_split  # noqa: E402
from scorers import SENT_BASELINES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "results" / "extra_checks.json"
STOP = set("the a an of in on at to and or is was were be by for with as from that this which who it its "
           "his her their he she they has had have are".split())


def bootstrap_auroc_diff(labels, a, b, groups, n=2000, seed=0):
    """Resample whole source questions (all six generators' answers move together)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        if len(set(labels[idx])) < 2:
            continue
        diffs.append(roc_auc_score(labels[idx], a[idx]) - roc_auc_score(labels[idx], b[idx]))
    diffs = np.array(diffs)
    return {"diff": float(roc_auc_score(labels, a) - roc_auc_score(labels, b)),
            "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
            "p_diff_le_0": float((diffs <= 0).mean()), "resamples": len(diffs)}


def word_overlap(context, answer):
    ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if w not in STOP]
    return sum(w in ctx for w in words) / max(1, len(words))


def lexical_baseline():
    def scored(split):
        rows = load_split(split)
        return rows, np.array([r["label"] for r in rows]), np.array([word_overlap(r["context"], r["answer"]) for r in rows])

    _, dev_y, dev_s = scored("dev")
    thr = tune_threshold(dev_s, dev_y)
    out = {"threshold_from_dev": float(thr)}
    for split in ["test", "ragtruth_qa"]:
        rows, y, s = scored(split)
        pred = (s >= thr).astype(int)
        out[split] = {"auroc": float(roc_auc_score(y, s)), "balanced_acc": float(balanced_accuracy_score(y, pred))}
        if split == "test":
            cond = np.array([r["condition"] for r in rows])
            out[split]["per_condition"] = {c: float(np.mean(pred[cond == c] == y[cond == c])) for c in sorted(set(cond))}
    return out


def paragraph_overlap():
    base = {r["id"]: r for split in ["train", "test"]
            for r in json.loads((REPO_ROOT / "data" / "base" / f"{split}.json").read_text(encoding="utf-8"))}
    text = lambda p: " ".join(p["sentences"])  # noqa: E731
    train_q = {r["qid"] for r in load_split("train")}
    seen = {text(p) for q in train_q for p in base[q]["gold"] + base[q]["distractors"]}
    rows = load_split("test")
    overlap_q = {r["qid"] for r in rows if any(text(p) in seen for p in base[r["qid"]]["gold"])}
    keep = np.array([r["qid"] not in overlap_q for r in rows])
    y = np.array([r["label"] for r in rows])
    thresholds = json.loads((REPO_ROOT / "data" / "results" / "report.json").read_text())["thresholds_from_dev"]
    out = {"test_questions": len({r["qid"] for r in rows}), "questions_with_gold_paragraph_in_train": len(overlap_q),
           "results_without_them": {}}
    for name in ["nli", "nli_base", "alignscore", "minicheck", "faithscorer", "faithscorer_sent"]:
        s = load_scores("test", name)[0]
        out["results_without_them"][name] = {
            "balanced_acc": float(balanced_accuracy_score(y[keep], (s[keep] >= thresholds[name]).astype(int))),
            "auroc": float(roc_auc_score(y[keep], s[keep]))}
    return out


def sentence_baselines(score_missing):
    out = {}
    for name in SENT_BASELINES:
        if score_missing:
            from run_scorers import main as run_main
            for split in ["ragtruth_qa_dev", "ragtruth_qa"]:
                sys.argv = ["run_scorers.py", "--split", split, "--scorers", name]
                run_main()
        dev, test = load_scores("ragtruth_qa_dev", name), load_scores("ragtruth_qa", name)
        if dev is None or test is None:
            continue
        dev_y = np.array([r["label"] for r in load_split("ragtruth_qa_dev")])
        y = np.array([r["label"] for r in load_split("ragtruth_qa")])
        thr = tune_threshold(dev[0], dev_y)
        out[name] = {"ragtruth_dev_auroc": float(roc_auc_score(dev_y, dev[0])),
                     "ragtruth_test_auroc": float(roc_auc_score(y, test[0])),
                     "ragtruth_test_balanced_acc_ragtruth_dev_thr": float(
                         balanced_accuracy_score(y, (test[0] >= thr).astype(int)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent-baselines", action="store_true", help="score the sentence-min baselines (GPU)")
    args = ap.parse_args()
    t0 = time.perf_counter()

    rt = load_split("ragtruth_qa")
    y = np.array([r["label"] for r in rt])
    report = {
        "bootstrap_faithscorer_sent_vs_alignscore": bootstrap_auroc_diff(
            y, load_scores("ragtruth_qa", "faithscorer_sent")[0], load_scores("ragtruth_qa", "alignscore")[0],
            np.array([r["qid"] for r in rt])),
        "lexical_overlap_baseline": lexical_baseline(),
        "train_test_paragraph_overlap": paragraph_overlap(),
        "sentence_min_baselines": sentence_baselines(args.sent_baselines),
    }
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"-> {OUT} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
