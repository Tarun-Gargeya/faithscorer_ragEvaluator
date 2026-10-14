"""Build the results report from cached scores (data/scores/) and judge verdicts (data/judge/).

For every scorer:
  - threshold tuned on the constructed *dev* split (maximising balanced accuracy), then frozen
    and applied unchanged to the test split and to RAGTruth (no OOD tuning);
  - overall balanced accuracy, F1 on the unfaithful class, AUROC, ms/example;
  - per-condition accuracy (fraction of that condition classified correctly).
LLM judges are compared on exactly the subset they judged, alongside every scorer on that subset.
Escalation curve: FaithScorer keeps confident cases, the least-confident k% go to the judge.

Writes data/results/report.json, data/results/report.md and plots in data/results/.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scorers import load_split  # noqa: E402
from scorers import SCORERS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = REPO_ROOT / "data" / "scores"
JUDGE_DIR = REPO_ROOT / "data" / "judge"
OUT_DIR = REPO_ROOT / "data" / "results"
EVAL_SPLITS = ["test", "ragtruth_qa"]
OURS = "faithscorer_sent"  # headline: sentence-level min, chosen on RAGTruth *dev* (README §6)
OOD_DEV = "ragtruth_qa_dev"  # human-labelled sample of RAGTruth's train split, OOD calibration only


def load_scores(split, name):
    p = SCORE_DIR / split / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return np.array(d["scores"]), 1000 * d["seconds"] / d["n"]


def tune_threshold(scores, labels):
    cands = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197)))
    accs = [balanced_accuracy_score(labels, (scores >= t).astype(int)) for t in cands]
    return float(cands[int(np.argmax(accs))])


def metrics(labels, preds, probs=None):
    m = {"n": int(len(labels)),
         "balanced_acc": float(balanced_accuracy_score(labels, preds)),
         "f1_unfaithful": float(f1_score(labels, preds, pos_label=0, zero_division=0))}
    if probs is not None and len(set(labels)) > 1:
        m["auroc"] = float(roc_auc_score(labels, probs))
    return m


def per_condition(rows, preds):
    out = {}
    for c in sorted({r["condition"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["condition"] == c]
        out[c] = {"n": len(idx), "label": rows[idx[0]]["label"],
                  "accuracy": float(np.mean([preds[i] == rows[i]["label"] for i in idx]))}
    return out


def load_judges(split):
    out = {}
    d = JUDGE_DIR / split
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        res = [r for r in json.loads(p.read_text()) if r["pred"] is not None]
        if res:
            out[p.stem] = res
    return out


def logit(p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def escalation_curve(labels, probs, thr, judge_idx, judge_pred):
    """Within the judged subset: send the k% least-confident FaithScorer cases to the judge."""
    labels, probs = labels[judge_idx], probs[judge_idx]
    own = (probs >= thr).astype(int)
    # Least confident first, measured in log-odds: with a threshold near 0 or 1, raw |p - thr| would
    # rank the most confidently-rejected answers as "least confident".
    order = np.argsort(np.abs(logit(probs) - logit(thr)))
    curve = []
    for frac in np.linspace(0, 1, 21):
        k = int(round(frac * len(order)))
        pred = own.copy()
        pred[order[:k]] = judge_pred[order[:k]]
        curve.append({"escalated": float(frac), "balanced_acc": float(balanced_accuracy_score(labels, pred))})
    return curve


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_labels = np.array([r["label"] for r in load_split("dev")])
    thresholds = {}
    for name in SCORERS:
        s = load_scores("dev", name)
        if s is not None:
            thresholds[name] = tune_threshold(s[0], dev_labels)

    # Second, deployment-style calibration for RAGTruth: thresholds tuned on RAGTruth's train split.
    ood_thresholds = {}
    try:
        ood_labels = np.array([r["label"] for r in load_split(OOD_DEV)])
        for name in SCORERS:
            s = load_scores(OOD_DEV, name)
            if s is not None:
                ood_thresholds[name] = tune_threshold(s[0], ood_labels)
    except FileNotFoundError:
        pass

    report = {"thresholds_from_dev": thresholds, "thresholds_from_ragtruth_dev": ood_thresholds, "splits": {}}
    md = ["# Results\n", "Thresholds are tuned on the constructed dev split and frozen for all test sets. "
          "For RAGTruth, a second balanced accuracy uses thresholds tuned on RAGTruth's train split "
          "(`ragtruth_qa_dev`, disjoint questions), for every scorer alike.\n"]

    for split in EVAL_SPLITS:
        try:
            rows = load_split(split)
        except FileNotFoundError:
            continue
        labels = np.array([r["label"] for r in rows])
        sp = {"scorers": {}, "judges": {}}
        ood = split.startswith("ragtruth") and bool(ood_thresholds)
        md += [f"\n## {split}  (n={len(rows)}, {int(labels.sum())} faithful / {int((1 - labels).sum())} unfaithful)\n",
               "| System | Bal. acc | F1 (unfaithful) | AUROC | ms / example |" + (" Bal. acc (RAGTruth-dev thr.) |" if ood else ""),
               "|---|---|---|---|---|" + ("---|" if ood else "")]
        all_probs, all_preds = {}, {}
        for name in SCORERS:
            s = load_scores(split, name)
            if s is None or name not in thresholds:
                continue
            probs, ms = s
            preds = (probs >= thresholds[name]).astype(int)
            all_probs[name], all_preds[name] = probs, preds
            m = metrics(labels, preds, probs)
            m.update(ms_per_example=ms, per_condition=per_condition(rows, preds))
            extra = ""
            if ood and name in ood_thresholds:
                m["balanced_acc_ood_threshold"] = float(balanced_accuracy_score(
                    labels, (probs >= ood_thresholds[name]).astype(int)))
                extra = f" {m['balanced_acc_ood_threshold']:.3f} |"
            sp["scorers"][name] = m
            bold = "**" if name == OURS else ""
            md.append(f"| {bold}{name}{bold} | {m['balanced_acc']:.3f} | {m['f1_unfaithful']:.3f} | "
                      f"{m.get('auroc', float('nan')):.3f} | {ms:.1f} |" + extra)

        # Per-condition table
        if sp["scorers"]:
            conds = sorted({r["condition"] for r in rows})
            names = list(sp["scorers"])
            md += ["", "Per-condition accuracy (label: 1 = faithful, 0 = unfaithful):", "",
                   "| Condition | label | n | " + " | ".join(names) + " |",
                   "|---|---|---|" + "---|" * len(names)]
            for c in conds:
                pc = [sp["scorers"][n]["per_condition"][c] for n in names]
                md.append(f"| {c} | {pc[0]['label']} | {pc[0]['n']} | " +
                          " | ".join(f"{p['accuracy']:.2f}" for p in pc) + " |")

        # Judges, compared on their own subset
        for jname, res in load_judges(split).items():
            idx = np.array([r["index"] for r in res])
            jpred = np.array([r["pred"] for r in res])
            jm = metrics(labels[idx], jpred)
            jm["ms_per_example"] = 1000 * float(np.mean([r["seconds"] for r in res]))
            jm["per_condition"] = per_condition([rows[i] for i in idx], jpred)
            jm["scorers_on_same_subset"] = {
                n: {**metrics(labels[idx], all_preds[n][idx], all_probs[n][idx]),
                    "kappa_with_judge": float(cohen_kappa_score(jpred, all_preds[n][idx]))}
                for n in all_preds}
            if OURS in all_probs:
                thr = ood_thresholds.get(OURS, thresholds[OURS]) if ood else thresholds[OURS]
                jm["escalation_curve"] = escalation_curve(labels, all_probs[OURS], thr, idx, jpred)
            sp["judges"][jname] = jm
            md += ["", f"### LLM judge `{jname}` on its {len(idx)}-example subset", "",
                   "| System | Bal. acc | F1 (unfaithful) | κ with judge |", "|---|---|---|---|",
                   f"| {jname} (judge) | {jm['balanced_acc']:.3f} | {jm['f1_unfaithful']:.3f} | — |"]
            for n, m in jm["scorers_on_same_subset"].items():
                md.append(f"| {n} | {m['balanced_acc']:.3f} | {m['f1_unfaithful']:.3f} | {m['kappa_with_judge']:.3f} |")
            if "escalation_curve" in jm:
                cur = jm["escalation_curve"]
                md.append("\nEscalation (FaithScorer + judge on least-confident cases): " + ", ".join(
                    f"{int(100 * c['escalated'])}% → {c['balanced_acc']:.3f}" for c in cur[::4]))
                plt.figure(figsize=(5, 3.5))
                plt.plot([100 * c["escalated"] for c in cur], [c["balanced_acc"] for c in cur], marker="o", ms=3)
                plt.axhline(jm["balanced_acc"], ls="--", c="gray", label=f"{jname} alone")
                plt.xlabel("% of cases escalated to LLM judge")
                plt.ylabel("Balanced accuracy")
                plt.title(f"Escalation curve — {split}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(OUT_DIR / f"escalation_{split}_{jname}.png", dpi=150)
                plt.close()
        report["splits"][split] = sp

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1))
    (OUT_DIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
