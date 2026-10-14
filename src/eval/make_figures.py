"""Report figures + extra statistics (data/results/figures/*.png, data/results/extra_stats.json).

Figures (static, for the printed report):
  fig_noise_effect.png        mean score on the same faithful answers: clean vs with distractors
  fig_condition_heatmap.png   per-condition accuracy, scorers x conditions (constructed test)
  fig_ragtruth_roc.png        ROC curves on RAGTruth test (human labels)
  fig_cost_accuracy.png       latency vs RAGTruth balanced accuracy (log-x)
  fig_escalation.png          FaithScorer + LLM judge escalation curve (RAGTruth)
  fig_training_curve.png      dev balanced accuracy / AUROC during training
Colours: fixed-order categorical slots from the dataviz reference palette, always paired with a
marker shape or direct label so identity is never colour-alone.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scorers import load_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = REPO_ROOT / "data" / "results"
FIG = RES / "figures"

# name -> (display label, colour slot, marker)
SYSTEMS = {
    "nli": ("NLI (DeBERTa-v3-base)", "#2a78d6", "o"),
    "alignscore": ("AlignScore-base", "#eb6834", "s"),
    "minicheck": ("MiniCheck (Flan-T5-L)", "#1baf7a", "^"),
    "faithscorer": ("FaithScorer, whole answer", "#eda100", "D"),
    "faithscorer_sent": ("FaithScorer, sentence-min", "#e87ba4", "P"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
CONDS = ["clean", "noisy", "cf_followed", "supported_extra",
         "cf_contradicted", "incomplete", "irrelevant", "unsupported_extra"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def scores(split, name):
    p = REPO_ROOT / "data" / "scores" / split / f"{name}.json"
    return json.loads(p.read_text())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    report = json.loads((RES / "report.json").read_text())
    stats = {}

    # ---- noise effect: same faithful answers, clean vs distractors --------------------------------
    test = load_split("test")
    cond = np.array([r["condition"] for r in test])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = [n for n in SYSTEMS if n != "faithscorer"]
    stats["noise_effect_mean_score"] = {}
    for k, n in enumerate(names):
        s = np.array(scores("test", n)["scores"])
        a, b = s[cond == "clean"].mean(), s[cond == "noisy"].mean()
        stats["noise_effect_mean_score"][n] = {"clean": float(a), "noisy": float(b)}
        label, col, mk = SYSTEMS[n]
        ax.plot([0, 1], [a, b], color=col, lw=2, marker=mk, ms=9, mec="white", mew=1.2)
        ax.annotate(f"{label}  {a:.2f} → {b:.2f}", (1, b), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_xticks([0, 1], ["clean context", "context + distractors"])
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean P(faithful) on faithful answers")
    ax.set_title("Distractor passages lower baseline scores on answers that are still faithful", loc="left")
    fig.savefig(FIG / "fig_noise_effect.png")
    plt.close(fig)

    # ---- per-condition heatmap ----------------------------------------------------------------------
    sp = report["splits"]["test"]["scorers"]
    hm_names = ["nli", "nli_base", "alignscore", "minicheck", "faithscorer", "faithscorer_sent"]
    hm_labels = ["NLI", "NLI-base\n(zero-shot)", "AlignScore", "MiniCheck", "FaithScorer\n(whole)", "FaithScorer\n(sent-min)"]
    M = np.array([[sp[n]["per_condition"][c]["accuracy"] for n in hm_names] for c in CONDS])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.grid(False)
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]), vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.8 else INK)
    ax.set_xticks(range(len(hm_names)), hm_labels, fontsize=8.5)
    ax.set_yticks(range(len(CONDS)), [f"{c}  ({'F' if c in CONDS[:4] else 'U'})" for c in CONDS])
    ax.set_title("Accuracy per retrieval condition (constructed test; F = faithful, U = unfaithful)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("accuracy")
    fig.savefig(FIG / "fig_condition_heatmap.png")
    plt.close(fig)

    # ---- RAGTruth ROC -------------------------------------------------------------------------------
    rt = load_split("ragtruth_qa")
    y = np.array([r["label"] for r in rt])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    for n, (label, col, mk) in SYSTEMS.items():
        s = np.array(scores("ragtruth_qa", n)["scores"])
        # positive class for ROC = hallucinated, so use 1 - P(faithful)
        fpr, tpr, _ = roc_curve(1 - y, 1 - s)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{label} (AUROC {roc_auc_score(y, s):.3f})",
                marker=mk, markevery=0.12, ms=6, mec="white")
    ax.set_xlabel("False-alarm rate (faithful answers flagged)")
    ax.set_ylabel("Hallucinations caught")
    ax.set_title("RAGTruth QA test (human labels)", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(FIG / "fig_ragtruth_roc.png")
    plt.close(fig)

    # ---- cost vs accuracy ---------------------------------------------------------------------------
    rsp = report["splits"]["ragtruth_qa"]
    dev_ms = {n: 1000 * scores("dev", n)["seconds"] / scores("dev", n)["n"] for n in SYSTEMS}
    dev_ms["faithscorer_sent"] = 1000 * scores("ragtruth_qa", "faithscorer_sent")["seconds"] / 900
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pts = []
    for n, (label, col, mk) in SYSTEMS.items():
        acc = rsp["scorers"][n]["balanced_acc_ood_threshold"]
        pts.append((label, dev_ms[n], acc))
        ax.scatter(dev_ms[n], acc, color=col, marker=mk, s=110, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(label, (dev_ms[n], acc), xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=INK)
    judges = {}
    for jn, jlabel in [("gptoss120b", "gpt-oss-120b judge"), ("gptoss20b", "gpt-oss-20b judge")]:
        if jn in rsp["judges"]:
            j = rsp["judges"][jn]
            judges[jn] = j
            ax.scatter(j["ms_per_example"], j["balanced_acc"], color=INK, marker="X", s=110, zorder=3)
            ax.annotate(f"{jlabel}\n(n={j['n']} subset)", (j["ms_per_example"], j["balanced_acc"]),
                        xytext=(-10, 8), textcoords="offset points", fontsize=8.5, color=INK, ha="right")
            pts.append((jlabel, j["ms_per_example"], j["balanced_acc"]))
    ax.set_xscale("log")
    ax.set_xlabel("Latency per answer (ms, log scale; judges = API round-trip only)")
    ax.set_ylabel("Balanced accuracy on RAGTruth")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Cost vs accuracy on real hallucinations", loc="left")
    fig.savefig(FIG / "fig_cost_accuracy.png")
    plt.close(fig)
    stats["cost_accuracy_points"] = [{"system": a, "ms": b, "balanced_acc": c} for a, b, c in pts]

    # ---- escalation ---------------------------------------------------------------------------------
    if "gptoss120b" in rsp["judges"] and "escalation_curve" in rsp["judges"]["gptoss120b"]:
        j = rsp["judges"]["gptoss120b"]
        cur = j["escalation_curve"]
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        ax.plot([100 * c["escalated"] for c in cur], [c["balanced_acc"] for c in cur], color="#e87ba4",
                lw=2, marker="P", ms=8, mec="white", label="FaithScorer + judge")
        ax.axhline(j["balanced_acc"], ls="--", color=MUTED, lw=1, label=f"gpt-oss-120b alone ({j['balanced_acc']:.3f})")
        ax.legend(loc="lower right")
        ax.set_xlabel("% of answers escalated to gpt-oss-120b (least-confident first)")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(f"Escalation on RAGTruth (judged subset, n={j['n']})", loc="left")
        fig.savefig(FIG / "fig_escalation.png")
        plt.close(fig)

    # ---- training curve -----------------------------------------------------------------------------
    log = (REPO_ROOT / "data" / "train_faithscorer.log").read_text(encoding="utf-8", errors="ignore")
    evals = [(float(e), float(b), float(a)) for b, a, e in re.findall(
        r"'eval_balanced_acc': '([\d.]+)'.*?'eval_auroc': '([\d.]+)'.*?'epoch': '([\d.]+)'", log)]
    if evals:
        ep, ba, au = map(np.array, zip(*evals))
        # epoch 0 = the zero-shot starting model (NLI-base P(entailment)) on the full dev split
        dev_y = [r["label"] for r in load_split("dev")]
        au0 = roc_auc_score(dev_y, scores("dev", "nli_base")["scores"])
        ep, ba, au = np.r_[0, ep], np.r_[np.nan, ba], np.r_[au0, au]
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(ep, au, color="#2a78d6", lw=2, marker="o", ms=6, mec="white", label="dev AUROC")
        ax.plot(ep[1:], ba[1:], color="#eb6834", lw=2, marker="s", ms=6, mec="white", label="dev balanced accuracy")
        ax.annotate(f"start: zero-shot NLI-base\n(dev AUROC {au0:.3f})", (0, au0), xytext=(12, -4),
                    textcoords="offset points", fontsize=8.5, color=MUTED)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.84, 1.0)
        ax.set_title("FaithScorer training (constructed dev split)", loc="left")
        ax.legend(loc="lower right")
        fig.savefig(FIG / "fig_training_curve.png")
        plt.close(fig)
        stats["training_curve"] = [{"epoch": float(a), "balanced_acc": float(b), "auroc": float(c)}
                                   for a, b, c in evals]

    # ---- extra statistics ---------------------------------------------------------------------------
    for split in ["train", "test"]:
        aug = json.loads((REPO_ROOT / "data" / "augmented" / f"{split}.json").read_text(encoding="utf-8"))
        ok = [a for a in aug if "failed" not in a]
        stats[f"augment_{split}"] = {
            "questions": len(aug), "passed_validation": len(ok),
            "failure_reasons": dict(Counter(a["failed"] for a in aug if "failed" in a).most_common()),
            "nli_grounded_pass": sum(a.get("nli_grounded", 0) >= 0.5 for a in ok),
            "nli_supported_pass": sum(a.get("nli_supported", 0) >= 0.5 for a in ok),
            "nli_unsupported_pass": sum(a.get("nli_unsupported", 1) < 0.5 for a in ok),
        }
    thr = report["thresholds_from_ragtruth_dev"]
    stats["ragtruth_breakdowns"] = {}
    gens = np.array([r["generator"] for r in rt])
    rcond = np.array([r["condition"] for r in rt])
    for n in ["alignscore", "faithscorer_sent"]:
        s = np.array(scores("ragtruth_qa", n)["scores"])
        pred = (s >= thr[n]).astype(int)
        by_gen = {g: float(balanced_accuracy_score(y[gens == g], pred[gens == g]))
                  if len(set(y[gens == g])) > 1 else None for g in sorted(set(gens))}
        types = Counter()
        caught = Counter()
        for c, yy, p in zip(rcond, y, pred):
            if yy == 0:
                for t in c.split("+"):
                    types[t] += 1
                    caught[t] += int(p == 0)
        stats["ragtruth_breakdowns"][n] = {
            "threshold_from_ragtruth_dev": thr[n],
            "confusion_matrix_[[TN,FP],[FN,TP]]_positive=faithful": confusion_matrix(y, pred).tolist(),
            "balanced_acc_by_generator": by_gen,
            "hallucination_recall_by_type": {t: {"n": types[t], "caught": caught[t] / types[t]} for t in types},
        }
    (RES / "extra_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
