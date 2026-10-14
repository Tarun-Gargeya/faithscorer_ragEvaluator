"""Score a dataset split with one or more scorers; cache per-example scores + timing.

Output: data/scores/<split>/<scorer>.json = {"scores": [...], "seconds": total_wall_time, "n": N}
Examples are scored in the order of data/constructed/<split>.json (or data/ood/<split>.json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorers import SCORERS, get_scorer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_split(split):
    for d in ["constructed", "ood"]:
        p = REPO_ROOT / "data" / d / f"{split}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--scorers", nargs="+", default=SCORERS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = load_split(args.split)
    contexts, answers = [r["context"] for r in rows], [r["answer"] for r in rows]
    out_dir = REPO_ROOT / "data" / "scores" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.scorers:
        path = out_dir / f"{name}.json"
        if path.exists() and not args.force:
            print(f"{name}: cached")
            continue
        scorer = get_scorer(name)
        scorer.score(contexts[:4], answers[:4])  # warm-up, excluded from timing
        t0 = time.perf_counter()
        scores = scorer.score(contexts, answers)
        secs = time.perf_counter() - t0
        path.write_text(json.dumps({"scores": [float(s) for s in scores], "seconds": secs, "n": len(rows)}))
        print(f"{name}: {len(rows)} examples in {secs:.1f}s ({1000 * secs / len(rows):.1f} ms/example)")
        del scorer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
