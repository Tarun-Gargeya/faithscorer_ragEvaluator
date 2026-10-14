"""Convert RAGTruth (QA task) into our (context, answer, label) format for OOD evaluation.

  --split test  -> data/ood/ragtruth_qa.json      final OOD test set (never used for any decision)
  --split train -> data/ood/ragtruth_qa_dev.json  random sample of RAGTruth's *train* split, used only to
                   choose OOD design options (e.g. sentence-level aggregation, thresholds); its source
                   questions are disjoint from the test split

RAGTruth: real responses from 6 LLMs to MS MARCO questions with retrieved passages, with
human span-level hallucination annotations. label = 1 (faithful) iff no span is annotated.
Neither split is used for training.

Download first:
  https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{response,source_info}.jsonl
  -> data/raw/ragtruth/
"""
import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "ragtruth"
OUT_DIR = REPO_ROOT / "data" / "ood"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--dev-size", type=int, default=900)
    args = ap.parse_args()
    out = OUT_DIR / ("ragtruth_qa.json" if args.split == "test" else "ragtruth_qa_dev.json")

    sources = {s["source_id"]: s for s in read_jsonl(RAW / "source_info.jsonl") if s["task_type"] == "QA"}
    rows = []
    for r in read_jsonl(RAW / "response.jsonl"):
        if r["source_id"] not in sources or r["split"] != args.split:
            continue
        info = sources[r["source_id"]]["source_info"]
        types = sorted({lab["label_type"] for lab in r["labels"]})
        rows.append({
            "qid": r["source_id"], "id": r["id"], "generator": r["model"],
            "condition": "+".join(types) if types else "faithful",
            "label": int(not r["labels"]),
            "question": info["question"], "context": info["passages"].strip(), "answer": r["response"].strip(),
        })
    if args.split == "train":
        rows = random.Random(0).sample(rows, min(args.dev_size, len(rows)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} examples ({pos} faithful / {len(rows) - pos} hallucinated) -> {out}")


if __name__ == "__main__":
    main()
