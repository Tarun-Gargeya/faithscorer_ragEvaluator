"""Stage B2: independent NLI verification of the LLM-written text, to clean the by-construction labels.

A manual spot-check of Stage B output found ~1 in 12 grounded answers smuggled in a claim the
context doesn't support (which would put a hallucinated answer under a 'faithful' label). We
therefore check every LLM-written piece against the *gold paragraphs only* with a large NLI model
that is NOT one of the evaluated scorers (DeBERTa-v3-large; our scorer starts from the -base model):

  grounded_answer    must be entailed     -> otherwise the whole question is dropped
  supported_extra    must be entailed     -> otherwise the supported_extra condition is skipped
  unsupported_extra  must NOT be entailed -> otherwise the unsupported_extra condition is skipped

Adds nli_grounded / nli_supported / nli_unsupported (P(entailment)) to data/augmented/<split>.json.
Known bias: kept examples are ones this NLI model agrees with; RAGTruth (human labels) is unaffected.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
THRESHOLD = 0.5


@torch.inference_mode()
def entailment(model, tok, premises, hypotheses, bs=16):
    ent = model.config.label2id.get("entailment", 0)
    out = []
    for i in range(0, len(premises), bs):
        enc = tok(premises[i:i + bs], hypotheses[i:i + bs], truncation="only_first", max_length=1024,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        out += torch.softmax(logits, -1)[:, ent].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(VERIFIER)
    model = AutoModelForSequenceClassification.from_pretrained(VERIFIER).to("cuda").eval()
    base = {r["id"]: r for r in json.loads((REPO_ROOT / "data/base" / f"{args.split}.json").read_text(encoding="utf-8"))}
    aug_path = REPO_ROOT / "data/augmented" / f"{args.split}.json"
    aug = json.loads(aug_path.read_text(encoding="utf-8"))
    ok = [a for a in aug if "failed" not in a]
    ctx = [" ".join(s for p in base[a["id"]]["gold"] for s in p["sentences"]) for a in ok]

    for field, key in [("grounded_answer", "nli_grounded"), ("supported_extra", "nli_supported"),
                       ("unsupported_extra", "nli_unsupported")]:
        for a, p in zip(ok, entailment(model, tok, ctx, [a[field] for a in ok])):
            a[key] = round(p, 4)

    aug_path.write_text(json.dumps(aug, indent=1), encoding="utf-8")
    n = len(ok)
    g = sum(a["nli_grounded"] >= THRESHOLD for a in ok)
    s = sum(a["nli_supported"] >= THRESHOLD for a in ok)
    u = sum(a["nli_unsupported"] < THRESHOLD for a in ok)
    print(f"{args.split}: {n} rows | grounded entailed {g} ({g / n:.0%}) | supported entailed {s} ({s / n:.0%}) "
          f"| unsupported not entailed {u} ({u / n:.0%})")


if __name__ == "__main__":
    main()
