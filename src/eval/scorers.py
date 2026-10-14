"""All faithfulness scorers behind one batch interface: score(contexts, answers) -> np.ndarray of P(faithful).

Baselines (off-the-shelf, no training on our data):
  nli          cross-encoder/nli-deberta-v3-base, P(entailment) over the whole context
  nli_base     our starting checkpoint (DeBERTa-v3-base-mnli-fever-anli) zero-shot, P(entailment)
               -> isolates the effect of our fine-tuning
  alignscore   AlignScore-base (nli_sp mode: chunks context, splits claims into sentences)
  minicheck    MiniCheck flan-t5-large
  (Vectara HHEM-2.1-Open was dropped: its remote code doesn't load its weights under transformers 5.x.)

Ours:
  faithscorer       DeBERTa-v3-base fine-tuned on the constructed retrieval-noise data (src/model/train.py)
  faithscorer_sent  same model, answer split into sentences, each scored against the full context;
                    answer score = MIN over sentences (one unsupported sentence => unfaithful)
  faithscorer_mean  same, MEAN over sentences (comparison for the aggregation choice)
"""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CrossEncoderScorer:
    """Any HF sequence-classification model taking (premise, hypothesis) pairs."""

    def __init__(self, name_or_path, positive_label, max_len=1024, batch_size=16):
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(name_or_path).to(DEVICE).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.pos = labels[positive_label]
        self.max_len, self.bs = max_len, batch_size

    @torch.inference_mode()
    def score(self, contexts, answers):
        out = []
        for i in range(0, len(contexts), self.bs):
            enc = self.tok(contexts[i:i + self.bs], answers[i:i + self.bs], truncation="only_first",
                           max_length=self.max_len, padding=True, return_tensors="pt").to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, -1)[:, self.pos].cpu().numpy())
        return np.concatenate(out)


class AlignScoreScorer:
    def __init__(self):
        from alignscore import AlignScore
        self.m = AlignScore(model="roberta-base", batch_size=32, device=DEVICE,
                            ckpt_path=str(REPO_ROOT / "models" / "AlignScore-base.ckpt"),
                            evaluation_mode="nli_sp", verbose=False)

    def score(self, contexts, answers):
        return np.asarray(self.m.score(contexts=list(contexts), claims=list(answers)), dtype=float)


class MiniCheckScorer:
    def __init__(self):
        from minicheck.minicheck import MiniCheck
        self.m = MiniCheck(model_name="flan-t5-large", batch_size=16)

    def score(self, contexts, answers):
        _, probs, _, _ = self.m.score(docs=list(contexts), claims=list(answers))
        return np.asarray(probs, dtype=float)


class SentenceScorer:
    """Score each answer sentence separately against the full context, then aggregate."""

    def __init__(self, inner, agg):
        import nltk
        self.split = nltk.sent_tokenize
        self.inner, self.agg = inner, agg

    def score(self, contexts, answers):
        pairs, owner = [], []
        for i, (c, a) in enumerate(zip(contexts, answers)):
            sents = [x for x in self.split(a) if x.strip()] or [a]
            pairs += [(c, x) for x in sents]
            owner += [i] * len(sents)
        flat = self.inner.score([c for c, _ in pairs], [x for _, x in pairs])
        owner = np.array(owner)
        return np.array([self.agg(flat[owner == i]) for i in range(len(contexts))])


def get_scorer(name):
    if name == "nli":
        return CrossEncoderScorer("cross-encoder/nli-deberta-v3-base", "entailment", max_len=512)
    if name == "nli_base":
        return CrossEncoderScorer("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "entailment")
    if name == "alignscore":
        return AlignScoreScorer()
    if name == "minicheck":
        return MiniCheckScorer()
    if name == "faithscorer":
        return CrossEncoderScorer(str(REPO_ROOT / "models" / "faithscorer"), "faithful")
    if name in ("faithscorer_sent", "faithscorer_mean"):
        inner = CrossEncoderScorer(str(REPO_ROOT / "models" / "faithscorer"), "faithful")
        return SentenceScorer(inner, np.min if name == "faithscorer_sent" else np.mean)
    if name.endswith("_sent") and name[:-5] in ("nli", "nli_base", "minicheck"):
        # Ablation: baselines with the same sentence-min aggregation as faithscorer_sent.
        return SentenceScorer(get_scorer(name[:-5]), np.min)
    raise ValueError(name)


SCORERS = ["nli", "nli_base", "alignscore", "minicheck", "faithscorer", "faithscorer_sent", "faithscorer_mean"]
# Not in SCORERS (so the main report tables are unchanged); scored by extra_checks.py on RAGTruth only.
SENT_BASELINES = ["nli_sent", "nli_base_sent", "minicheck_sent"]
