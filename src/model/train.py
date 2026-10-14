"""Fine-tune an NLI-pretrained DeBERTa-v3 cross-encoder as a binary faithfulness scorer.

Input pair: (context, answer). Output: P(faithful). Starts from an MNLI/FEVER/ANLI checkpoint
so it already knows entailment; we re-initialise the head for 2 classes and train on the
constructed retrieval-noise data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "constructed"
BASE_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def load(name):
    rows = json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return Dataset.from_list([{"context": r["context"], "answer": r["answer"], "label": r["label"]} for r in rows])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {"balanced_acc": balanced_accuracy_score(labels, preds),
            "f1_unfaithful": f1_score(labels, preds, pos_label=0),
            "auroc": roc_auc_score(labels, probs)}


def nli_to_binary_head(model):
    """Turn the 3-way NLI head into a 2-way faithful/unfaithful head, warm-started from NLI semantics.

    faithful <- entailment row; unfaithful <- mean of neutral and contradiction rows. A randomly
    re-initialised head made training diverge (loss spikes, grad-norm > 400) and collapse to a
    constant prediction; this way training starts from the zero-shot NLI model instead.
    """
    old = model.classifier
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    new = torch.nn.Linear(old.in_features, 2).to(old.weight.device, old.weight.dtype)
    with torch.no_grad():
        new.weight[1] = old.weight[lab["entailment"]]
        new.bias[1] = old.bias[lab["entailment"]]
        new.weight[0] = (old.weight[lab["neutral"]] + old.weight[lab["contradiction"]]) / 2
        new.bias[0] = (old.bias[lab["neutral"]] + old.bias[lab["contradiction"]]) / 2
    model.classifier = new
    model.num_labels = model.config.num_labels = 2
    model.config.id2label = {0: "unfaithful", 1: "faithful"}
    model.config.label2id = {"unfaithful": 0, "faithful": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "faithscorer"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="for quick sanity runs")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    # fp32 master weights: transformers 5.x otherwise loads this checkpoint in its stored fp16, and training
    # pure-fp16 weights silently collapses (updates round away / overflow). bf16 autocast is applied by Trainer.
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, dtype=torch.float32)
    nli_to_binary_head(model)

    def tokenize(batch):
        # Truncate only the context; the answer is what we're checking and must stay whole.
        return tok(batch["context"], batch["answer"], truncation="only_first", max_length=args.max_len)

    train_ds = load("train").map(tokenize, batched=True, remove_columns=["context", "answer"])
    dev_ds = load("dev").map(tokenize, batched=True, remove_columns=["context", "answer"])

    # 10% linear warmup (transformers 5.x dropped warmup_ratio, so convert to steps).
    total_steps = int(len(train_ds) / (args.bs * args.grad_accum) * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out + "_ckpt", num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.grad_accum, warmup_steps=max(1, total_steps // 10), weight_decay=0.01,
        bf16=True, gradient_checkpointing=True, eval_strategy="steps", eval_steps=min(200, args.max_steps) if args.max_steps > 0 else 200, save_strategy="steps",
        save_steps=min(200, args.max_steps) if args.max_steps > 0 else 200,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="balanced_acc",
        logging_steps=50, report_to="none", seed=42, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=1))
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "dev_metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
