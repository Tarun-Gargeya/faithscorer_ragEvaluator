"""FaithScorer live demo (Gradio).

Paste a retrieved context and a generated answer; FaithScorer scores every answer sentence against
the full context (sentence-min, the headline method) and returns a verdict with per-sentence evidence.
Optional: compare with NLI / AlignScore baselines, and escalate uncertain cases to the gpt-oss-120b
judge on Groq (needs GROQ_API_KEY in .env; free tier is rate-limited).

Run:  .venv/Scripts/python.exe demo/app.py      -> opens http://127.0.0.1:7860
"""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import nltk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # demo/rag.py
from llm_judge import JUDGES, PROMPT, parse  # noqa: E402
from scorers import AlignScoreScorer, CrossEncoderScorer  # noqa: E402

REPORT = json.loads((ROOT / "data" / "results" / "report.json").read_text())
ESCALATE_FRACTION = 0.20  # matches the escalation result in the README (top 20% least-confident)

# Calibration modes: threshold for sentence-min FaithScorer, tuned on that mode's dev split.
MODES = {
    "Web / RAG answers (calibrated on RAGTruth)": ("ragtruth_qa_dev", "thresholds_from_ragtruth_dev"),
    "Wikipedia-style answers (calibrated on constructed dev)": ("dev", "thresholds_from_dev"),
}


def logit(p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def mode_params(mode):
    split, key = MODES[mode]
    thr = REPORT[key]["faithscorer_sent"]
    dev = np.array(json.loads((ROOT / "data" / "scores" / split / "faithscorer_sent.json").read_text())["scores"])
    # "Uncertain" band = the 20% of dev answers closest to the threshold in log-odds.
    band = float(np.quantile(np.abs(logit(dev) - logit(thr)), ESCALATE_FRACTION))
    return thr, band


PARAMS = {m: mode_params(m) for m in MODES}


@lru_cache(maxsize=None)
def model(name):
    if name == "faithscorer":
        m = CrossEncoderScorer(str(ROOT / "models" / "faithscorer"), "faithful")
    elif name == "nli":
        m = CrossEncoderScorer("cross-encoder/nli-deberta-v3-base", "entailment", max_len=512)
    elif name == "alignscore":
        m = AlignScoreScorer()
    else:
        raise ValueError(name)
    m.score(["Warm-up context sentence."], ["Warm-up answer."])  # so displayed latency excludes CUDA warm-up
    return m


def split_sentences(text):
    return [s for s in nltk.sent_tokenize(text.strip()) if s.strip()] or [text.strip()]


# ---------------------------------------------------------------------------------------------------
# Examples: the worked HotpotQA question in all 8 conditions + two real RAGTruth answers
# ---------------------------------------------------------------------------------------------------
def load_examples():
    ex, labels = [], []
    web, wiki = list(MODES)
    test = json.loads((ROOT / "data" / "constructed" / "test.json").read_text(encoding="utf-8"))
    nice = {"clean": "faithful", "noisy": "faithful, with distractors", "cf_followed": "faithful but factually wrong",
            "supported_extra": "faithful extra detail", "cf_contradicted": "contradicts the context",
            "incomplete": "evidence missing", "irrelevant": "context irrelevant",
            "unsupported_extra": "invented detail"}
    for r in test:
        if r["qid"] == "5a89fea655429970aeb701eb":
            ex.append([r["question"], r["context"], r["answer"], wiki])
            labels.append(f"Estevez · {r['condition']} ({nice[r['condition']]})")
    rt = {r["id"]: r for r in json.loads((ROOT / "data" / "ood" / "ragtruth_qa.json").read_text(encoding="utf-8"))}
    for rid, lab in [("12295", "RAGTruth · real answer, faithful"), ("16360", "RAGTruth · real answer, hallucinated")]:
        if rid in rt:
            ex.append([rt[rid]["question"], rt[rid]["context"], rt[rid]["answer"], web])
            labels.append(lab)
    return ex, labels


EXAMPLES, EXAMPLE_LABELS = load_examples()


# ---------------------------------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------------------------------
def verdict_card(faithful, p, thr, uncertain, ms, n_sent, weakest):
    color = "#1baf7a" if faithful else "#e34948"
    title = "FAITHFUL" if faithful else "UNFAITHFUL"
    icon = "✔" if faithful else "✖"
    conf = ("<b>Uncertain</b>: in the least-confident 20% band → worth escalating to the LLM judge"
            if uncertain else "<b>Confident</b>: outside the uncertainty band")
    why = ("Every sentence is supported by the context." if faithful
           else f"Weakest sentence: <i>“{weakest}”</i>")
    return f"""
<div style="border:2px solid {color};border-radius:12px;padding:16px 18px;">
  <div style="font-size:26px;font-weight:700;color:{color};letter-spacing:.5px">{icon} {title}</div>
  <div style="margin-top:6px;opacity:.9">{why}</div>
  <table style="margin-top:10px;width:100%;font-size:14px">
    <tr><td>Answer score (min over {n_sent} sentence{'s' if n_sent != 1 else ''})</td><td style="text-align:right"><b>{p:.4f}</b></td></tr>
    <tr><td>Decision threshold</td><td style="text-align:right">{thr:.4f}</td></tr>
    <tr><td>Confidence</td><td style="text-align:right">{'uncertain' if uncertain else 'confident'}</td></tr>
    <tr><td>FaithScorer latency</td><td style="text-align:right">{ms:.0f} ms</td></tr>
  </table>
  <div style="margin-top:8px;font-size:13px;opacity:.8">{conf}</div>
</div>"""


def call_judge(question, context, answer):
    import os

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    base_url, key_env, model_id, _, extra = JUDGES["gptoss120b"]
    if not os.environ.get(key_env):
        return "⚠️ No `GROQ_API_KEY` in `.env`, so the LLM judge is unavailable."
    client = OpenAI(base_url=base_url, api_key=os.environ[key_env], timeout=60, max_retries=0)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_id, temperature=0, messages=[{"role": "user", "content": PROMPT.format(
                context=context, question=question or "(not given)", answer=answer)}], **extra)
    except Exception as e:  # rate limits on the free tier are common
        return f"⚠️ Judge call failed: `{type(e).__name__}` — {str(e)[:200]}"
    secs = time.perf_counter() - t0
    v = parse(resp.choices[0].message.content or "")
    label = {1: "✔ FAITHFUL", 0: "✖ UNFAITHFUL", None: "? unparseable reply"}[v]
    return f"**gpt-oss-120b judge:** {label}  \n*{secs * 1000:.0f} ms API round-trip (FaithScorer is ~25–50 ms, local)*"


def check(question, context, answer, mode, compare, auto_escalate):
    if not context.strip() or not answer.strip():
        raise gr.Error("Please provide both a context and an answer.")
    thr, band = PARAMS[mode]
    sents = split_sentences(answer)
    t0 = time.perf_counter()
    probs = model("faithscorer").score([context] * len(sents), sents)
    ms = 1000 * (time.perf_counter() - t0)
    p = float(probs.min())
    faithful = p >= thr
    uncertain = abs(float(logit(p) - logit(thr))) < band
    weakest = sents[int(probs.argmin())]

    card = verdict_card(faithful, p, thr, uncertain, ms, len(sents), weakest)
    highlights = [(s + " ", "supported" if q >= thr else "unsupported") for s, q in zip(sents, probs)]
    table = [[i + 1, s, round(float(q), 4), "supported" if q >= thr else "unsupported"]
             for i, (s, q) in enumerate(zip(sents, probs))]

    rows = []
    if compare:
        key = MODES[mode][1]
        whole = model("faithscorer").score([context], [answer])
        rows.append(["FaithScorer, whole answer (184M)", round(float(whole[0]), 4),
                     "faithful" if whole[0] >= REPORT[key]["faithscorer"] else "unfaithful"])
        for name, label in [("nli", "NLI DeBERTa-v3-base (184M)"), ("alignscore", "AlignScore-base (125M)")]:
            t1 = time.perf_counter()
            s = float(model(name).score([context], [answer])[0])
            rows.append([f"{label} · {1000 * (time.perf_counter() - t1):.0f} ms", round(s, 4),
                         "faithful" if s >= REPORT[key][name] else "unfaithful"])
    judge = call_judge(question, context, answer) if (auto_escalate and uncertain) else (
        "*Not escalated: FaithScorer is confident.*" if auto_escalate else "")
    return card, highlights, table, gr.update(value=rows, visible=compare), judge


def ask_document(file, question, k, worst, prompt_name, mode, auto_escalate):
    """Full RAG round trip on the user's document, then the faithfulness check on the result."""
    import rag
    if file is None:
        raise gr.Error("Upload a PDF or text document first.")
    if not question.strip():
        raise gr.Error("Type a question about the document.")
    path = file if isinstance(file, str) else file.name
    t0 = time.perf_counter()
    hits, n_chunks = rag.retrieve(path, question, k=int(k), worst=worst)
    t1 = time.perf_counter()
    passages = [h[0] for h in hits]
    answer = rag.generate(question, passages, prompt_name)
    t2 = time.perf_counter()
    context = "\n\n".join(passages)

    which = "**least** relevant (simulated bad retrieval)" if worst else "most relevant"
    shown = "\n\n".join(f"**[{i + 1}]** *(similarity {s:.2f})* {p}" for i, (p, s) in enumerate(hits))
    status = (f"Retrieved the {len(hits)} {which} of {n_chunks} chunks in {1000 * (t1 - t0):.0f} ms · "
              f"Qwen2.5-7B answered in {t2 - t1:.1f} s")
    if rag.is_abstention(answer):
        card = ("<div style='border:2px solid #6b6a63;border-radius:12px;padding:16px 18px'>"
                "<div style='font-size:22px;font-weight:700'>— ABSTAINED</div>"
                "<div style='margin-top:6px'>The generator said the passages don't contain the answer, "
                "so there is no claim to check. That is the correct behaviour under bad retrieval.</div></div>")
        return answer, shown, status, card, [], [], ""
    card, hl, table, _, judge = check(question, context, answer, mode, False, auto_escalate)
    return answer, shown, status, card, hl, table, judge


# ---------------------------------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------------------------------
CSS = """
#title h1 {margin-bottom: 0}
.gradio-container {max-width: 1280px !important}
"""

with gr.Blocks(title="FaithScorer demo") as demo:
    gr.Markdown(
        "# FaithScorer: is this RAG answer faithful to its retrieved context?\n"
        "A 184M-parameter scorer trained with *labels by construction* across 8 retrieval conditions. "
        "It scores each answer sentence against the full context and flags the answer if any sentence is unsupported. "
        "**Faithful** means supported by the context, *not* necessarily true in the real world.",
        elem_id="title")

    with gr.Tab("Ask your document"):
        gr.Markdown("Upload a document and ask a question. A small RAG system answers it (MiniLM retrieval + "
                    "local Qwen2.5-7B), and FaithScorer checks whether the answer is supported by what was retrieved.")
        with gr.Row():
            with gr.Column(scale=5):
                doc = gr.File(label="Document (PDF, TXT or MD)", file_types=[".pdf", ".txt", ".md"])
                d_question = gr.Textbox(label="Question", lines=2)
                with gr.Row():
                    d_k = gr.Slider(1, 5, value=3, step=1, label="Passages to retrieve")
                    d_worst = gr.Checkbox(label="Simulate bad retrieval (fetch the LEAST relevant passages)",
                                          value=False)
                d_prompt = gr.Radio(list(__import__("rag").PROMPTS), value=list(__import__("rag").PROMPTS)[0],
                                    label="Generator instructions")
                d_mode = gr.Radio(list(MODES), value=list(MODES)[0], label="Calibration")
                d_esc = gr.Checkbox(label="Escalate uncertain answers to gpt-oss-120b", value=False)
                d_go = gr.Button("Retrieve, answer and check", variant="primary")
                d_status = gr.Markdown()
                with gr.Accordion("Retrieved passages (the context FaithScorer checks against)", open=False):
                    d_passages = gr.Markdown()
            with gr.Column(scale=4):
                d_answer = gr.Textbox(label="Generated answer", lines=5, interactive=False)
                d_card = gr.HTML()
                d_hl = gr.HighlightedText(label="Answer, sentence by sentence",
                                          color_map={"supported": "#1baf7a", "unsupported": "#e34948"},
                                          show_legend=True, combine_adjacent=False)
                d_table = gr.Dataframe(headers=["#", "sentence", "P(faithful)", "verdict"],
                                       label="Per-sentence scores", wrap=True, interactive=False)
                d_judge = gr.Markdown()
        sample = str(ROOT / "demo" / "sample_docs" / "paris_landmarks.txt")
        strict, relaxed = list(__import__("rag").PROMPTS)
        gr.Examples(
            examples=[[sample, "When was the Eiffel Tower built and why?", False, strict],
                      [sample, "When was the Eiffel Tower built and why?", True, strict],
                      [sample, "When was the Eiffel Tower built and why?", True, relaxed],
                      [sample, "When did Notre-Dame reopen after the fire?", False, strict]],
            example_labels=["Good retrieval → faithful answer",
                            "Bad retrieval, strict generator → abstains",
                            "Bad retrieval, relaxed generator → answers from memory (should be flagged)",
                            "Notre-Dame question, good retrieval"],
            inputs=[doc, d_question, d_worst, d_prompt], label="Sample document: a short guide to Paris landmarks")
        gr.Markdown("<small>Try the same question twice: once normally, once with *Simulate bad retrieval* and "
                    "*Relaxed* instructions. The generator then answers from its own memory, and FaithScorer "
                    "should flag the answer even if it is factually correct.</small>")
        d_go.click(ask_document, [doc, d_question, d_k, d_worst, d_prompt, d_mode, d_esc],
                   [d_answer, d_passages, d_status, d_card, d_hl, d_table, d_judge])

    with gr.Tab("Check an answer"):
        with gr.Row():
            with gr.Column(scale=5):
                question = gr.Textbox(label="Question (optional)", lines=1)
                context = gr.Textbox(label="Retrieved context", lines=12, placeholder="Paste the retrieved passages…")
                answer = gr.Textbox(label="Generated answer", lines=4, placeholder="Paste the RAG system's answer…")
                mode = gr.Radio(list(MODES), value=list(MODES)[0], label="Calibration")
                with gr.Row():
                    compare = gr.Checkbox(label="Compare with baselines (NLI, AlignScore)", value=False)
                    auto_esc = gr.Checkbox(label="Escalate uncertain answers to gpt-oss-120b", value=False)
                with gr.Row():
                    go = gr.Button("Check faithfulness", variant="primary")
                    ask = gr.Button("Ask the LLM judge")
            with gr.Column(scale=4):
                card = gr.HTML(label="Verdict")
                hl = gr.HighlightedText(label="Answer, sentence by sentence",
                                        color_map={"supported": "#1baf7a", "unsupported": "#e34948"},
                                        show_legend=True, combine_adjacent=False)
                table = gr.Dataframe(headers=["#", "sentence", "P(faithful)", "verdict"], label="Per-sentence scores",
                                     wrap=True, interactive=False)
                base = gr.Dataframe(headers=["scorer", "score", "verdict"], label="Baselines (same calibration mode)",
                                    visible=False, interactive=False)
                judge = gr.Markdown()
        gr.Examples(examples=EXAMPLES, inputs=[question, context, answer, mode], example_labels=EXAMPLE_LABELS,
                    label="Examples: one HotpotQA question in all 8 retrieval conditions, plus real RAGTruth answers")
        go.click(check, [question, context, answer, mode, compare, auto_esc], [card, hl, table, base, judge])
        ask.click(call_judge, [question, context, answer], judge)

    gr.Markdown(
        "<small>Calibration: *Web / RAG* uses the threshold tuned on RAGTruth's training split (real LLM answers "
        "over web passages); *Wikipedia-style* uses the threshold tuned on the constructed dev split. "
        "The uncertainty band is the 20% of dev answers closest to the threshold. Details: README, Chapter 5.</small>")

if __name__ == "__main__":
    print("Loading FaithScorer…")
    model("faithscorer")
    try:
        demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
    except TypeError:  # older Gradio: theme/css belong to Blocks
        demo.launch(inbrowser=True)
