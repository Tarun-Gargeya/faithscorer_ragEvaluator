"""Minimal RAG pipeline for the demo: document -> chunks -> hybrid retrieval -> local Qwen answer.

Kept small on purpose: the point of the demo is the faithfulness check on top, not the RAG system.
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

GENERATOR = "qwen2.5:7b-instruct"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

PROMPTS = {
    "Strict (answer only from the passages)": (
        "Answer the question using ONLY the passages below. Answer in 2-4 sentences. "
        "If the passages do not contain the answer, say exactly: I don't know.\n\n"
        "Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"),
    "Relaxed (may use its own knowledge)": (
        "Answer the question in 2-4 sentences. Use the passages below where helpful; you may also use "
        "your own knowledge.\n\nPassages:\n{passages}\n\nQuestion: {question}\nAnswer:"),
}
ABSTAIN_RE = re.compile(r"\b(i don't know|i do not know|(?:do|does) not (?:contain|mention|provide|include)|"
                        r"no information|cannot (?:determine|answer))\b", re.I)


def read_document(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text, size=120, overlap=30):
    """Word windows of `size` words with `overlap`; small enough that top-3 fits FaithScorer's 1,024 tokens."""
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step) if words[i:i + size]]


@lru_cache(maxsize=1)
def embedder():
    return SentenceTransformer(EMBEDDER, device="cuda")


@lru_cache(maxsize=8)
def index(path, mtime):  # mtime in the cache key so an edited file is re-indexed
    from sklearn.feature_extraction.text import TfidfVectorizer
    chunks = chunk(read_document(path))
    vecs = embedder().encode(chunks, normalize_embeddings=True, batch_size=64)
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(chunks)
    return chunks, vecs, tfidf, tfidf.transform(chunks)


def retrieve(path, question, k=3, worst=False):
    """Top-k chunks by hybrid similarity; worst=True returns the k LEAST similar (simulated bad retrieval).

    Hybrid = mean of MiniLM cosine and TF-IDF cosine: embeddings alone latched onto chunks that merely
    mention the topic and missed keyword-heavy ones (tables, numbers).
    """
    chunks, vecs, tfidf, tf_chunks = index(str(path), Path(path).stat().st_mtime)
    dense = vecs @ embedder().encode([question], normalize_embeddings=True)[0]
    sparse = (tf_chunks @ tfidf.transform([question]).T).toarray().ravel()
    sims = 0.5 * dense + 0.5 * sparse
    order = np.argsort(sims)
    picked = order[:k] if worst else order[::-1][:k]
    return [(chunks[i], float(sims[i])) for i in picked], len(chunks)


def generate(question, passages, prompt_name):
    text = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    resp = ollama.chat(model=GENERATOR, messages=[{"role": "user", "content": PROMPTS[prompt_name].format(
        passages=text, question=question)}], options={"temperature": 0, "seed": 0, "num_predict": 300})
    return resp["message"]["content"].strip()


def is_abstention(answer):
    """A pure refusal: the first sentence says the passages lack the answer, and little else follows.

    Longer replies that refuse and then answer from memory anyway are NOT abstentions; they get scored.
    """
    sentences = re.split(r"(?<=[.!?])\s", answer.strip())
    return bool(ABSTAIN_RE.search(sentences[0])) and len(sentences) <= 2
