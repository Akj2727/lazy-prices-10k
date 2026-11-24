"""Optional FinBERT tone score for a section of text.

Needs the finbert extra: pip install -e ".[finbert]". The score is the mean of
P(positive) - P(negative) over a sample of sentences, so it runs on a laptop CPU.
The signal uses the year-over-year change in this score.
"""
from __future__ import annotations

import numpy as np

from .features import sentences

MODEL = "ProsusAI/finbert"
MAX_SENTENCES = 200

_pipe = None


def _pipeline():
    global _pipe
    if _pipe is None:
        from transformers import pipeline  # imported lazily: optional dependency
        _pipe = pipeline("text-classification", model=MODEL, top_k=None, truncation=True)
    return _pipe


def tone(text: str | None, max_sentences: int = MAX_SENTENCES, seed: int = 0) -> float:
    if not text:
        return float("nan")
    sents = sentences(text)
    if not sents:
        return float("nan")
    if len(sents) > max_sentences:
        rng = np.random.default_rng(seed)
        sents = list(rng.choice(sents, max_sentences, replace=False))
    scores = []
    for out in _pipeline()(sents, batch_size=16):
        p = {d["label"].lower(): d["score"] for d in out}
        scores.append(p.get("positive", 0.0) - p.get("negative", 0.0))
    return float(np.mean(scores))
