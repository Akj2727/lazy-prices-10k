"""Year-over-year text change features between a filing section and last year's version.

  * cosine:     cosine similarity of word-count vectors (the Lazy Prices measure)
  * jaccard:    overlap of the two vocabularies
  * len_change: log ratio of word counts, so growth in risk disclosures is positive
  * new_share:  share of this year's sentences that do not appear last year
"""
from __future__ import annotations

import math
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

TOKEN = re.compile(r"[a-z]{2,}")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    norm = (re.sub(r"[^a-z ]", "", p.lower()).strip() for p in parts)
    return [s for s in norm if len(s.split()) >= 5]


def cosine(a: str, b: str) -> float:
    vec = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", lowercase=True)
    m = vec.fit_transform([a, b]).astype(float)
    x, y = m[0].toarray().ravel(), m[1].toarray().ravel()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom else float("nan")


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokens(a)), set(tokens(b))
    union = sa | sb
    return len(sa & sb) / len(union) if union else float("nan")


def len_change(cur: str, prev: str) -> float:
    n_cur, n_prev = len(tokens(cur)), len(tokens(prev))
    return math.log(n_cur / n_prev) if n_cur and n_prev else float("nan")


def new_share(cur: str, prev: str) -> float:
    cur_s, prev_s = sentences(cur), set(sentences(prev))
    return sum(s not in prev_s for s in cur_s) / len(cur_s) if cur_s else float("nan")


def change_features(cur: str | None, prev: str | None) -> dict[str, float]:
    if not cur or not prev:
        return {k: float("nan") for k in ("cosine", "jaccard", "len_change", "new_share")}
    return {
        "cosine": cosine(cur, prev),
        "jaccard": jaccard(cur, prev),
        "len_change": len_change(cur, prev),
        "new_share": new_share(cur, prev),
    }
