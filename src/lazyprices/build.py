"""Build the filing-level feature table: one row per 10-K that has a prior-year 10-K.

For each firm, consecutive 10-Ks are compared section by section. A pair only counts
if the earlier filing is 9 to 15 months older, so a missed year or a fiscal-year change
does not get compared against the wrong document.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import edgar
from .features import change_features
from .sections import extract_sections

log = logging.getLogger(__name__)
SECTION_NAMES = ("risk", "mdna")


def _pairs(filings: list[edgar.Filing]):
    for prev, cur in zip(filings, filings[1:]):
        gap = (pd.Timestamp(cur.filing_date) - pd.Timestamp(prev.filing_date)).days
        if 270 <= gap <= 460:
            yield prev, cur


def firm_rows(ticker: str, cik: int, start_year: int, end_year: int,
              finbert: bool = False) -> list[dict]:
    filings = edgar.list_10k(ticker, cik, start_year - 1, end_year)
    texts: dict[str, dict] = {}

    def sections_of(f: edgar.Filing) -> dict:
        if f.accession not in texts:
            texts[f.accession] = extract_sections(edgar.fetch_document(f))
        return texts[f.accession]

    rows = []
    for prev, cur in _pairs(filings):
        if int(cur.filing_date[:4]) < start_year:
            continue
        s_prev, s_cur = sections_of(prev), sections_of(cur)
        row = {"ticker": ticker, "cik": cik, "sic": cur.sic, "filing_date": cur.filing_date,
               "prev_filing_date": prev.filing_date, "accession": cur.accession}
        for name in SECTION_NAMES:
            row[f"{name}_found"] = bool(s_cur[name]) and bool(s_prev[name])
            for k, v in change_features(s_cur[name], s_prev[name]).items():
                row[f"{name}_{k}"] = v
        if finbert:
            from .sentiment import tone
            row["mdna_tone_change"] = tone(s_cur["mdna"]) - tone(s_prev["mdna"])
        rows.append(row)
    return rows


def build_features(tickers: list[str], start_year: int, end_year: int, out: Path,
                   finbert: bool = False) -> pd.DataFrame:
    ciks = edgar.ticker_to_cik()
    rows, missing = [], []
    for i, t in enumerate(tickers, 1):
        cik = ciks.get(t.upper().replace(".", "-"))
        if cik is None:
            missing.append(t)
            continue
        try:
            rows += firm_rows(t, cik, start_year, end_year, finbert)
        except Exception as exc:  # one bad filing should not stop a 500-firm run
            log.warning("%s failed: %s", t, exc)
        if i % 25 == 0:
            log.info("%d/%d firms, %d filing pairs so far", i, len(tickers), len(rows))
    if missing:
        log.info("no CIK for %d tickers: %s", len(missing), ", ".join(missing[:20]))
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
