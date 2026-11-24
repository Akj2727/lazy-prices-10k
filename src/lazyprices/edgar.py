"""Download 10-K filings from SEC EDGAR, with an on-disk cache.

SEC asks every client to identify itself, so set SEC_USER_AGENT to something like
"Your Name your.email@example.com" before running. Requests are throttled to stay
under SEC's limit of 10 per second.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

CACHE = Path(__file__).resolve().parents[2] / "cache" / "edgar"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
MIN_INTERVAL = 0.12  # seconds between requests

_last_request = 0.0


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    form: str
    filing_date: str  # YYYY-MM-DD, the date the filing became public
    report_date: str  # fiscal period end
    accession: str
    primary_doc: str
    sic: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""),
                                  doc=self.primary_doc)


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError('Set SEC_USER_AGENT, e.g. "Your Name your.email@example.com"')
    return ua


def _get(url: str) -> bytes:
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        _last_request = time.monotonic()
        resp = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=60)
        if resp.status_code == 200:
            return resp.content
        if resp.status_code in (429, 503):
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"gave up on {url}")


def _cached(path: Path, url: str) -> bytes:
    if path.exists():
        return path.read_bytes()
    data = _get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def ticker_to_cik() -> dict[str, int]:
    raw = json.loads(_cached(CACHE / "company_tickers.json", TICKERS_URL))
    return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}


def _rows(block: dict) -> list[dict]:
    keys = ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument")
    return [dict(zip(keys, vals)) for vals in zip(*(block.get(k, []) for k in keys))]


def list_10k(ticker: str, cik: int, start_year: int, end_year: int) -> list[Filing]:
    """Original 10-K filings (no amendments) filed between start_year and end_year."""
    sub = json.loads(_cached(CACHE / "submissions" / f"{cik}.json",
                             SUBMISSIONS_URL.format(cik=cik)))
    rows = _rows(sub["filings"]["recent"])
    for extra in sub["filings"].get("files", []):
        name = extra["name"]
        page = json.loads(_cached(CACHE / "submissions" / name,
                                  f"https://data.sec.gov/submissions/{name}"))
        rows += _rows(page)
    out = []
    for r in rows:
        year = int(r["filingDate"][:4])
        if r["form"] == "10-K" and start_year <= year <= end_year and r["primaryDocument"]:
            out.append(Filing(ticker, cik, r["form"], r["filingDate"], r["reportDate"],
                              r["accessionNumber"], r["primaryDocument"], str(sub.get("sic", ""))))
    return sorted(out, key=lambda f: f.filing_date)


def fetch_document(filing: Filing) -> bytes:
    path = CACHE / "docs" / str(filing.cik) / f"{filing.accession}.html"
    return _cached(path, filing.url)
