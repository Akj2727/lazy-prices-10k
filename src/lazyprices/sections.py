"""Turn a 10-K HTML document into plain text and cut out Item 1A and Item 7.

Headings like "Item 1A. Risk Factors" usually appear twice: once in the table of
contents and once at the real section. The table-of-contents copy is followed almost
immediately by the next heading, so for every candidate start we measure the span to
the next end heading and keep the longest one.
"""
from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SECTIONS = {
    "risk": (r"item\s*1a\s*[\.\:\-–—]?\s*risk\s+factors",
             r"item\s*(?:1b\s*[\.\:\-–—]?\s*unresolved|1c\s*[\.\:\-–—]?\s*cybersecurity|2\s*[\.\:\-–—]?\s*properties)"),
    "mdna": (r"item\s*7\s*[\.\:\-–—]?\s*management\W*s\s+discussion(?:\s+and\s+analysis)?(?:\s+of\s+financial\s+condition\s+and\s+results\s+of\s+operations)?",
             r"item\s*(?:7a\s*[\.\:\-–—]?\s*quantitative|8\s*[\.\:\-–—]?\s*financial\s+statements)"),
}
MIN_CHARS = 2000


def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()  # hidden inline XBRL facts
    text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract(text: str, start_pat: str, end_pat: str, min_chars: int = MIN_CHARS) -> str | None:
    starts = [m.end() for m in re.finditer(start_pat, text, re.I)]
    ends = [m.start() for m in re.finditer(end_pat, text, re.I)]
    best = ""
    for s in starts:
        e = next((e for e in ends if e > s), len(text))
        if e - s > len(best):
            best = text[s:e]
    best = best.strip()
    return best if len(best) >= min_chars else None


def extract_sections(html: bytes | str) -> dict[str, str | None]:
    text = html_to_text(html)
    return {name: extract(text, s, e) for name, (s, e) in SECTIONS.items()}
