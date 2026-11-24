from lazyprices.sections import extract_sections, html_to_text

BODY_RISK = " ".join(["Our business faces competition and supply chain risk in many markets."] * 60)
BODY_MDNA = " ".join(["Revenue grew because demand for our products increased this year."] * 60)

HTML = f"""<html><body>
<p>Table of Contents</p>
<p>Item 1A. Risk Factors 12</p><p>Item 1B. Unresolved Staff Comments 20</p>
<p>Item 7. Management's Discussion and Analysis 30</p><p>Item 7A. Quantitative and Qualitative 40</p>
<div style="display:none">ix hidden fact 123</div>
<h2>Item 1A.&nbsp;Risk Factors</h2><p>{BODY_RISK}</p>
<h2>Item 1B. Unresolved Staff Comments</h2><p>None.</p>
<h2>Item 2. Properties</h2><p>We lease offices.</p>
<h2>ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS</h2><p>{BODY_MDNA}</p>
<h2>Item 7A. Quantitative and Qualitative Disclosures</h2><p>Rates.</p>
<h2>Item 8. Financial Statements</h2>
</body></html>"""


def test_skips_table_of_contents_and_finds_real_sections():
    s = extract_sections(HTML)
    assert s["risk"].startswith("Our business faces competition")
    assert "Unresolved" not in s["risk"]
    assert s["mdna"].startswith("Revenue grew")
    assert "Quantitative" not in s["mdna"]


def test_hidden_xbrl_text_is_dropped():
    assert "ix hidden fact" not in html_to_text(HTML)


def test_missing_section_returns_none():
    s = extract_sections("<p>Item 1A. Risk Factors</p><p>Short.</p><p>Item 2. Properties</p>")
    assert s["risk"] is None and s["mdna"] is None
