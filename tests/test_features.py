import math

from lazyprices.features import change_features, cosine, jaccard, len_change, new_share

A = "We rely on suppliers in Asia for our components. Demand may fall in a recession. " * 20
B = A + "A new cyber attack disrupted our systems and we may face lawsuits over the breach. " * 10


def test_identical_text_scores_as_unchanged():
    f = change_features(A, A)
    assert math.isclose(f["cosine"], 1.0) and math.isclose(f["jaccard"], 1.0)
    assert f["len_change"] == 0.0 and f["new_share"] == 0.0


def test_added_disclosure_lowers_similarity_and_raises_new_share():
    assert cosine(B, A) < 1.0 and jaccard(B, A) < 1.0
    assert len_change(B, A) > 0
    assert 0 < new_share(B, A) < 1


def test_missing_text_gives_nan():
    assert all(math.isnan(v) for v in change_features(None, A).values())
