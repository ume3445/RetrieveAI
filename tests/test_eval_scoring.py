from evaluation.textnorm import contains_span


def test_span_match_ignores_pdf_artifacts():
    chunk = "it is unlikely that identity mappings are op-\ntimal, the classiﬁcation of size r"
    assert contains_span(chunk, "mappings are optimal")
    assert contains_span(chunk, "classification")
    assert contains_span(chunk, "of sizer")  # spacing differences don't matter


def test_span_match_rejects_absent_answer():
    assert not contains_span("BLEU of 41.8 on English-to-French", "28.4 BLEU")
