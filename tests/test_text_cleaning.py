from src.ingestion import _clean_text
from src.retrieval import _tokenize


def test_ligatures_are_expanded():
    assert _clean_text("classiﬁcation and ﬂow") == "classification and flow"


def test_tokenizer_matches_across_ligatures():
    assert _tokenize("ﬁne-tuning classiﬁcation") == _tokenize("fine-tuning classification")
    assert "classification" in _tokenize("classiﬁcation")


def test_hyphenated_line_breaks_are_rejoined():
    assert _clean_text("it is unlikely to be op-\ntimal") == "it is unlikely to be optimal"


def test_real_hyphens_and_capitalized_breaks_are_kept():
    assert _clean_text("pre-trained") == "pre-trained"
    assert _clean_text("ResNet-\nResNet") == "ResNet-\nResNet"
