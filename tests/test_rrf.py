from src.retrieval import RRF_K, _rrf_fuse


def test_chunk_in_both_lists_beats_chunk_in_one():
    fused = _rrf_fuse([["a", "b", "c"], ["b", "x"]])
    assert fused["b"] > fused["a"]


def test_single_list_chunk_keeps_its_rank_credit():
    # The old min-max blend scored a chunk missing from one list as 0 on that
    # side. With RRF, BM25's #1 is worth exactly what dense's #1 is worth.
    fused = _rrf_fuse([["dense_top", "d2"], ["bm25_top", "b2"]])
    assert fused["bm25_top"] == fused["dense_top"] == 1 / (RRF_K + 1)
    assert fused["bm25_top"] > fused["d2"]


def test_empty_ranking_is_ignored():
    assert _rrf_fuse([["a"], []]) == {"a": 1 / (RRF_K + 1)}
