"""
Ground-truth question/PDF/relevant-page triples for the IR eval harness (eval.py).

relevant_pages lists the 1-indexed page numbers known to contain the answer;
recall@k and MRR are computed by treating each retrieved chunk's page number
as a chunk_id proxy and checking whether it falls in relevant_pages.
"""

EVAL_PAIRS = [
    # {
    #     "question": "What is the termination notice period?",
    #     "pdf_path": "docs/contract.pdf",
    #     "relevant_pages": [3],
    # },
]
