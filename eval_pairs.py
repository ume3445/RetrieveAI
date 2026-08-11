"""
Ground-truth question/PDF/relevant-page triples for the IR eval harness (eval.py).

relevant_pages lists the 1-indexed page numbers known to contain the answer;
recall@k and MRR are computed by treating each retrieved chunk's page number
as a chunk_id proxy and checking whether it falls in relevant_pages.
"""

EVAL_PAIRS = [
    {"question": "What is the top-5 error rate of the ensemble ResNet on the ImageNet test set?",
     "pdf_path": "papers/1512.03385v1.pdf", "relevant_pages": [1, 6]},
    {"question": "How many layers does the deepest ResNet evaluated on ImageNet have?",
     "pdf_path": "papers/1512.03385v1.pdf", "relevant_pages": [1]},
    {"question": "What top-1 and top-5 error does ResNet-152 achieve as a single model?",
     "pdf_path": "papers/1512.03385v1.pdf", "relevant_pages": [6]},
    {"question": "What classification error does the 110-layer ResNet achieve on CIFAR-10?",
     "pdf_path": "papers/1512.03385v1.pdf", "relevant_pages": [7]},

    {"question": "By how many times does LoRA reduce trainable parameters compared to GPT-3 175B fine-tuned with Adam?",
     "pdf_path": "papers/2106.09685v2.pdf", "relevant_pages": [1]},
    {"question": "By how many times does LoRA reduce GPU memory requirement compared to full fine-tuning?",
     "pdf_path": "papers/2106.09685v2.pdf", "relevant_pages": [1]},
    {"question": "How much does LoRA reduce VRAM usage from and to when training GPT-3 175B?",
     "pdf_path": "papers/2106.09685v2.pdf", "relevant_pages": [5]},
    {"question": "What WikiSQL and MNLI-m accuracy does GPT-3 LoRA achieve with 37.7M trainable parameters?",
     "pdf_path": "papers/2106.09685v2.pdf", "relevant_pages": [8]},

    {"question": "What non-parametric memory source does the RAG model use?",
     "pdf_path": "papers/2005.11401v4.pdf", "relevant_pages": [1, 2]},
    {"question": "What pre-trained seq2seq model is used as the generator in RAG?",
     "pdf_path": "papers/2005.11401v4.pdf", "relevant_pages": [1, 3]},
    {"question": "What score does RAG-Sequence achieve on the Natural Questions open-domain QA task?",
     "pdf_path": "papers/2005.11401v4.pdf", "relevant_pages": [5]},

    {"question": "What does the acronym BERT stand for?",
     "pdf_path": "papers/1810.04805v2.pdf", "relevant_pages": [1]},
    {"question": "What GLUE score does BERT achieve according to the abstract?",
     "pdf_path": "papers/1810.04805v2.pdf", "relevant_pages": [1]},
    {"question": "What are the layer count and hidden size of BERT-large?",
     "pdf_path": "papers/1810.04805v2.pdf", "relevant_pages": [4]},

    {"question": "What BLEU score does the Transformer big model achieve on WMT 2014 English-to-German?",
     "pdf_path": "papers/1706.03762v7.pdf", "relevant_pages": [1]},
    {"question": "What BLEU score does the Transformer big model achieve on WMT 2014 English-to-French?",
     "pdf_path": "papers/1706.03762v7.pdf", "relevant_pages": [1]},
    {"question": "How many parallel attention heads does the base Transformer model use?",
     "pdf_path": "papers/1706.03762v7.pdf", "relevant_pages": [5]},
    {"question": "What BLEU score does the Transformer base model achieve on English-to-German per Table 2?",
     "pdf_path": "papers/1706.03762v7.pdf", "relevant_pages": [8]},
]
