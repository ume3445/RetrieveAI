# Retrieval eval results

100 questions over 5 papers, top_k=5, each query searches only its own paper. A hit means a retrieved chunk contains the question's gold answer span (whitespace/ligature-insensitive). See the README for how the questions were sampled.

Baseline: `d89162d` (original pipeline). After: `45a264c`.

| Mode | Hit@1 | Hit@5 | MRR@5 | | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| | **baseline** | | | | **after fixes** | | |
| Random (expected) | 0.01 | 0.04 | 0.02 | | 0.01 | 0.04 | 0.02 |
| BM25 only | 0.57 | 0.89 | 0.68 | | 0.55 | 0.88 | 0.67 |
| Dense only | 0.56 | 0.84 | 0.66 | | 0.57 | 0.80 | 0.65 |
| Hybrid | 0.60 | 0.91 | 0.71 | | 0.62 | 0.93 | 0.74 |
| **App path** (what the LLM sees) | 0.56 | 0.84 | 0.66 | | 0.62 | 0.93 | 0.74 |

## Change on the app path and the hybrid retriever (paired bootstrap, 10k resamples)

- App path (what the LLM sees): Hit@5 0.84 -> 0.93 (diff +0.09, 95% CI +0.02 to +0.16)
- Hybrid: Hit@5 0.91 -> 0.93 (diff +0.02, 95% CI -0.03 to +0.07)
- BM25 only: Hit@5 0.89 -> 0.88 (diff -0.01, 95% CI -0.05 to +0.02)
- App path Hit@1: 0.56 -> 0.62 (diff +0.06, 95% CI -0.02 to +0.14)

## Index

| | chunks | max chars | over CHUNK_SIZE |
|---|---|---|---|
| baseline | 1051 | 825 | 50 |
| after | 1010 | 512 | 0 |

## Retrieval latency after fixes (query embedding excluded, cached)

| Mode | p50 ms | p95 ms |
|---|---|---|
| bm25 | 0.3 | 0.5 |
| dense | 6.0 | 6.3 |
| hybrid | 6.2 | 6.5 |
| app | 6.2 | 6.5 |

## Per paper, app path Hit@5

| Paper | n | baseline | after |
|---|---|---|---|
| 1512.03385v1.pdf | 17 | 0.88 | 0.94 |
| 1706.03762v7.pdf | 13 | 0.77 | 0.92 |
| 1810.04805v2.pdf | 21 | 0.90 | 0.95 |
| 2005.11401v4.pdf | 22 | 0.86 | 0.91 |
| 2106.09685v2.pdf | 27 | 0.78 | 0.93 |

## App-path misses after fixes (7)

- **q001** (1512.03385v1.pdf) In the detection setup, how many candidate boxes does the proposal stage produce?  
  gold: `region proposal network (RPN, generating 300 proposals)`  
  top-1: Our box refinement partially follows the iterative localization in [6]. In Faster R-CNN, the final output is a regressed box that is different from its proposal
- **q030** (1706.03762v7.pdf) Besides attention, what does every encoder and decoder layer contain, and how is it applied across positions?  
  gold: `applied to each position separately and identically`  
  top-1: This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in se
- **q032** (1810.04805v2.pdf) Which task do the authors use to compare fine-tuning against extracting fixed features?  
  gold: `by applying BERT to the CoNLL-2003`  
  top-1: 5.3 Feature-based Approach with BERT All of the BERT results presented so far have used the fine-tuning approach, where a simple classification layer is added t
- **q063** (2005.11401v4.pdf) What decoding difficulty arises for the sequence-level variant of the model?  
  gold: `hence we cannot solve it with a single beam search`  
  top-1: We use greedy decoding for QA as we did not find beam search improved results. For Open-MSMarco and Jeopardy question generation, we report test numbers using t
- **q073** (2005.11401v4.pdf) On which task do the authors show the model is more factual and specific than BART?  
  gold: `more factual and specific than BART for Jeopardy question generation`  
  top-1: Evaluators indicated that BART was more factual than RAG in only 7.1% of cases, while RAG was more factual in 42.7% of cases, and both RAG and BART were factual
- **q078** (2106.09685v2.pdf) How does LoRA compare to baselines in terms of trainable parameter count on the GLUE results?  
  gold: `outperforms several baselines with comparable or fewer trainable parameters`  
  top-1: We cite numbers from prior works whenever possible to maximize the number of baselines we compare with; they are in rows with an asterisk (*) in the first colum
- **q082** (2106.09685v2.pdf) What is the largest model LoRA is tested on?  
  gold: `we scale up to GPT-3 with 175 billion parameters`  
  top-1: 8 C ONCLUSION AND FUTURE WORK Fine-tuning enormous language models is prohibitively expensive in terms of the hardware required and the storage/switching cost f
