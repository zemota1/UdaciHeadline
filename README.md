# UdaciHeadline — LLM Inference Optimization

Generate short news headlines from article summaries with a small Llama model, then
measure how different inference optimizations change **speed, memory, and quality**.

## What's in this repo

| File | What it is |
|------|------------|
| `UdaciHeadline_Project_Corrected.ipynb` | The project notebook (all code and results). |
| `UdaciHeadline_Final_Report.pdf` | Short written report with the results and trade-offs. |
| `README.md` | This file. |

## The task

An ML engineer at a news portal needs to auto-generate catchy headlines from article
summaries. The model works, but inference is slow and costly. The goal is to speed it
up without hurting headline quality.

- **Model:** `meta-llama/Llama-3.2-1B-Instruct` (plus `Llama-3.2-3B-Instruct` for speculative decoding)
- **Data:** News Category Dataset v3, 100 evaluation examples
- **Decoding:** greedy, fixed at 24 new tokens so every run is compared fairly
- **Quality metric:** ROUGE-1 / ROUGE-2 / ROUGE-L vs. the real headlines

## Techniques compared

1. Baseline (no KV cache)
2. KV caching
3. Pruning (30%)
4. Quantization (4-bit)
5. Multi-GPU sharding (tensor + pipeline parallelism)
6. Speculative decoding (3B target, with the 1B model as draft)

## Results

Speed and memory (lower latency = better, higher throughput = better):

| Technique | Mean latency (s) | Tokens/s | Peak memory (MB) |
|---|---|---|---|
| Baseline (no cache) | 0.208 | 49.7 | 2492 |
| **KV caching** | **0.176** | **59.1** | 2495 |
| Pruning 30% | 0.167 | 58.6 | 2495 |
| Quantization 4-bit | 0.323 | 37.9 | **1091** |
| Tensor parallelism | 0.251 | 41.5 | 788 |
| Pipeline parallelism | 0.252 | 41.5 | 788 |
| Speculative (target alone) | 0.415 | 30.2 | 8934 |
| Speculative decoding | 0.541 | 23.3 | 8963 |

Quality stayed roughly flat across techniques (ROUGE-L ≈ 0.11), **except pruning**, which
dropped it noticeably (ROUGE-L 0.088).

## Takeaways

- **KV caching is the recommended default** — fastest practical option, no quality loss,
  no retraining, already built into the generation API.
- **Quantization (4-bit)** is the best choice when *memory* is the constraint (memory cut
  by more than half), at the cost of speed.
- **Pruning, multi-GPU sharding, and speculative decoding are not worth it here** — pruning
  hurt quality, and the others were slower because the 1B model is already small enough to
  fit comfortably on one GPU.

Full reasoning is in `UdaciHeadline_Final_Report.pdf`.

## How to run

The notebook was run on a CUDA GPU machine (the saved results used 4× Tesla T4, float16).

1. Open `UdaciHeadline_Project_Corrected.ipynb`.
2. Run the first cell to install dependencies (transformers, PyTorch, Accelerate,
   bitsandbytes, evaluate, rouge-score, etc.).
3. Set a Hugging Face token (the Llama models are gated):
   ```bash
   export HF_TOKEN=your_token_here
   ```
4. Make sure `News_Category_Dataset_v3.json` is available (from Kaggle).
5. Run all cells top to bottom. Set `EVAL_N = 5` near the top for a quick smoke test.

> A GPU is required for the published numbers. CPU runs work for checking the code but are
> too slow for meaningful benchmarks.
