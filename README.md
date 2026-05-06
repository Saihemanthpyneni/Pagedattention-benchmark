# PagedAttention Benchmark

This repository contains the code and results for our CAP6614 
(Efficient AI) course project at the University of Central Florida. 
We benchmarked vLLM against HuggingFace's standard generation pipeline 
to study the real-world impact of PagedAttention on LLM serving efficiency.

The paper we worked from is:
**"Efficient Memory Management for Large Language Model Serving with 
PagedAttention"** — Kwon et al., SOSP 2023

---

## What We Did

We deployed vLLM on a Google Colab T4 GPU and compared it against 
HuggingFace generate() under four different experimental conditions:

- Throughput at different batch sizes (1, 2, 4, 8, 16)
- GPU memory usage at each batch size
- Latency at different sequence lengths (64, 128, 256, 512 tokens)
- Block size ablation to find the optimal KV cache block size on our hardware

All experiments used OPT-1.3B as the base model — same model family 
as the original paper.

---

## Key Results

vLLM outperformed HuggingFace at every batch size. At batch size 16, 
vLLM achieved 1.42x higher throughput. The advantage grew consistently 
as batch size increased, which matches the paper's finding that 
PagedAttention's benefits are more pronounced under higher load.

| Batch Size | HF (tok/s) | vLLM (tok/s) | Speedup |
|------------|------------|--------------|---------|
| 1          | 54.08      | 68.86        | 1.27×   |
| 2          | 89.21      | 124.61       | 1.40×   |
| 4          | 217.72     | 258.87       | 1.19×   |
| 8          | 381.93     | 480.98       | 1.26×   |
| 16         | 581.91     | 827.51       | 1.42×   |

For the block size ablation, block size 8 gave the best throughput on 
our T4 GPU. Block size 4 caused an out-of-memory error and block size 
64 is not supported by vLLM on Volta-generation GPUs, so we tested 
8, 16, and 32.

---

## Hardware and Software

We ran everything on Google Colab with a T4 GPU (16 GB VRAM).

- Python 3.12
- PyTorch 2.6.0
- vLLM 0.8.5
- HuggingFace Transformers 4.51.0
- Model: facebook/opt-1.3b

---

## How to Run

### Step 1 — Set up environment

We recommend running on Google Colab with a T4 GPU since vLLM 
requires Linux and a CUDA-compatible GPU.

Open a new Colab notebook, set the runtime to T4 GPU, then install:

```bash
pip install vllm==0.8.5 transformers==4.51.0 matplotlib numpy
```

If you're running locally on Linux:

```bash
git clone https://github.com/YOUR_USERNAME/pagedattention-benchmark
cd pagedattention-benchmark
pip install -r requirements.txt
```

### Step 2 — Run experiments in order

Important: run vLLM benchmarks before HuggingFace in the same session 
to avoid GPU memory conflicts. Restart the runtime between experiments 
if you hit memory errors.

```bash
# Experiment 1 and 2 — Throughput and memory vs batch size
python benchmark_vllm.py
python benchmark_hf.py

# Experiment 3 — Latency vs sequence length
python benchmark_seqlen.py

# Experiment 4 — Block size ablation
python benchmark_blocksize.py

# Generate all charts
python plot_results.py
```

Each script saves its results as a JSON file. The plot script reads 
all four JSON files and generates all six figures as a single PNG.

---

---

## Notes

- Block size 4 is not usable on T4 — causes OOM during cache 
  initialization due to the large number of blocks required
- Block size 64 is not supported by vLLM's PagedAttention kernel 
  on Volta-generation GPUs
- vLLM shows higher peak memory than HuggingFace because it 
  pre-reserves a GPU memory pool at startup. This is by design — 
  it manages that pool efficiently through PagedAttention rather 
  than allocating dynamically per request
- Results are lower than the paper's reported 2-4x because we used 
  a smaller model (1.3B vs 13B+) on a smaller GPU (T4 vs A100). 
  The trend is consistent with the paper

---

## Repository Structure
pagedattention-benchmark/
│
├── README.md                  — this file
├── requirements.txt           — python dependencies
│
├── benchmark_hf.py            — HuggingFace baseline benchmark
├── benchmark_vllm.py          — vLLM benchmark
├── benchmark_seqlen.py        — latency vs sequence length
├── benchmark_blocksize.py     — block size ablation
├── plot_results.py            — generates all 6 charts
│
├── results_hf.json            — HuggingFace raw results
├── results_vllm.json          — vLLM raw results
├── results_seqlen.json        — sequence length results
├── results_blocksize.json     — block size ablation results
│
└── all_charts.png             — all 6 figures in one image

## Team

Varun Kumar Jasti  
Sai Hemanth Pyneni  
Darshan Reddy Velkur  
Pranay Polishetty
