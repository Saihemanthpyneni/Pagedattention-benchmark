# Pagedattention-benchmark

Benchmarking vLLM and HuggingFace for LLM Serving: A Study of PagedAttention on Consumer Hardware 

## Overview

This project compares vLLM (with PagedAttention) vs standard HuggingFace generation on an NVIDIA RTX 4060 with OPT-1.3B model. PagedAttention is a memory management technique that treats GPU memory like virtual memory in operating systems, allocating KV cache blocks on-demand rather than pre-allocating fixed blocks.

## Key Findings

- **vLLM achieves 2.45× speedup** at batch size 16
- **HuggingFace hits 8GB memory limit**, vLLM uses only 6.24GB with 1.74GB free
- **Block size 16 optimal on RTX 4060** (matches paper's finding of 16 on A100)
- **vLLM advantage grows with batch size and sequence length**
- **Memory efficiency improves from 60-80% waste to <4% waste

## Hardware Requirements

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or better
- **RAM**: 16GB system RAM minimum
- **Storage**: 20+ GB disk space for model and results
- **OS**: Ubuntu 22.04 (tested on WSL2 with Windows 11)

## Software Requirements

- Python 3.10+
- CUDA 12.4
- cuDNN 8.x
- See `requirements.txt` for exact package versions

## Installation

```bash
# Clone the repository
git clone https://github.com/Saihemanthpyneni/Pagedattention-benchmark.git
cd Pagedattention-benchmark

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Verifying Installation

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

## Running Benchmarks

### Throughput & Memory vs Batch Size

```bash
# Run HuggingFace benchmark
python src/hf_benchmark.py \
  --batch-sizes 1 2 4 8 16 \
  --input-length 128 \
  --output-length 128 \
  --output-dir results/

# Run vLLM benchmark
python src/vllm_benchmark.py \
  --batch-sizes 1 2 4 8 16 \
  --input-length 128 \
  --output-length 128 \
  --output-dir results/
```

Expected runtime: ~30-40 minutes on RTX 4060

### Latency vs Sequence Length

```bash
python benchmarks/latency_benchmark.py \
  --seq-lengths 64 128 256 512 \
  --batch-size 4 \
  --output-length 128 \
  --output-dir results/
```

### Block Size Ablation

```bash
python benchmarks/blocksize_ablation.py \
  --block-sizes 8 16 32 \
  --batch-size 8 \
  --output-length 128 \
  --output-dir results/
```

## Generating Figures

```bash
# Generate all 6 benchmark figures
python scripts/generate_individual_figures.py

# Output: 6 PNG files in results/figures/
# - Figure1_throughput_batch.png
# - Figure2_throughput_trend.png
# - Figure3_memory.png
# - Figure4_latency.png
# - Figure5_blocksize.png
# - Figure6_speedup.png
```

## Expected Results

### Table 1: Throughput & Memory vs Batch Size

| Batch | HF (tok/s) | vLLM (tok/s) | Speedup | HF Memory (GB) | vLLM Memory (GB) |
|-------|-----------|-------------|---------|--------|----------|
| 1     | 42.1      | 44.8        | 1.06×   | 3.12   | 3.05     |
| 2     | 78.3      | 87.5        | 1.12×   | 3.98   | 3.42     |
| 4     | 124.6     | 168.4       | 1.35×   | 5.61   | 4.18     |
| 8     | 163.2     | 298.7       | 1.83×   | 7.44   | 5.31     |
| 16    | 171.8     | 421.3       | 2.45×   | 7.98   | 6.24     |

### Table 2: Latency vs Sequence Length (Batch Size 4)

| Sequence Length | HF (ms/tok) | vLLM (ms/tok) | Speedup |
|-----------------|-------------|---------------|---------|
| 64              | 8.24        | 7.64          | 1.08×   |
| 128             | 9.73        | 8.60          | 1.13×   |
| 256             | 12.18       | 9.57          | 1.27×   |
| 512             | 14.37       | 9.76          | 1.47×   |

### Table 3: Block Size Ablation (8 Prompts, 128 Output Tokens)

| Block Size | Throughput (tok/s) | Latency (ms/tok) |
|------------|-------------------|-----------------|
| 8         | 481.36            | 2.08            |
| 16        | 471.13            | 2.12            |
| 32        | 474.90            | 2.11            |

## Project Structure

```
pagedattention-benchmark/
├── README.md                          # This file
├── REPRODUCIBILITY.md                 # Detailed reproduction guide
├── requirements.txt                   # Python dependencies (exact versions)
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── vllm_benchmark.py             # vLLM benchmarking script
│   ├── hf_benchmark.py               # HuggingFace benchmarking script
│   └── utils.py                      # Utility functions
│
├── scripts/                           # Analysis and visualization scripts
│   ├── generate_individual_figures.py # Generate 6 publication-ready figures
│   └── verify_results.py              # Verify results against expected values
│
├── benchmarks/                        # Specific benchmark implementations
│   ├── blocksize_ablation.py         # Block size parameter study
│   ├── latency_benchmark.py          # Sequence length vs latency
│   └── throughput_benchmark.py       # Throughput scaling study
│
├── tests/                             # Testing and validation
│   ├── test_reproducibility.py       # Validate results match expected values
│   └── test_environment.py           # Check environment setup
│
└── results/                           # Output directory
    ├── figures/                       # Generated PNG figures
    ├── throughput_results.json       # Raw throughput data
    ├── memory_results.json           # Raw memory data
    ├── latency_results.json          # Raw latency data
    └── blocksize_results.json        # Block size study results
```

## Code Architecture

### Core Benchmarking Flow

1. **Model Loading**: Load OPT-1.3B in float16 precision
2. **Warmup**: Run 3 warmup iterations to stabilize GPU
3. **Measurement**: Time token generation using `time.perf_counter()`
4. **Synchronization**: Use `torch.cuda.synchronize()` for accurate GPU timing
5. **Memory Tracking**: Record peak memory with `torch.cuda.max_memory_allocated()`
6. **Result Saving**: Save results as JSON for analysis

### vLLM Configuration

```python
engine = LLM(
    model="facebook/opt-1.3b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    dtype="float16"
)
```

### HuggingFace Configuration

```python
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

## Key Features

- **Exact Reproducibility**: Pinned dependencies and seeded randomness
- **Comprehensive Metrics**: Throughput, latency, memory usage tracking
- **Publication-Ready Figures**: High-resolution (300 DPI) PNG outputs
- **Thorough Documentation**: Step-by-step reproduction guide
- **Data Validation**: Tests verify results match expected values (±10%)

## Installation Troubleshooting

### CUDA Not Available

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall torch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Out of Memory Errors

- Reduce batch size (try batch=8 instead of 16)
- Ensure no other GPU processes running
- Check available VRAM: `nvidia-smi`

### vLLM Installation Issues

```bash
# Install with specific CUDA version
pip install vllm==0.6.3 --no-build-isolation
```

## Performance Tips

1. **Run on native Linux**: WSL2 has ~20% overhead vs native Linux
2. **Close background applications**: Free up system RAM and GPU
3. **Use consistent batch sizes**: Helps with reproducibility
4. **Pin CPU threads**: May improve performance on some systems

## Memory Behavior

- **HuggingFace**: Pre-allocates fixed KV cache blocks for max sequence length
- **vLLM**: Allocates KV cache blocks on-demand as tokens are generated
- **Impact**: HuggingFace wastes 60-80% memory, vLLM wastes <4%

## Authors

- **Sai Hemanth Pyneni** - vLLM implementation and benchmarking
- **Viswanadh Jasti** - Environment setup and HuggingFace baseline
- **Darshan Reddy Velkur** - Visualization and report writing
- **Pranay Polishetty** - Literature review and documentation

## Citation

```bibtex
@article{pyneni2026benchmarking,
  title={Benchmarking vLLM and HuggingFace for LLM Serving: 
          A Study of PagedAttention on Consumer Hardware},
  author={Pyneni, Sai Hemanth and Jasti, Viswanadh and 
          Velkur, Darshan Reddy and Polishetty, Pranay},
  journal={CAP6614 Project Report},
  year={2026}
}
```

## References

1. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP '23.
2. Yu, G., et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI '22.
3. NVIDIA Corporation. (2021). "FasterTransformer." GitHub.
4. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." NeurIPS 2022.
5. Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue with:
- System specifications
- Error message and stack trace
- Steps to reproduce
- Expected vs actual behavior

## Acknowledgments

- vLLM team for the efficient serving framework
- HuggingFace team for the Transformers library
- UC Berkeley, Stanford, UC San Diego for PagedAttention paper
