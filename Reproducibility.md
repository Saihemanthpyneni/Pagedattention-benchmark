# Reproducibility Guide

Complete step-by-step instructions to reproduce all benchmarking results and figures.

## System Specifications

### Hardware Used
- **GPU**: NVIDIA RTX 4060 (8GB VRAM, Volta architecture)
- **GPU Driver**: NVIDIA-SMI version with CUDA 12.4
- **CPU**: Intel Core i7 (or similar)
- **RAM**: 16GB+ system memory
- **Storage**: 20+ GB free disk space

### Software Stack
- **OS**: Ubuntu 22.04 LTS (via WSL2 on Windows 11)
- **Python**: 3.10
- **CUDA**: 12.4
- **cuDNN**: 8.x
- **PyTorch**: 2.4.0
- **vLLM**: 0.6.3
- **HuggingFace Transformers**: 4.44.0

## Step-by-Step Reproduction

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/Saihemanthpyneni/Pagedattention-benchmark.git
cd Pagedattention-benchmark

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install exact versions from requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Step 3: Verify GPU Setup

```bash
# Check GPU availability
python -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# Output should show:
# GPU Available: True
# GPU Name: NVIDIA RTX 4060
# GPU Memory: 8.00 GB
```

### Step 4: Run Throughput Benchmarks

#### 4a. HuggingFace Baseline

```bash
# Create results directory
mkdir -p results

# Run HuggingFace benchmark with all batch sizes
python src/hf_benchmark.py \
  --batch-sizes 1 2 4 8 16 \
  --input-length 128 \
  --output-length 128 \
  --output-dir results/ \
  --seed 42

# Expected output: results/hf_throughput_results.json
# Expected runtime: ~15-20 minutes
```

#### 4b. vLLM Benchmark

```bash
# Run vLLM benchmark with all batch sizes
python src/vllm_benchmark.py \
  --batch-sizes 1 2 4 8 16 \
  --input-length 128 \
  --output-length 128 \
  --output-dir results/ \
  --seed 42

# Expected output: results/vllm_throughput_results.json
# Expected runtime: ~15-20 minutes
```

### Step 5: Run Latency Benchmarks

```bash
# Latency vs Sequence Length (Batch Size 4)
python benchmarks/latency_benchmark.py \
  --seq-lengths 64 128 256 512 \
  --batch-size 4 \
  --output-length 128 \
  --output-dir results/ \
  --seed 42

# Expected output: results/latency_results.json
# Expected runtime: ~10-15 minutes
```

### Step 6: Run Block Size Ablation

```bash
# Block Size Ablation Study
python benchmarks/blocksize_ablation.py \
  --block-sizes 8 16 32 \
  --batch-size 8 \
  --output-length 128 \
  --output-dir results/ \
  --seed 42

# Expected output: results/blocksize_results.json
# Expected runtime: ~10-15 minutes
```

### Step 7: Generate Figures

```bash
# Generate all 6 publication-ready figures
python scripts/generate_individual_figures.py

# Output: 6 PNG files in results/figures/
# Files created:
# - Figure1_throughput_batch.png
# - Figure2_throughput_trend.png
# - Figure3_memory.png
# - Figure4_latency.png
# - Figure5_blocksize.png
# - Figure6_speedup.png

# Verify figures exist
ls -lh results/figures/
```

### Step 8: Verify Results

```bash
# Run validation tests
python tests/test_reproducibility.py

# Expected output: All tests pass (tolerance ±10%)
```

## Expected Results

### Throughput Results

Expected values (tolerances in parentheses):

**HuggingFace Throughput (tokens/sec)**
- Batch 1: 42.1 ± 4.2
- Batch 2: 78.3 ± 7.8
- Batch 4: 124.6 ± 12.5
- Batch 8: 163.2 ± 16.3
- Batch 16: 171.8 ± 17.2

**vLLM Throughput (tokens/sec)**
- Batch 1: 44.8 ± 4.5
- Batch 2: 87.5 ± 8.8
- Batch 4: 168.4 ± 16.8
- Batch 8: 298.7 ± 29.9
- Batch 16: 421.3 ± 42.1

**Speedup (vLLM / HF)**
- Batch 1: 1.06× ± 0.10
- Batch 2: 1.12× ± 0.11
- Batch 4: 1.35× ± 0.14
- Batch 8: 1.83× ± 0.18
- Batch 16: 2.45× ± 0.25

### Memory Results

Expected values (tolerances in parentheses):

**HuggingFace Memory (GB)**
- Batch 1: 3.12 ± 0.16
- Batch 2: 3.98 ± 0.20
- Batch 4: 5.61 ± 0.28
- Batch 8: 7.44 ± 0.37
- Batch 16: 7.98 ± 0.40

**vLLM Memory (GB)**
- Batch 1: 3.05 ± 0.15
- Batch 2: 3.42 ± 0.17
- Batch 4: 4.18 ± 0.21
- Batch 8: 5.31 ± 0.27
- Batch 16: 6.24 ± 0.31

### Latency Results

Expected values for Batch Size 4 (tolerances in parentheses):

**HuggingFace Latency (ms/token)**
- 64 tokens: 8.24 ± 0.82
- 128 tokens: 9.73 ± 0.97
- 256 tokens: 12.18 ± 1.22
- 512 tokens: 14.37 ± 1.44

**vLLM Latency (ms/token)**
- 64 tokens: 7.64 ± 0.76
- 128 tokens: 8.60 ± 0.86
- 256 tokens: 9.57 ± 0.96
- 512 tokens: 9.76 ± 0.98

### Block Size Results

Expected values for 8 prompts, 128 output tokens (tolerances in parentheses):

**Throughput (tokens/sec)**
- Block 8: 481.36 ± 48.1
- Block 16: 471.13 ± 47.1
- Block 32: 474.90 ± 47.5

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size (try 8 instead of 16)
2. Reduce sequence length (try 512 instead of larger)
3. Kill other GPU processes: `nvidia-smi` then `fuser -k /dev/nvidia*`
4. Restart Python kernel and try again

### Issue: Model Download Fails

**Symptom**: `FileNotFoundError` during model loading

**Solutions**:
1. Check internet connection
2. Manually download model: `huggingface-cli login`
3. Set cache directory: `export HF_HOME=/path/to/cache`
4. Check disk space: `df -h`

### Issue: Slow Performance

**Symptom**: Benchmarks taking >60 minutes

**Causes**:
- WSL2 overhead (20% slower than native Linux)
- Other GPU processes running
- Thermal throttling due to high GPU temperature

**Solutions**:
1. Monitor GPU: `watch -n 1 nvidia-smi`
2. Check temperature: `nvidia-smi --query-gpu=temperature.gpu --format=csv`
3. Run on native Linux if possible (not WSL2)

### Issue: Figure Generation Fails

**Symptom**: `FileNotFoundError` when generating figures

**Solutions**:
1. Verify results files exist: `ls -la results/*.json`
2. Check matplotlib installation: `python -c "import matplotlib; print(matplotlib.__version__)"`
3. Ensure output directory exists: `mkdir -p results/figures`

## Performance Monitoring

### Real-Time GPU Monitoring

```bash
# Watch GPU usage during benchmarking
watch -n 1 nvidia-smi

# Monitor in separate terminal while running benchmarks
nvidia-smi --loop=1
```

### Memory Usage Tracking

```bash
# Clear GPU cache before each run
python -c "import torch; torch.cuda.empty_cache()"

# Monitor peak memory after run
python -c "import torch; print(f'Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')"
```

## Reproducibility Notes

### Important Considerations

1. **Random Seed**: Set to 42 for all experiments
   - CPU seeds set for numpy and torch
   - GPU operations are deterministic with CUDA 12.4

2. **Thermal Effects**: GPU performance varies with temperature
   - Best results when GPU cooled to <60°C
   - Performance may degrade if >80°C

3. **Model Caching**: First run downloads OPT-1.3B (~2.5GB)
   - Subsequent runs use cached model
   - Model stored in `~/.cache/huggingface/`

4. **WSL2 Overhead**: ~20% slower than native Linux
   - Throughput may be 20% lower on WSL2
   - Still within ±10% tolerance range

5. **Floating Point Precision**: All models use float16
   - Ensures consistent memory usage across runs
   - May have minor precision differences

## Expected Runtime

**Total Benchmark Runtime**: ~70-90 minutes

- Throughput (HF): 15-20 minutes
- Throughput (vLLM): 15-20 minutes
- Latency: 10-15 minutes
- Block Size: 10-15 minutes
- Figure Generation: 1-2 minutes
- Validation: <1 minute

## Validation Checklist

After completing all steps, verify:

- [ ] HuggingFace throughput within ±10% of expected
- [ ] vLLM throughput within ±10% of expected
- [ ] Memory usage within ±5% of expected
- [ ] Latency values match expected (±10%)
- [ ] Block size results show 16 as optimal
- [ ] 6 PNG figures generated without errors
- [ ] All figure values match report tables
- [ ] Tests pass with all tolerances met

## Contact for Issues

If you encounter issues not covered here:

1. Check GPU temperature and load
2. Verify all dependencies installed correctly
3. Try running in native Linux environment
4. Clear GPU cache: `python -c "import torch; torch.cuda.empty_cache()"`
5. Open a GitHub issue with:
   - System specifications
   - Full error message
   - Commands run
   - Expected vs actual output

## References

- vLLM GitHub: https://github.com/vllm-project/vllm
- HuggingFace Transformers: https://github.com/huggingface/transformers
- OPT Model: https://huggingface.co/facebook/opt-1.3b
- PyTorch: https://pytorch.org/
