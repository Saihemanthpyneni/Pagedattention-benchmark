"""
Throughput and Memory vs Batch Size
"""

import torch, time, json
from vllm import LLM, SamplingParams

MODEL       = "facebook/opt-1.3b"
BATCH_SIZES = [1, 2, 4, 8, 16]
INPUT_LEN   = 128
OUTPUT_LEN  = 128

llm = LLM(model=MODEL, dtype="float16",
          gpu_memory_utilization=0.6,
          max_model_len=512)

prompt   = "The history of artificial intelligence " * 20
prompt   = " ".join(prompt.split()[:INPUT_LEN])
sampling = SamplingParams(max_tokens=OUTPUT_LEN, temperature=0)

results = []
for bs in BATCH_SIZES:
    prompts = [prompt] * bs
    llm.generate([prompt], SamplingParams(max_tokens=5))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    outputs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    r = {
        "system":     "vLLM",
        "batch_size": bs,
        "throughput": round(total_tokens / elapsed, 2),
        "latency_ms": round(elapsed / total_tokens * 1000, 3),
        "memory_gb":  round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    results.append(r)
    print(f"batch={bs}: {r['throughput']} tok/s | {r['memory_gb']} GB")

json.dump(results, open("results_vllm.json", "w"), indent=2)
print("Saved results_vllm.json")
