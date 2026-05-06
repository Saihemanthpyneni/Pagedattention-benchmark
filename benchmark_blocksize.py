"""
vLLM throughput at block sizes 8, 16, 32
"""

import torch, time, json, gc
from vllm import LLM, SamplingParams

MODEL       = "facebook/opt-1.3b"
BLOCK_SIZES = [8, 16, 32]
PROMPTS     = ["The history of artificial intelligence dates back to"] * 8
SAMPLING    = SamplingParams(max_tokens=128, temperature=0)

results = []
for bs in BLOCK_SIZES:
    print(f"\nTesting block size = {bs}")
    llm = LLM(model=MODEL, dtype="float16",
              gpu_memory_utilization=0.6,
              block_size=bs, max_model_len=512)

    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=5))

    torch.cuda.synchronize()
    start   = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total = sum(len(o.outputs[0].token_ids) for o in outputs)
    r = {
        "block_size":  bs,
        "throughput":  round(total / elapsed, 2),
        "latency_ms":  round(elapsed / total * 1000, 3),
    }
    results.append(r)
    print(f"block={bs}: {r['throughput']} tok/s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

json.dump(results, open("results_blocksize.json", "w"), indent=2)
print("Saved results_blocksize.json")
