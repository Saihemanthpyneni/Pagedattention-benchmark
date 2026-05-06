"""
Throughput and Memory vs Batch Size
"""

import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL       = "facebook/opt-1.3b"
BATCH_SIZES = [1, 2, 4, 8, 16]
INPUT_LEN   = 128
OUTPUT_LEN  = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16).cuda()
model.eval()

prompt = "The history of artificial intelligence " * 20
prompt = " ".join(prompt.split()[:INPUT_LEN])

results = []
for bs in BATCH_SIZES:
    prompts = [prompt] * bs
    inputs  = tokenizer(prompts, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=INPUT_LEN).to("cuda")

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=OUTPUT_LEN,
                       do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = bs * OUTPUT_LEN
    r = {
        "system":      "HuggingFace",
        "batch_size":  bs,
        "throughput":  round(total_tokens / elapsed, 2),
        "latency_ms":  round(elapsed / total_tokens * 1000, 3),
        "memory_gb":   round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    results.append(r)
    print(f"batch={bs}: {r['throughput']} tok/s | {r['memory_gb']} GB")

json.dump(results, open("results_hf.json", "w"), indent=2)
print("Saved results_hf.json")
