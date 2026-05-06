"""
vLLM vs HuggingFace at varying sequence lengths
"""

import torch, time, json, gc
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL       = "facebook/opt-1.3b"
SEQ_LENGTHS = [64, 128, 256, 512]
BATCH       = 4
OUTPUT_LEN  = 128
base_prompt = "The history of artificial intelligence " * 30
results     = []

# vLLM
llm      = LLM(model=MODEL, dtype="float16",
               gpu_memory_utilization=0.5,
               max_model_len=768, enforce_eager=True)
sampling = SamplingParams(max_tokens=OUTPUT_LEN, temperature=0)

for sl in SEQ_LENGTHS:
    prompt  = " ".join(base_prompt.split()[:sl])
    prompts = [prompt] * BATCH
    llm.generate([prompt], SamplingParams(max_tokens=5))
    torch.cuda.synchronize()
    start   = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total   = sum(len(o.outputs[0].token_ids) for o in outputs)
    results.append({"system":"vLLM","seq_len":sl,
                    "latency_ms":round(elapsed/total*1000,3)})
    print(f"vLLM seq={sl}: {results[-1]['latency_ms']} ms/tok")

del llm
gc.collect()
torch.cuda.empty_cache()

# HuggingFace
tokenizer = AutoTokenizer.from_pretrained(MODEL)
hf_model  = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16).cuda()
hf_model.eval()

for sl in SEQ_LENGTHS:
    prompt  = " ".join(base_prompt.split()[:sl])
    prompts = [prompt] * BATCH
    inputs  = tokenizer(prompts, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=sl).to("cuda")
    with torch.no_grad():
        hf_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        hf_model.generate(**inputs, max_new_tokens=OUTPUT_LEN,
                          do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total   = BATCH * OUTPUT_LEN
    results.append({"system":"HuggingFace","seq_len":sl,
                    "latency_ms":round(elapsed/total*1000,3)})
    print(f"HF   seq={sl}: {results[-1]['latency_ms']} ms/tok")

json.dump(results, open("results_seqlen.json","w"), indent=2)
print("Saved results_seqlen.json")
