import json, matplotlib.pyplot as plt, numpy as np

hf   = json.load(open("results_hf.json"))
vl   = json.load(open("results_vllm.json"))
blk  = json.load(open("results_blocksize.json"))
seq  = json.load(open("results_seqlen.json"))

bs      = [r["batch_size"] for r in hf]
hf_t    = [r["throughput"] for r in hf]
vl_t    = [r["throughput"] for r in vl]
hf_m    = [r["memory_gb"]  for r in hf]
vl_m    = [r["memory_gb"]  for r in vl]
speedup = [v/h for v,h in zip(vl_t,hf_t)]
vl_s    = [r for r in seq if r["system"]=="vLLM"]
hf_s    = [r for r in seq if r["system"]=="HuggingFace"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("vLLM vs HuggingFace — OPT-1.3B on Google Colab T4",
             fontsize=14, fontweight="bold")

x = np.arange(len(bs))

# Fig 1
ax = axes[0][0]
ax.bar(x-0.2, hf_t, 0.38, label="HuggingFace", color="#5B6FB5")
ax.bar(x+0.2, vl_t, 0.38, label="vLLM",        color="#E08030")
ax.set_xticks(x); ax.set_xticklabels(bs)
ax.set_xlabel("Batch Size"); ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Figure 1: Throughput vs Batch Size"); ax.legend()

# Fig 2
ax = axes[0][1]
ax.plot(bs, hf_t, "o-", color="#5B6FB5", lw=2, label="HuggingFace")
ax.plot(bs, vl_t, "s-", color="#E08030", lw=2, label="vLLM")
ax.set_xlabel("Batch Size"); ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Figure 2: Throughput Trend"); ax.legend()

# Fig 3
ax = axes[0][2]
ax.bar(x-0.2, hf_m, 0.38, label="HuggingFace", color="#5B6FB5")
ax.bar(x+0.2, vl_m, 0.38, label="vLLM",        color="#3A9B5C")
ax.axhline(y=15.6, color="#C94040", lw=1.5,
           linestyle="--", label="15.6 GB limit")
ax.set_xticks(x); ax.set_xticklabels(bs)
ax.set_xlabel("Batch Size"); ax.set_ylabel("Peak Memory (GB)")
ax.set_title("Figure 3: GPU Memory vs Batch Size"); ax.legend()

# Fig 4
ax = axes[1][0]
ax.plot([r["seq_len"] for r in vl_s],
        [r["latency_ms"] for r in vl_s],
        "s-", color="#E08030", lw=2, label="vLLM")
ax.plot([r["seq_len"] for r in hf_s],
        [r["latency_ms"] for r in hf_s],
        "o-", color="#5B6FB5", lw=2, label="HuggingFace")
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Latency (ms/token)")
ax.set_title("Figure 4: Latency vs Sequence Length"); ax.legend()

# Fig 5
ax = axes[1][1]
colors = ["#5B6FB5","#E08030","#5B6FB5"]
ax.bar([str(r["block_size"]) for r in blk],
       [r["throughput"] for r in blk], color=colors)
ax.set_xlabel("Block Size"); ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Figure 5: Block Size Ablation (vLLM)")

# Fig 6
ax = axes[1][2]
colors6 = ["#5B6FB5","#5B6FB5","#E08030","#E08030","#3A9B5C"]
ax.bar([str(b) for b in bs], speedup, color=colors6)
ax.axhline(y=1.0, color="#C94040", lw=1.5,
           linestyle="--", label="Baseline 1×")
ax.set_xlabel("Batch Size"); ax.set_ylabel("Speedup (vLLM / HF)")
ax.set_title("Figure 6: Speedup Ratio"); ax.legend()

plt.tight_layout()
plt.savefig("all_charts.png", dpi=180, bbox_inches="tight")
plt.show()
print("Saved all_charts.png")
