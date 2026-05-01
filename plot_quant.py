#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "matplotlib"]
# ///
import sys
import pandas as pd
import matplotlib.pyplot as plt

corpus = sys.argv[1] if len(sys.argv) > 1 else "wikitext"

cols = ["inputfile","layer","kv","ntokens","attnhead","ndims","strategy","max","total_mse","norm_mse"]
df = pd.concat([
	pd.read_csv(f"quant_{corpus}_s1.csv", header=None, names=cols),
	pd.read_csv(f"quant_{corpus}_s2.csv", header=None, names=cols),
	pd.read_csv(f"quant_{corpus}_s3.csv", header=None, names=cols),
], ignore_index=True)

grouped = df.groupby(["layer","kv","strategy"])["norm_mse"].mean().reset_index()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle(f"KV Cache Int8 Quant: Normalized MSE by Layer [{corpus}]\n(avg over 2 GQA heads)", fontsize=13)

for ax, kv_type in zip(axes, ["k", "v"]):
	sub = grouped[grouped["kv"] == kv_type]
	for strat, label, style in [(1, "strat1 (global)", "-o"), (2, "strat2 (per-token)", "-s"), (3, "strat3 (outlier skip)", "-^")]:
		s = sub[sub["strategy"] == strat].sort_values("layer")
		ax.plot(s["layer"], s["norm_mse"], style, label=label, markersize=4)
	ax.set_title(f"{'Keys' if kv_type=='k' else 'Values'}")
	ax.set_ylabel("Norm MSE")
	ax.legend()
	ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Layer")
plt.tight_layout()
out = f"kv_quant_{corpus}.png"
plt.savefig(out, dpi=150)
print(f"saved {out}")
