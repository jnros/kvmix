## kvmix — KV cache int8 quantization benchmark

Benchmarks int8 quantization strategies on the KV cache of Qwen2.5-0.5B (fp16), using post-RoPE K and V tensors across 24 layers, 2 GQA heads, 4096 tokens.

Motivated by DeepSeek's mixed-precision KV cache work — exploring whether the same insight applies to an earlier open model.

### Build & run
```
make
./run.sh [corpus] [strategy] [csv_bool]   # e.g. ./run.sh c4 2 true
uv run plot_quant.py [corpus]             # line chart: norm MSE by layer
```

Corpora: `wikitext` (wikitext-2-raw-v1 test), `c4` (allenai/c4 en validation)

### Strategies
- **strat1** — global int8 scale per head (one scale over all 4096 tokens)
- **strat2** — per-token int8 scale per head
- **strat3** — mixed: fp16 for outlier K heads, int8 elsewhere

### Key findings

**The cast bug.** `(int8_t) x` vs `(int8_t) nearbyintf(x)` — the first syntax costs ~4x MSE. Nearly universal in C/C++ quant code.

**V is ~50x more compressible than K.** Values quantize cleanly across all layers. Keys are noisier, especially at early layers.

**Layer 8, K head 0 is a hard outlier.** One head at one layer dominates K quantization error. The pattern is model-intrinsic — reproduced identically on both corpora.

**Mixed precision pays off.** strat3 keeps fp16 only for the 6 outlier (layer, head) pairs out of 48 total, cutting cache to 54.7% of fp16 size with near-zero error on the problem heads. Naive int8 (strat1) is 50% but blows up on the outliers.

### Cache size
| strategy | bytes/token | 4096-tok cache |
|---|---|---|
| fp16 baseline | 12,288 | 48.0 MB |
| strat1 (global int8) | 6,144 | 24.0 MB |
| strat2 (per-token int8) | 6,528 | 25.5 MB |
| strat3 (mixed precision) | 6,720 | 26.2 MB |
