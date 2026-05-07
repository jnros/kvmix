#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# validate attention output error: orig vs quantized K/V per strategy
import sys
import numpy as np
from pathlib import Path

N_PROMPTS = 8
SEQ_LEN = 512
TOKENS = N_PROMPTS * SEQ_LEN
KV_HEADS = 2
HEAD_DIM = 64
SCALE = HEAD_DIM ** -0.5

OUTLIER_K = {(0,0),(0,1),(1,1),(2,0),(2,1),(8,0)}


def quant_global(x):
	"""x: [tokens, head_dim] -> dequant float32, one scale per head"""
	mx = np.max(np.abs(x))
	if mx == 0:
		return x.copy()
	s = 127.0 / mx
	return np.round(x * s).clip(-128, 127).astype(np.int8).astype(np.float32) / s


def quant_per_token(x):
	"""x: [tokens, head_dim] -> dequant float32, one scale per token"""
	mx = np.max(np.abs(x), axis=1, keepdims=True)
	mx = np.where(mx == 0, 1.0, mx)
	s = 127.0 / mx
	return np.round(x * s).clip(-128, 127).astype(np.int8).astype(np.float32) / s


def quantize(K, V, strategy, layer):
	"""K,V: [tokens, kv_heads, head_dim] float32 -> dequantized copies"""
	Kq, Vq = K.copy(), V.copy()
	for h in range(KV_HEADS):
		if strategy == 1:
			Kq[:, h, :] = quant_global(K[:, h, :])
			Vq[:, h, :] = quant_global(V[:, h, :])
		elif strategy == 2:
			Kq[:, h, :] = quant_per_token(K[:, h, :])
			Vq[:, h, :] = quant_per_token(V[:, h, :])
		elif strategy == 3:
			if (layer, h) not in OUTLIER_K:
				Kq[:, h, :] = quant_global(K[:, h, :])
			Vq[:, h, :] = quant_per_token(V[:, h, :])
	return Kq, Vq


def attn_output(Q, K, V):
	"""
	Q: [tokens, q_heads, head_dim]
	K,V: [tokens, kv_heads, head_dim]
	Returns: [tokens, q_heads, head_dim]
	Causal attention within each SEQ_LEN-token prompt block.
	GQA: kv head for q head qh = qh // group.
	"""
	n_q = Q.shape[1]
	group = n_q // KV_HEADS
	out = np.empty_like(Q)
	causal = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e9), k=1)

	for p in range(N_PROMPTS):
		s, e = p * SEQ_LEN, (p + 1) * SEQ_LEN
		Qp, Kp, Vp = Q[s:e], K[s:e], V[s:e]
		for qh in range(n_q):
			kh = qh // group
			scores = Qp[:, qh, :] @ Kp[:, kh, :].T * SCALE + causal
			scores -= scores.max(axis=-1, keepdims=True)
			w = np.exp(scores)
			w /= w.sum(axis=-1, keepdims=True)
			out[s:e, qh, :] = w @ Vp[:, kh, :]

	return out


def metrics(out_orig, out_quant):
	diff = out_orig - out_quant
	nmse = np.mean(diff**2) / (np.mean(out_orig**2) + 1e-12)
	num = np.sum(out_orig * out_quant, axis=-1)
	denom = (np.linalg.norm(out_orig, axis=-1) *
	         np.linalg.norm(out_quant, axis=-1) + 1e-12)
	return float(nmse), float(np.mean(num / denom))


def report(layer, out_orig, out_quant, csv_out):
	nmse, cos = metrics(out_orig, out_quant)
	print(f"layer {layer:02d}  nmse={nmse:.3e}  cos={cos:.6f}")
	if csv_out is not None:
		csv_out.write(f"{layer},all,{nmse:.6e},{cos:.6f}\n")
	for h in range(out_orig.shape[1]):
		h_nmse, h_cos = metrics(out_orig[:, h:h+1, :], out_quant[:, h:h+1, :])
		print(f"  head {h:2d}  nmse={h_nmse:.3e}  cos={h_cos:.6f}")
		if csv_out is not None:
			csv_out.write(f"{layer},{h},{h_nmse:.6e},{h_cos:.6f}\n")


def run_layer(data_dir, layer, strategy, csv_out):
	K = np.fromfile(data_dir / f"layer_{layer:02d}.k.bin",
	                dtype=np.float16).reshape(TOKENS, KV_HEADS, HEAD_DIM).astype(np.float32)
	V = np.fromfile(data_dir / f"layer_{layer:02d}.v.bin",
	                dtype=np.float16).reshape(TOKENS, KV_HEADS, HEAD_DIM).astype(np.float32)
	qraw = np.fromfile(data_dir / f"layer_{layer:02d}.q.bin", dtype=np.float16)
	n_q = qraw.size // (TOKENS * HEAD_DIM)
	Q = qraw.reshape(TOKENS, n_q, HEAD_DIM).astype(np.float32)

	Kq, Vq = quantize(K, V, strategy, layer)
	out_orig = attn_output(Q, K, V)
	out_quant = attn_output(Q, Kq, Vq)
	report(layer, out_orig, out_quant, csv_out)


def main():
	corpus     = sys.argv[1] if len(sys.argv) > 1 else "c4"
	strategy   = int(sys.argv[2]) if len(sys.argv) > 2 else 3
	layers_arg = sys.argv[3] if len(sys.argv) > 3 else "spot"
	csv_flag   = sys.argv[4] if len(sys.argv) > 4 else "false"

	data_dir = Path(__file__).parent / "data" / f"{corpus}_fp16"

	if layers_arg == "spot":
		layers = [0, 7, 8, 23]
	elif layers_arg == "all":
		layers = list(range(24))
	else:
		layers = [int(layers_arg)]

	csv_out = None
	if csv_flag == "true":
		csv_path = Path(__file__).parent / f"attn_{corpus}_s{strategy}.csv"
		csv_out = open(csv_path, "w")
		csv_out.write("layer,head,nmse,cos_sim\n")

	print(f"corpus={corpus} strategy={strategy} layers={layers}")
	for layer in layers:
		run_layer(data_dir, layer, strategy, csv_out)

	if csv_out is not None:
		csv_out.close()
		print(f"wrote {csv_path}")


if __name__ == "__main__":
	main()
