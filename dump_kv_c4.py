#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "datasets", "numpy", "accelerate"]
# ///
# dump post-RoPE K/V/Q cache from Qwen2.5-0.5B over C4 chunks
import os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

MODEL = "Qwen/Qwen2.5-0.5B"
N_PROMPTS = 8
SEQ_LEN = 512
OUT = Path(__file__).parent / "data" / "c4_fp16"

def main():
	torch.set_grad_enabled(False)
	OUT.mkdir(parents=True, exist_ok=True)

	tok = AutoTokenizer.from_pretrained(MODEL)
	model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
						     device_map="cuda")
	model.eval()
	cfg = model.config
	print(f"model: layers={cfg.num_hidden_layers} kv_heads={cfg.num_key_value_heads} "
	      f"head_dim={cfg.hidden_size // cfg.num_attention_heads}")

	n_q = cfg.num_attention_heads
	hd = cfg.hidden_size // n_q
	q_buf = {}
	hooks = []

	def make_q_hook(li):
		def hook(module, args, kwargs):
			hs = kwargs['hidden_states'] if 'hidden_states' in kwargs else (args[0] if args else None)
			pos_emb = kwargs.get('position_embeddings')
			if hs is None or pos_emb is None:
				return
			bsz, slen, _ = hs.shape
			cos, sin = pos_emb  # [batch, seq, head_dim] each
			q = module.q_proj(hs).view(bsz, slen, n_q, hd).transpose(1, 2)
			q_rot = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
			q_buf.setdefault(li, []).append(
				q_rot[0].permute(1, 0, 2).contiguous().cpu().numpy()
			)
		return hook

	for li in range(cfg.num_hidden_layers):
		hooks.append(model.model.layers[li].self_attn.register_forward_pre_hook(
			make_q_hook(li), with_kwargs=True
		))

	ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
	text = ""
	for ex in ds:
		text += " " + ex["text"].strip()
		if len(text) > 500_000:
			break
	ids_all = tok(text, return_tensors="pt").input_ids[0]
	print(f"c4 tokens: {ids_all.numel()}")

	need = N_PROMPTS * SEQ_LEN
	if ids_all.numel() < need:
		raise RuntimeError(f"need {need} tokens, got {ids_all.numel()}")
	chunks = torch.stack([ids_all[i*SEQ_LEN:(i+1)*SEQ_LEN] for i in range(N_PROMPTS)])
	input_ids = chunks.cuda()

	k_buf, v_buf = {}, {}
	for b in range(N_PROMPTS):
		out = model(input_ids[b:b+1], use_cache=True, return_dict=True)
		cache = out.past_key_values
		n_layers = len(cache)
		for li in range(n_layers):
			k = cache.layers[li].keys    # [1, kv_heads, seq, head_dim]
			v = cache.layers[li].values
			# -> [seq, kv_heads, head_dim]
			k = k[0].permute(1, 0, 2).contiguous().cpu().numpy()
			v = v[0].permute(1, 0, 2).contiguous().cpu().numpy()
			k_buf.setdefault(li, []).append(k)
			v_buf.setdefault(li, []).append(v)
		del out, cache
		torch.cuda.empty_cache()
		print(f"prompt {b+1}/{N_PROMPTS}")

	for h in hooks:
		h.remove()

	for li in sorted(k_buf):
		K = np.concatenate(k_buf[li], axis=0)  # [N*seq, kv_heads, head_dim]
		V = np.concatenate(v_buf[li], axis=0)
		K.astype(np.float16).tofile(OUT / f"layer_{li:02d}.k.bin")
		V.astype(np.float16).tofile(OUT / f"layer_{li:02d}.v.bin")

	for li in sorted(q_buf):
		Q = np.concatenate(q_buf[li], axis=0)  # [N*seq, n_q_heads, head_dim]
		Q.astype(np.float16).tofile(OUT / f"layer_{li:02d}.q.bin")
	print(f"saved {len(k_buf)} layers -> {OUT}")

	# validate
	k0 = np.fromfile(OUT / "layer_00.k.bin", dtype=np.float16)
	v0 = np.fromfile(OUT / "layer_00.v.bin", dtype=np.float16)
	print(f"layer_00.k: {k0.shape} -> reshape {k0.reshape(N_PROMPTS*SEQ_LEN, 2, 64).shape} {k0.dtype}")
	print(f"layer_00.v: {v0.shape} -> reshape {v0.reshape(N_PROMPTS*SEQ_LEN, 2, 64).shape} {v0.dtype}")
	q0 = np.fromfile(OUT / "layer_00.q.bin", dtype=np.float16)
	print(f"layer_00.q: {q0.shape} -> reshape {q0.reshape(N_PROMPTS*SEQ_LEN, n_q, hd).shape} {q0.dtype}")

if __name__ == "__main__":
	main()
