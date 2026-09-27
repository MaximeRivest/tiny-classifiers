"""Prepare the phone run: tokenized data, starting weights, and a reference to check the phone's math.

banking77 from recipe/prepare.py (run it first), 500 training messages held out; the Ettin-17M tokenizer and
starting weights (seed S seeds the new classifier layer). Writes ../web/data/:

  tokens.bin    int32 token ids, all rows back to back (train then test)
  rows.json     per split: offsets, lengths, label ids; plus label names
  weights.bin   float32 parameters, Linear weights transposed to [in, out]
  weights.json  name -> {shape, offset}
  ref.json      PyTorch forward/backward/AdamW-step on a fixed batch, for checking

    python prep.py [--seed 0]
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, DATA, OUT = "jhu-clsp/ettin-encoder-17m", f"{HERE}/../../data/tasks/banking77", f"{HERE}/../web/data"
os.makedirs(OUT, exist_ok=True)
MAXLEN, VAL_ROWS, SPLIT_SEED = 128, 500, 0

# ---- data: identical to cpu_train.py ---------------------------------------------
# banking77 as built by recipe/prepare.py: all training messages, 500 of them held out (seed 0)
task = json.load(open(f"{DATA}/task.json"))
LABELS = task["labels"]
rows = [json.loads(line) for line in open(f"{DATA}/train_big.jsonl")]
test = [json.loads(line) for line in open(f"{DATA}/test.jsonl")]
pool = list(rows)
random.Random(SPLIT_SEED).shuffle(pool)
val_texts = {r["text"] for r in pool[:VAL_ROWS]}
train = [r for r in rows if r["text"] not in val_texts]

torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(MODEL)
enc = lambda rs: [tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"] for r in rs]
train_ids, test_ids = enc(train), enc(test)

flat, meta = [], {"labels": LABELS, "pad_id": tok.pad_token_id}
for name, ids, rs in (("train", train_ids, train), ("test", test_ids, test)):
    off = sum(len(x) for x in flat)
    offsets, n = [], off
    for x in ids:
        offsets.append(n)
        n += len(x)
    flat.extend(ids)
    meta[name] = {"offsets": offsets, "lengths": [len(x) for x in ids], "y": [LABELS.index(r["label"]) for r in rs]}
np.concatenate([np.asarray(x, np.int32) for x in flat]).tofile(f"{OUT}/tokens.bin")
json.dump(meta, open(f"{OUT}/rows.json", "w"))
lens = np.array([len(x) for x in train_ids])
print(f"train {len(train_ids)} rows, test {len(test_ids)}; tokens/row mean {lens.mean():.1f}, max {lens.max()}, "
      f">65: {(lens > 65).sum()}")

# ---- weights (after torch.manual_seed(seed), as in cpu_train.py) ------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS),
                                                           attn_implementation="sdpa", dtype=torch.float32)
model.train()
index, off, chunks = {}, 0, []
for name, p in model.named_parameters():
    t = p.detach().cpu().float()
    if t.ndim == 2 and "tok_embeddings" not in name:
        t = t.T.contiguous()                   # Linear [out, in] -> [in, out] for x @ W
    arr = t.numpy().astype(np.float32).ravel()
    index[name] = {"shape": list(t.shape), "offset": off}
    off += arr.size
    chunks.append(arr)
np.concatenate(chunks).tofile(f"{OUT}/weights.bin")
json.dump(index, open(f"{OUT}/weights.json", "w"), indent=0)
print(f"{len(index)} tensors, {off / 1e6:.2f}M parameters")

# ---- reference batch: 7 ordinary rows + the longest row (exercises the sliding window) ----
ref_rows = list(range(7)) + [int(lens.argmax())]
b_ids = [train_ids[i] for i in ref_rows]
e = tok.pad({"input_ids": b_ids}, return_tensors="pt")
y = torch.tensor([LABELS.index(train[i]["label"]) for i in ref_rows])
logits = model(**e).logits
loss = F.cross_entropy(logits, y)
loss.backward()
grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
gnorm = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
with torch.no_grad():
    logits_after = model(**e).logits


def sample(name):                              # grad values at a few fixed positions, in the exported layout
    g = grads[name]
    if g.ndim == 2 and "tok_embeddings" not in name:
        g = g.T.contiguous()
    return {"shape": list(g.shape), "norm": float(g.norm()), "first": g.ravel()[:6].tolist()}


ref = {
    "rows": ref_rows, "labels": y.tolist(), "lengths": [len(x) for x in b_ids],
    "logits": logits.detach().tolist(), "loss": float(loss), "grad_norm": gnorm,
    "grads": {n: sample(n) for n in ["classifier.weight", "classifier.bias", "head.dense.weight",
                                     "model.layers.1.attn.Wqkv.weight", "model.layers.0.attn.Wqkv.weight",
                                     "model.layers.4.mlp.Wi.weight", "model.embeddings.tok_embeddings.weight"]},
    "logits_after_adamw_step": logits_after.tolist(),
}
json.dump(ref, open(f"{OUT}/ref.json", "w"))
print(f"reference: loss {float(loss):.5f}, grad norm {gnorm:.4f}, lengths {ref['lengths']}")
