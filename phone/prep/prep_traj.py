"""Reference trajectory: STEPS AdamW steps in PyTorch on fixed batches that cover every length bucket.

The phone replays the same batches (traj.html) and compares loss and gradient norm step by step, which
checks every compiled shape and the optimizer state across steps, not just one step.
Writes ../web/data/traj.json.
"""

import json
import math
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, OUT = "jhu-clsp/ettin-encoder-17m", f"{HERE}/../web/data"
STEPS, BATCH, LR = 60, 32, 1e-4
meta = json.load(open(f"{OUT}/rows.json"))
lens = meta["train"]["lengths"]
bucket = lambda n: max(8, math.ceil(n / 8) * 8)

# batches: sort rows by length, cut into runs of 32, pick batches spread over every bucket
order = sorted(range(len(lens)), key=lambda i: lens[i])
all_b = [order[i:i + BATCH] for i in range(0, len(order) - BATCH + 1, BATCH)]
by_bucket = {}
for b in all_b:
    by_bucket.setdefault(bucket(max(lens[i] for i in b)), []).append(b)
rnd = random.Random(0)
batches = []
while len(batches) < STEPS:
    for k in sorted(by_bucket):
        if len(batches) < STEPS:
            batches.append(rnd.choice(by_bucket[k]))
rnd.shuffle(batches)

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(meta["labels"]),
                                                           attn_implementation="sdpa", dtype=torch.float32)
model.train()
tokens = open(f"{OUT}/tokens.bin", "rb").read()
import numpy as np
tokens = np.frombuffer(tokens, np.int32)
off = meta["train"]["offsets"]
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
out = {"batches": batches, "lr": LR, "loss": [], "grad_norm": [], "buckets": []}
for b in batches:
    ids = [tokens[off[i]:off[i] + lens[i]].tolist() for i in b]
    e = tok.pad({"input_ids": ids}, return_tensors="pt")
    y = torch.tensor([meta["train"]["y"][i] for i in b])
    loss = F.cross_entropy(model(**e).logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    out["loss"].append(float(loss)); out["grad_norm"].append(float(gn)); out["buckets"].append(bucket(max(lens[i] for i in b)))
json.dump(out, open(f"{OUT}/traj.json", "w"))
print("buckets", sorted(set(out["buckets"])))
print("loss", [round(x, 4) for x in out["loss"][:5]], "...", [round(x, 4) for x in out["loss"][-5:]])
