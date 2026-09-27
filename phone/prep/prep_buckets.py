"""Loss at the starting weights for one batch of 32 per length bucket (8..128), for bucket-by-bucket checks."""
import json, math, os
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/../web/data"
meta = json.load(open(f"{OUT}/rows.json")); lens = meta["train"]["lengths"]; off = meta["train"]["offsets"]
tokens = np.fromfile(f"{OUT}/tokens.bin", np.int32)
bucket = lambda n: max(8, math.ceil(n / 8) * 8)
order = sorted(range(len(lens)), key=lambda i: lens[i])
picks = {}
for i in range(0, len(order) - 31):                  # a batch whose longest row sets the bucket: 31 rows before + that row
    b = bucket(lens[order[i + 31]])
    if b not in picks: picks[b] = order[i:i + 32]
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-17m")
model = AutoModelForSequenceClassification.from_pretrained("jhu-clsp/ettin-encoder-17m", num_labels=77, attn_implementation="sdpa", dtype=torch.float32).eval()
res = []
for b, rows in sorted(picks.items()):
    e = tok.pad({"input_ids": [tokens[off[i]:off[i] + lens[i]].tolist() for i in rows]}, return_tensors="pt")
    with torch.no_grad():
        loss = F.cross_entropy(model(**e).logits, torch.tensor([meta["train"]["y"][i] for i in rows]))
    res.append({"bucket": b, "rows": rows, "loss": float(loss)})
json.dump(res, open(f"{OUT}/buckets.json", "w")); print([(r["bucket"], round(r["loss"], 4)) for r in res])
