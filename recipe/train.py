"""Fine-tune Ettin-17M (or any encoder) on a task's messages, test it, and save it.

    uv run recipe/train.py data/tasks/agnews                                   # human labels
    uv run recipe/train.py data/tasks/agnews --labels labels/agnews-train.jsonl  # a teacher's labels (label.py)
    uv run recipe/train.py data/tasks/banking77 --split train_big --save models/banking77
    uv run recipe/train.py data/tasks/agnews --device cpu                      # no GPU: ~10-40 min

The recipe, the same for every task and never tuned per task: all weights trained (no LoRA), batch 32,
AdamW lr 1e-4, weight decay 0.01, 5% warmup then cosine decay, gradient clipping at 1.0, cross-entropy.
Passes over the data: 6 for ~9,500 messages, more for fewer (same number of steps: 28 passes for 2,000).
Max length 128 tokens, or up to 512 when the texts are long. bf16 on a GPU, fp32 on a CPU.

Prints the accuracy on the test messages (human labels) and writes runs/<task>__<target>__s<seed>.json.
With --save DIR, the model loads back with transformers:
    pipeline("text-classification", model=DIR)("Where is my new card?")
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("task", help="a task folder: task.json, <split>.jsonl, test.jsonl")
ap.add_argument("--labels", default="human", help="'human' (the dataset's labels) or a JSONL of {id, pred} from label.py")
ap.add_argument("--split", default="train", help="train (2,000 messages) or train_big (up to 150 per category)")
ap.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
ap.add_argument("--epochs", type=int, help="default: max(6, round(6 * 9493 / messages))")
ap.add_argument("--maxlen", type=int, help="default: 128, or up to 512 if 5% of texts are longer")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto", help="auto, cuda, cpu")
ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: PyTorch's default)")
ap.add_argument("--save", help="save the trained model here")
a = ap.parse_args()
T0 = time.time()

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
random.seed(a.seed)

task = json.load(open(os.path.join(a.task, "task.json")))
LABELS = task["labels"]
train = [json.loads(l) for l in open(os.path.join(a.task, f"{a.split}.jsonl"))]
test = [json.loads(l) for l in open(os.path.join(a.task, "test.jsonl"))]
if a.labels == "human":
    target, dropped = "human", 0
    train = [{**r, "target": r["label"]} for r in train]
else:
    target = "teacher"
    pred = {}
    for l in open(a.labels):
        r = json.loads(l)
        pred[r["id"]] = r.get("pred")
    n0 = len(train)
    train = [{**r, "target": pred[r["id"]]} for r in train if pred.get(r["id"]) in LABELS]
    dropped = n0 - len(train)            # messages the teacher did not label (or not with a valid label)

tok = AutoTokenizer.from_pretrained(a.model)
if a.maxlen:
    MAXLEN = a.maxlen
else:
    lens = sorted(len(tok(r["text"])["input_ids"]) for r in train[:2000])
    p95 = lens[int(.95 * len(lens))]
    MAXLEN = 128 if p95 <= 128 else min(512, 64 * math.ceil(p95 / 64))
EPOCHS = a.epochs or max(6, round(6 * 9493 / len(train)))
BATCH, LR, WARMUP = 32, 1e-4, 0.05

ids = lambda rs: [tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"] for r in rs]
train_ids, test_ids = ids(train), ids(test)
y = torch.tensor([LABELS.index(r["target"]) for r in train])
gold = torch.tensor([LABELS.index(r["label"]) for r in test])

try:                                      # flash attention if installed (faster on a GPU), else PyTorch's SDPA
    import flash_attn  # noqa: F401
    attn = "flash_attention_2" if DEVICE == "cuda" else "sdpa"
except ImportError:
    attn = "sdpa"
model = AutoModelForSequenceClassification.from_pretrained(
    a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    attn_implementation=attn, dtype=torch.float32).to(DEVICE)


def batches(xs, shuffle, bs=BATCH):
    order = list(range(len(xs)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(xs), bs):
        b = order[i:i + bs]
        yield b, tok.pad({"input_ids": [xs[j] for j in b]}, return_tensors="pt").to(DEVICE)


def logits_of(enc):
    if DEVICE == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(**enc).logits.float()
    return model(**enc).logits


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(train) / BATCH)
warm = max(1, int(WARMUP * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
print(f"{task['name']}: {len(train)} messages ({target} labels{f', {dropped} unlabelled dropped' if dropped else ''}), "
      f"{len(LABELS)} categories, {EPOCHS} passes = {steps} steps, max {MAXLEN} tokens, {DEVICE}, {attn}", flush=True)

model.train()
t_train = time.time()
step = 0
for epoch in range(EPOCHS):
    for b, enc in batches(train_ids, True):
        loss = F.cross_entropy(logits_of(enc), y[b].to(DEVICE))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
    if DEVICE == "cpu" or epoch == EPOCHS - 1:
        print(f"  pass {epoch + 1}/{EPOCHS}  loss {loss.item():.4f}  {time.time() - t_train:.0f} s", flush=True)
if DEVICE == "cuda":
    torch.cuda.synchronize()
train_s = time.time() - t_train

model.eval()
t_pred = time.time()
with torch.no_grad():
    pred = torch.cat([logits_of(e).argmax(1).cpu() for _, e in batches(test_ids, False, 256)])
pred_s = time.time() - t_pred
acc = float((pred == gold).float().mean())
res = {"task": task["name"], "target": target, "labels_file": None if target == "human" else a.labels, "split": a.split,
       "seed": a.seed, "model": a.model, "device": DEVICE, "train_rows": len(train), "dropped": dropped,
       "epochs": EPOCHS, "maxlen": MAXLEN, "test_n": len(test), "accuracy": round(acc, 4),
       "train_s": round(train_s, 1), "test_messages_per_s": round(len(test) / pred_s), "total_s": round(time.time() - T0, 1)}
os.makedirs("runs", exist_ok=True)
json.dump(res, open(f"runs/{task['name']}__{target}__{a.split}__s{a.seed}__{DEVICE}.json", "w"), indent=1)
print(f"accuracy {acc:.1%} on {len(test)} test messages; training {train_s:.0f} s", flush=True)
if a.save:
    model.save_pretrained(a.save)
    tok.save_pretrained(a.save)
    print(f"saved to {a.save}")
