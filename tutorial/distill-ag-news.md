---
rat:
  project: .
  python:
    requires: ">=3.12,<3.13"
    dependencies:
      - dpyr>=1.8.1
      - lm15>=1.1
      - polars
      - pyarrow
      - torch
      - transformers>=5
---
# Distill a frontier LLM into a 17M classifier, in a few minutes

This notebook turns a classification task into a tiny, fast, private model:

1. **Tidy the data**: every dataset becomes one Parquet table, one row per message.
2. **Label with LLMs**: two teachers label the messages through [lm15](https://github.com/lm15-dev/lm15-python):
   **Kimi K3** (a 2.8T-parameter frontier model, on Fireworks) and **Qwen3.8-27B** (a mid-size open model, on Parasail).
3. **Fine-tune** Ettin-17M, a 17-million-parameter encoder, on each teacher's labels (and on the human labels, for reference).
4. **Compare** accuracy, cost and time.

The tutorial uses **AG News** (news articles, 4 topics). Section 1 tidies nine more datasets the same way,
and changing `TASK` reruns everything on any of them.

**You need**
- `FIREWORKS_API_KEY` and `PARASAIL_API_KEY` in the environment (or in a `keys.env` file next to this notebook);
- an NVIDIA GPU for Section 4 (a 3090 trains Ettin-17M on 2,000 messages in about 45 s; a laptop CPU takes about 10 min);
- about **US$1.60** of API credit for the labels (Kimi K3 ~$1.40, Qwen3.8-27B ~$0.15).

Data processing uses [dpyr](https://github.com/MaximeRivest/dpyr) (dplyr's verbs for Python, on polars).

```python
import os, json, time, math, random, asyncio, urllib.request
from pathlib import Path
from dpyr import read, col, n, row_number, lit, if_else

DATA = Path("data")
(DATA / "raw").mkdir(parents=True, exist_ok=True)
(DATA / "tidy").mkdir(parents=True, exist_ok=True)

# API keys: from the environment, or from a keys.env file (KEY=value lines) next to this notebook.
KEY_FILE = Path("keys.env")
if KEY_FILE.exists():
    for line in KEY_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
print({k: ("set" if os.environ.get(k) else "MISSING") for k in ("FIREWORKS_API_KEY", "PARASAIL_API_KEY")})
```

## 1. Tidy data: one Parquet table per dataset

Each dataset is published on the Hugging Face Hub with its own column names and label encoding
(integers for some, strings for others). We make them all look the same:

| column | meaning |
|---|---|
| `task` | dataset name |
| `split` | `train` or `test` |
| `id` | stable message id, e.g. `ag_news-test-17` |
| `text` | the message |
| `label` | the human label, as a readable string |

We download each split once (reading remote Parquet directly works too, but the Hub rate-limits anonymous
requests quickly), then tidy it with dpyr.

```python
# task: (Hub dataset, config, text column, label column)
DATASETS = {
    "ag_news":              ("fancyzhx/ag_news",                   "default", "text",     "label"),
    "banking77":            ("legacy-datasets/banking77",          "default", "text",     "label"),
    "clinc150":             ("clinc/clinc_oos",                    "plus",    "text",     "intent"),
    "massive_en":           ("SetFit/amazon_massive_intent_en-US", "default", "text",     "label_text"),
    "massive_fr":           ("SetFit/amazon_massive_intent_fr-FR", "default", "text",     "label_text"),
    "trec":                 ("SetFit/TREC-QC",                     "default", "text",     "label_text"),
    "sst5":                 ("SetFit/sst5",                        "default", "text",     "label_text"),
    "financial_phrasebank": ("atrost/financial_phrasebank",        "default", "sentence", "label"),
    "ledgar":               ("coastalcph/lex_glue",                "ledgar",  "text",     "label"),
    "tweet_hate":           ("cardiffnlp/tweet_eval",              "hate",    "text",     "label"),
}
HUB = "https://datasets-server.huggingface.co"

def hub_json(path):
    return json.load(urllib.request.urlopen(f"{HUB}/{path}"))

def tidy(task):
    repo, config, text_col, label_col = DATASETS[task]
    files = [f for f in hub_json(f"parquet?dataset={repo}")["parquet_files"]
             if f["config"] == config and f["split"] in ("train", "test")]
    feature = hub_json(f"info?dataset={repo}&config={config}")["dataset_info"]["features"][label_col]
    names = feature.get("names")                      # integer labels come with their names
    out = DATA / "tidy" / task
    out.mkdir(exist_ok=True)
    for split in ("train", "test"):
        parts = []
        for f in sorted((f for f in files if f["split"] == split), key=lambda f: f["filename"]):
            raw = DATA / "raw" / f"{task}-{split}-{f['filename']}"
            if not raw.exists():
                urllib.request.urlretrieve(f["url"], raw)
            parts.append(str(raw))
        df = read(parts[0]) if len(parts) == 1 else read(str(DATA / "raw" / f"{task}-{split}-*.parquet"))
        df = df.rename(text=col[text_col], label_raw=col[label_col]).select(col.text, col.label_raw)
        if names:                                     # integer code -> readable name, with a join
            df = (df.left_join(read({"label_raw": list(range(len(names))), "label": names}), on=col.label_raw))
        else:
            df = df.mutate(label=col.label_raw)             # already strings
        (df.mutate(task=lit(task), split=lit(split), row=row_number() - 1,
                   label=if_else(col.label == "oos", "out_of_scope", col.label))   # CLINC's "none of these"
           .unite("id", [col.task, col.split, col.row], sep="-", remove=False)
           .select(col.task, col.split, col.id, col.text, col.label)
           .write(str(out / f"{split}.parquet")))
    return read(str(out / "*.parquet"))

for task in DATASETS:
    if not (DATA / "tidy" / task / "test.parquet").exists():
        tidy(task)
    t = read(str(DATA / "tidy" / task / "*.parquet"))
    print(f"{task:22s} {len(t):7,d} messages, {len(t.distinct(col.label)):4d} labels")
```

Every dataset now reads the same way. AG News:

```python
news = read("data/tidy/ag_news/*.parquet")
news.glimpse()
news.count(col.split, col.label)
```

## 2. The tutorial task: AG News

Four balanced topics; the articles are a headline plus one or two sentences.

```python
TASK = "ag_news"
INSTRUCTION = "Classify the news article by its topic."
N_TRAIN, N_TEST, SEED = 2000, 1000, 0

data = read(f"data/tidy/{TASK}/*.parquet")
LABELS = sorted(data.distinct(col.label).pull(col.label))
(data.mutate(chars=col.text.str_len())
     .group_by(col.split)
     .summarize(messages=n(), median_chars=col.chars.median(), max_chars=col.chars.max()))
```

We label and train on a random **2,000** training messages (what an analyst might afford to label) and
test on a random **1,000** test messages. With 4 topics that is 500 examples per topic, plenty for a small model.
Tasks with many categories (banking77: 77, CLINC150: 151) need more: aim for 100–150 examples per category.

```python
train = data.filter(col.split == "train").slice_sample(n=N_TRAIN, seed=SEED).persist()
test  = data.filter(col.split == "test").slice_sample(n=N_TEST, seed=SEED).persist()
print(LABELS)
train.count(col.label)
```

## 3. Label with two teachers, through lm15

lm15 speaks every provider's API through one `Request` type. Fireworks and Parasail are built in (lm15 1.1
and later): each reads its key from `FIREWORKS_API_KEY` / `PARASAIL_API_KEY`, and knows how to turn thinking off.

| teacher | provider | why this provider | price per million tokens (input / cached / output) |
|---|---|---|---|
| Kimi K3 (2.8T parameters) | Fireworks | the fastest and most reliable Kimi K3 host we measured | $3.00 / $0.30 / $15.00 |
| Qwen3.8-27B | Parasail | Fireworks and Together do not serve it; DeepInfra does, at a similar price | $0.24 / $0.05 / $2.20 |

```python
from lm15 import AsyncLMRouter, RouterConfig, Config, Message, Request, Reasoning, RETRYABLE_ERRORS
```

**The prompt.** The instruction and the list of labels go in the system message, the article in the user
message, and thinking is off (one fast answer, no reasoning trace).

**Why the list is in the prompt, and a JSON schema only for Qwen.** A JSON schema whose `enum` lists the
labels guarantees a valid answer, and lm15 hands it back parsed in `response.data`. We measured three
variants on 300 test articles:

| | Kimi K3 on Fireworks | Qwen3.8-27B on Parasail |
|---|---|---|
| list in the prompt + schema | 91.7%, 206 input tokens | 88.3%, 98 tokens |
| schema only | 93.0%, 188 tokens | **73.0%**: Parasail does not show the schema to the model |
| list in the prompt, no schema | 91.7%, **111 tokens**, 100% valid answers | 88.0%, 98 tokens |

So the list always goes in the prompt. The schema is free on Parasail and kept there; Fireworks writes it
into the prompt, **+85% input cost** for answers that were already valid, so Kimi answers in plain text and
we match that text to a label (anything that is not exactly one label is counted as "no label", not guessed).

```python
# name: lm15 model id, $ per million (input, cached input, output) tokens, use a schema?, requests in flight to start with
TEACHERS = {
    "kimi_k3":    dict(model="fireworks:accounts/fireworks/models/kimi-k3", price=(3.00, 0.30, 15.00), schema=False, start=256),
    "qwen38_27b": dict(model="parasail:parasail-qwen38-27b",                price=(0.24, 0.05,  2.20), schema=True,  start=32),
}
SYSTEM = f"{INSTRUCTION} Answer with exactly one label from this list:\n" + "\n".join(LABELS)
SCHEMA = {"type": "json_schema", "name": "label", "strict": True,
          "schema": {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}},
                     "required": ["label"], "additionalProperties": False}}

def config(teacher):
    return Config(max_tokens=30, temperature=0, reasoning=Reasoning(effort="off"),
                  response_format=SCHEMA if TEACHERS[teacher]["schema"] else None)

def answer(response):
    if response.data:                                       # schema: already parsed
        return response.data.get("label")
    text = (response.text or "").strip().strip("`*.\"' ")
    return {l.lower(): l for l in LABELS}.get(text.lower())  # exactly one label, or None

async def probe(text):
    async with AsyncLMRouter(config=RouterConfig()) as router:
        for name, t in TEACHERS.items():
            r = await router.complete(Request(model=t["model"], system=SYSTEM,
                                              messages=(Message.user(text),), config=config(name)))
            print(f"{name:11s} -> {answer(r)!r:12s} ({r.usage.input_tokens} tokens in, {r.usage.output_tokens} out)")

# asyncio.run() works in a script and in this kernel; in Jupyter (a loop already running) write `await probe(...)`.
asyncio.run(probe(test.pull(col.text)[0]))
```

**Many requests at once, safely.** A provider's capacity changes from minute to minute (the same Fireworks
account answered 340 requests/s one day and 65/s the next), so a fixed number of requests in flight is
either too timid or overloads it. `Throttle` adapts, the way TCP does: it halves the number in flight
whenever the provider answers "rate limited / overloaded", and adds one back after each full round of
successes. Every request has a 60 s timeout; lm15 never retries by itself, so retryable errors (rate limits,
timeouts, server errors) are retried here, with a growing random wait.

```python
class Throttle:
    """At most `limit` requests in flight: halved on overload, +1 after `limit` successes."""
    def __init__(self, start, ceiling=None):
        self.limit, self.ceiling, self.inflight, self.ok = float(start), ceiling or 4 * start, 0, 0
        self.cond, self.lowest = asyncio.Condition(), float(start)
    async def __aenter__(self):
        async with self.cond:
            await self.cond.wait_for(lambda: self.inflight < int(self.limit))
            self.inflight += 1
    async def __aexit__(self, *exc):
        async with self.cond:
            self.inflight -= 1
            self.cond.notify_all()
    def success(self):
        self.ok += 1
        if self.ok >= self.limit:
            self.ok, self.limit = 0, min(self.ceiling, self.limit + 1)
    def overloaded(self):
        self.limit = max(2.0, self.limit / 2)
        self.lowest = min(self.lowest, self.limit)

async def label_all(teacher, frame, attempts=8, timeout=60):
    t = TEACHERS[teacher]
    p_in, p_cached, p_out = t["price"]
    rows = frame.select(col.id, col.text).collect().to_dicts()
    throttle, out, t0 = Throttle(t["start"]), [], time.monotonic()
    async with AsyncLMRouter(config=RouterConfig()) as router:
        async def one(row):
            for attempt in range(attempts):
                try:
                    async with throttle:
                        r = await asyncio.wait_for(router.complete(Request(
                            model=t["model"], system=SYSTEM, messages=(Message.user(row["text"]),),
                            config=config(teacher))), timeout)
                    throttle.success()
                    u, cached = r.usage, r.usage.cache_read_tokens or 0
                    out.append({"id": row["id"], "pred": answer(r),
                                "usd": ((u.input_tokens - cached) * p_in + cached * p_cached + u.output_tokens * p_out) / 1e6})
                    if len(out) % 500 == 0:
                        print(f"  {teacher}: {len(out):5d}/{len(rows)}  {len(out) / (time.monotonic() - t0):4.0f}/s  "
                              f"{int(throttle.limit)} in flight", flush=True)
                    return
                except (*RETRYABLE_ERRORS, asyncio.TimeoutError):
                    throttle.overloaded()
                    await asyncio.sleep(min(30, 0.5 * 2 ** attempt) * random.uniform(0.5, 1.5))
            out.append({"id": row["id"], "pred": None, "usd": 0.0})   # gave up: counted, never hidden
        await asyncio.gather(*(one(row) for row in rows))
    seconds = time.monotonic() - t0
    labels = read({k: [o[k] for o in out] for k in ("id", "pred", "usd")})
    print(f"{teacher}: {len(out)} labels in {seconds:.1f} s ({len(out) / seconds:.0f}/s), "
          f"${sum(o['usd'] for o in out):.3f}, {sum(o['pred'] is None for o in out)} without a label, "
          f"in flight ended at {int(throttle.limit)} (lowest {int(throttle.lowest)})")
    return labels, seconds
```

Label the 2,000 training messages (what the students learn from) and the 1,000 test messages (to see how
good each teacher is). The two teachers run at the same time, on different providers.

```python
async def label_everything():
    jobs = {(teacher, part): label_all(teacher, frame)
            for teacher in TEACHERS for part, frame in (("train", train), ("test", test))}
    done = await asyncio.gather(*jobs.values())
    return dict(zip(jobs, done))

t0 = time.monotonic()
runs = asyncio.run(label_everything())
print(f"all labels: {time.monotonic() - t0:.1f} s wall")
```

How good is each teacher? Its answers on the test messages, against the human labels:

```python
def accuracy(pred_frame, truth=test):
    return (truth.select(col.id, col.label).inner_join(pred_frame, on=col.id)
                 .summarize(accuracy=if_else(col.pred == col.label, 1.0, 0.0).mean())
                 .pull(col.accuracy)[0])

rows = []
for t in TEACHERS:
    (train_labels, train_s), (test_labels, _) = runs[(t, "train")], runs[(t, "test")]
    rows.append({"teacher": t, "test_accuracy": accuracy(test_labels),
                 "train_labels_usd": sum(train_labels.pull(col.usd)), "train_labels_seconds": train_s})
teachers = read({k: [r[k] for r in rows] for k in rows[0]})
teachers
```

## 4. Fine-tune Ettin-17M on each set of labels

Three students, identical except for who labelled their 2,000 training messages: the humans who built
AG News, Kimi K3, and Qwen3.8-27B. Every weight is trained (no LoRA, no frozen layers), with the same recipe
used for all ten tasks of the benchmark: batch 32, AdamW (learning rate 1e-4, weight decay 0.01), 5% warm-up then
cosine decay, gradient clipping at 1.0, cross-entropy. The number of passes keeps the number of steps about the
same whatever the data size: 28 passes over 2,000 messages, 6 over 9,500.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STUDENT = "jhu-clsp/ettin-encoder-17m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(STUDENT)
train_rows = train.select(col.id, col.text, col.label).collect().to_dicts()
test_rows = test.select(col.id, col.text, col.label).collect().to_dicts()
test_texts, test_gold = [r["text"] for r in test_rows], [r["label"] for r in test_rows]

def fine_tune(texts, targets, seed=0, batch=32, lr=1e-4, maxlen=128):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    epochs = max(6, round(6 * 9493 / len(texts)))
    ids = [tok(t, truncation=True, max_length=maxlen)["input_ids"] for t in texts]
    y = torch.tensor([LABELS.index(t) for t in targets])
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(ids) / batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    model.train()
    t0 = time.monotonic()
    for _ in range(epochs):
        order = list(range(len(ids)))
        rng.shuffle(order)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            enc = tok.pad({"input_ids": [ids[j] for j in b]}, return_tensors="pt").to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(**enc).logits.float()
            loss = F.cross_entropy(logits, y[b].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return model.eval(), time.monotonic() - t0

@torch.no_grad()
def predict(model, texts, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=128, padding=True, return_tensors="pt").to(DEVICE)
        out += [LABELS[k] for k in model(**enc).logits.argmax(1).tolist()]
    return out
```

Train the three students (about 35 s each on an RTX 3090; about 10 minutes each on a laptop CPU). A message
the teacher did not label with a valid label is left out, never guessed.

```python
students, rows = {}, []
for source in ["human", *TEACHERS]:
    if source == "human":
        pairs = [(r["text"], r["label"]) for r in train_rows]
    else:
        pred = {o["id"]: o["pred"] for o in runs[(source, "train")][0].collect().to_dicts()}
        pairs = [(r["text"], pred[r["id"]]) for r in train_rows if pred.get(r["id"]) in LABELS]
    model, seconds = fine_tune([t for t, _ in pairs], [l for _, l in pairs])
    accuracy_ = sum(p == g for p, g in zip(predict(model, test_texts), test_gold)) / len(test_gold)
    students[source] = model
    rows.append({"labels_from": source, "messages": len(pairs), "student_accuracy": accuracy_, "train_seconds": seconds})
    print(f"{source:11s} {len(pairs)} messages -> {accuracy_:.1%} on the test set, trained in {seconds:.0f} s on {DEVICE}")
```

## 5. Compare

```python
report = (read({k: [r[k] for r in rows] for k in rows[0]})
          .left_join(teachers.rename(labels_from=col.teacher, teacher_accuracy=col.test_accuracy,
                                     labels_usd=col.train_labels_usd, labels_seconds=col.train_labels_seconds),
                     on=col.labels_from)
          .select(col.labels_from, col.teacher_accuracy, col.labels_usd, col.labels_seconds,
                  col.student_accuracy, col.train_seconds))
report
```

How to read it (our run: people 89.2%, Kimi K3 85.1%, Qwen3.8-27B 81.8%; the teachers themselves 88.3% and 85.0%):
- **The student lands a few points under its teacher** (about 3 here). It learns the teacher's mistakes along
  with its answers.
- **Human labels break that ceiling.** People labelled AG News and the test set is scored against those same
  people, so the student taught by humans ends up best, even above Kimi K3 itself.
- **The cheap teacher costs a tenth and loses about 3 points.** Qwen3.8-27B labelled the same 2,000 messages for
  $0.10 instead of $0.91. Whether 3 points are worth 80 cents depends on your task; for tasks with many
  fine-grained categories (banking77: 77, CLINC150: 151), the bigger teacher matters more.

## 6. Use the model

Save it, and it loads back like any Hugging Face model. It runs on a CPU: no GPU and no API needed.

```python
best = max(rows[1:], key=lambda r: r["student_accuracy"])["labels_from"]    # the best AI-taught student
path = f"models/{TASK}-{best}"
students[best].save_pretrained(path)
tok.save_pretrained(path)

from transformers import pipeline
classify = pipeline("text-classification", model=path, device="cpu")
print(classify("NASA's new telescope finds water vapour on a distant planet"))

t0 = time.monotonic()
for t in test_texts[:200]:
    classify(t)
print(f"{(time.monotonic() - t0) / 200 * 1000:.0f} ms per message, one at a time, on the CPU")
```

## 7. Your own task

- **Your data:** make a table with `id` and `text` (and `label` for a few hundred rows you checked by hand:
  that is your test set), set `TASK`, `INSTRUCTION` and `LABELS`, and run from Section 3.
- **Many categories?** Aim for 100–150 labelled messages per category, not 2,000 in total.
- **Check the teacher's terms first.** Train only on answers from a model whose license and provider allow it.
- **Compare before you ship.** Score the student, the teacher and any API you are considering on the same
  hand-checked test set. Fine-tune when it is better, or when your volume makes it cheaper
  (`recipe/breakeven.py` in this repository counts the messages it takes).
- **When your messages change,** label a few hundred new ones and train again: it takes a minute.
