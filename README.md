# tiny-classifiers

**Train your own 17-million-parameter text classifier in about a minute, with labels made by a big AI model,
and know when it beats an API.**

A 17M model (Ettin), fine-tuned in 42 seconds on one gaming GPU, sorts bank messages into 77 categories about
as well as Claude Opus (91.5% vs 92%), for about $0 instead of $8,120 per million messages, at 5 ms a message.
It also trains on a laptop CPU (11 min) and on a phone's GPU (21 min).

This repository has everything behind those numbers: the recipe, a step-by-step tutorial, the measured results
on ten public tasks, the phone trainer, and the source of the explainer video.

![banking77: accuracy vs training time](results/charts/banking77_accuracy_vs_training_time.png)

## What we found

- **The model is the easy part; the labels are the whole game.** More labelled examples keep helping, for every
  model size we tried ([learning curves](results/README.md#more-labels-better-model-no-curve-has-flattened)).
- **No labels? Borrow a teacher.** A big model (Kimi K3) labels your messages: 2,000 in about 20 seconds for
  about $1. The student trained on them lands **a few points under its teacher**; human labels break that ceiling.
- **Ten tasks, one recipe.** On human labels, Ettin 17M beats Jev (a zero-shot classification API) on 6 of 10
  tasks. On Kimi's labels, it ties Jev on 2 and is within 5 points on 5 more; it loses where the teacher itself
  is weaker ([table](results/README.md#ten-tasks-one-recipe)).
- **When it pays:** against Jev's price, fine-tuning pays for itself after roughly **50,000 to 500,000 messages**
  ([per task](results/README.md#when-it-pays)). Below that, or before you know you need scale, use an API.
- **Evaluate first.** To trust any model in production, you need a few hundred messages with checked answers.
  Those are also most of what you need to train your own.

## Start here

**The tutorial** — [`tutorial/distill-ag-news.md`](tutorial/distill-ag-news.md): a runnable notebook, top to
bottom in about 10 minutes and $1.50 of API credit. Tidy ten datasets, label AG News with two teachers
(Kimi K3 and Qwen3.8-27B), fine-tune Ettin-17M on each set of labels and on the human labels, compare, and use
the model. Our run: human labels 89.2%, Kimi's 85.1%, Qwen's 81.8%; 6 ms per message on a CPU.

**The recipe**, four scripts ([uv](https://docs.astral.sh/uv/) installs everything on first run):

```sh
uv run recipe/prepare.py                         # download the 10 public datasets into data/tasks/<task>/
export OPENROUTER_API_KEY=...                    # and/or FIREWORKS_, TOGETHER_, DEEPINFRA_, PARASAIL_API_KEY
uv run recipe/label.py data/tasks/agnews         # Kimi K3 labels the 2,000 training messages (~20 s, ~$1)
uv run recipe/train.py data/tasks/agnews --labels labels/agnews-train.jsonl --save models/agnews
uv run recipe/train.py data/tasks/agnews         # the same on the human labels, for comparison
uv run recipe/breakeven.py                       # when fine-tuning pays, per task
```

```python
from transformers import pipeline
classify = pipeline("text-classification", model="models/agnews", device="cpu")
classify("NASA's new telescope finds water vapour on a distant planet")   # [{'label': 'Sci/Tech', ...}]
```

| script | what it does |
|---|---|
| [`recipe/prepare.py`](recipe/prepare.py) | Downloads banking77, CLINC150, MASSIVE (EN, FR), TREC, AG News, SST-5, Financial PhraseBank, LEDGAR and tweet hate speech from the Hugging Face Hub; builds `task.json` (one instruction sentence + the labels), 2,000 training messages, up to 150 per category (`train_big`), and the test set. |
| [`recipe/label.py`](recipe/label.py) | Labels messages with Kimi K3 across several providers at once, as fast as each allows (adaptive concurrency, retries, hedging of slow requests, a budget per provider). Resumable. Reports time, cost and, on the test set, the teacher's accuracy. |
| [`recipe/train.py`](recipe/train.py) | Fine-tunes all weights of Ettin-17M (or any encoder, `--model`) on human or teacher labels, on a GPU or a CPU; prints test accuracy; saves a standard Hugging Face model. |
| [`recipe/breakeven.py`](recipe/breakeven.py) | After how many messages the upfront cost (labels + training) is paid back, in money and in time. |

**Your own data:** make a folder like `data/tasks/agnews/`: a `task.json` with `{"name", "instruction",
"labels"}`, and `train.jsonl` / `test.jsonl` with one `{"id", "text", "label"}` per line (`train.jsonl` can
leave `label` out when a teacher labels it). Check a few hundred test messages by hand: that is what every
comparison stands on.

## The recipe, in one paragraph

[Ettin-17M](https://huggingface.co/jhu-clsp/ettin-encoder-17m) is an open encoder from Johns Hopkins (a cousin of
BERT, 17M parameters, most of them its vocabulary). It reads the whole message at once and gives one score per
category. We add one output per category and train every weight: batch 32, AdamW with learning rate 1e-4 and
weight decay 0.01, 5% warm-up then cosine decay, gradient clipping at 1.0, cross-entropy; 6 passes over ~9,500
messages, or the same number of steps for fewer (28 passes over 2,000). Nothing is tuned per task.

## Also here

- [`results/`](results/README.md): every number above, how it was measured, and the charts.
- [`phone/`](phone/README.md): the same training on a phone's GPU, in Chrome, with WebGPU and jax-js.
- [`video/`](video/README.md): the source of the explainer video (script, animated charts, music and
  captions made from code; voice by Kokoro on a local GPU).

## Good to know

- **Check the teacher's terms.** Train only on answers from a model whose license and provider terms allow it.
  We used Kimi K3; the answers of some models and APIs may not be used to train other models.
- **Numbers vary a little between runs** (±0.5 points from the random seed; ±1.5–2 from test-set size). The
  tables report means of 3 runs.
- **Datasets keep their own licenses**; this repository downloads them, it does not redistribute them.
- **NixOS:** if training fails with `/sbin/ldconfig` not found, set `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`.

MIT license. By [Maxime Rivest](https://x.com/MaximeRivest).
