# Results

Everything measured between 2026-09-24 and 2026-09-26, on one RTX 3090 unless said otherwise.
Students are fine-tuned with the recipe in [`recipe/train.py`](../recipe/train.py), never tuned per task.
Accuracy = share of test messages whose predicted label matches the dataset's human label.

## banking77: a 17M model against frontier AI

Bank-customer messages, 77 categories. Students trained on 9,493 human-labelled messages, 6 passes;
tested on the 3,076 test messages.

| model | how | % right | training | per message | cost per 1M messages |
|---|---|---|---|---|---|
| ModernBERT-large (395M) | fine-tuned | 94.2 | 150 s | | ≈ $0 |
| Qwen3.5 0.8B | fine-tuned | 93.8 | 229 s | | ≈ $0 |
| ModernBERT-base (150M) | fine-tuned | 93.2 | 78 s | 14 ms | ≈ $0 |
| Ettin 68M | fine-tuned | 93.2 | 113 s | | ≈ $0 |
| Ettin 32M | fine-tuned | 92.5 | 60 s | | ≈ $0 |
| Claude Opus 5.5 | prompted, no training | 92 | – | 2.4 s | $8,120 |
| **Ettin 17M** | fine-tuned | **91.5** | **42 s** | **5 ms** | ≈ $0 |
| BERT-tiny (4M) | fine-tuned | 89.2 | 85 s | | ≈ $0 |
| Kimi K3 | prompted, no training | 82 (81.6 on all 3,076) | – | 0.5 s | $1,540 |
| Jev | classification API, zero-shot | 79.4 | – | 0.12 s | $43 |

Opus and Kimi: 200 of the test messages (±2–3 points) unless said otherwise. Per message: API models include the
network round trip. Fine-tuned students: mean of up to 3 runs. Charts:
[accuracy vs training time](charts/banking77_accuracy_vs_training_time.png).

**Same model, same recipe, other hardware** ([chart](charts/gpu_laptop_phone.png)):

| trained on | training time | % right | per message |
|---|---|---|---|
| RTX 3090 (2020 gaming GPU) | 42 s | 91.5 | 5 ms |
| Laptop CPU (2019 Intel i7, 4 cores) | 11 min | 91.0 | 14 ms |
| Phone GPU (Galaxy S21 Ultra, Chrome, WebGPU) | 21 min | 91.4 | 39 ms |

## More labels, better model; no curve has flattened

[`banking77_learning_curves.json`](banking77_learning_curves.json) · [chart](charts/banking77_learning_curves.png).
Each point: a model fine-tuned on that many human-labelled messages ([rows, % right, training seconds]).

| messages | 100 | 250 | 500 | 1,000 | 2,000 | 4,000 | 9,493 |
|---|---|---|---|---|---|---|---|
| Ettin 17M | 14.6 | 27.1 | 44.7 | 66.7 | 81.0 | 87.2 | 91.5 |
| ModernBERT-large | 23.8 | 36.9 | 62.6 | 79.4 | 88.8 | 91.7 | 94.2 |

## A student lands just under its teacher

The same banking77 messages, labelled by people or by Kimi K3 (which itself scores 81.6%). Ettin 17M, mean of
3 runs ([`banking77_human_vs_kimi_labels.txt`](banking77_human_vs_kimi_labels.txt)):

| messages | 100 | 250 | 500 | 1,000 | 2,000 | 4,000 | 9,493 |
|---|---|---|---|---|---|---|---|
| human labels | 14.6 | 27.1 | 44.7 | 66.7 | 81.0 | 87.2 | 91.5 |
| Kimi K3's labels | 14.8 | 25.9 | 43.2 | 61.3 | 72.2 | 77.0 | 79.5 |

Up to a few hundred examples, the teacher's labels are as good as a person's. Past that, the student learns the
teacher's mistakes too, and stops just under the teacher.
[This chart](charts/banking77_ai_vs_human_labels.png) shows the same with an ensemble pipeline of open models
(1,748 messages), for every model size.

## Answers or probabilities?

Ettin 17M trained on all banking77 messages twice, changing only the target: the teacher's top answer, or its
full 77-way probabilities. Mean of 3 runs, 3,076 test messages.

| | answer only | probabilities |
|---|---|---|
| % right | 77.2 | 77.3 |
| right answer in its top 3 | 88.4 | 91.4 |
| average confidence (actually right: ~77%) | 95% | 85% |
| calibration error (lower is better) | 0.183 | 0.077 |
| % right on its most confident half | 91.1 | 95.9 |

With plenty of data, probabilities do not buy accuracy (with 1,894 messages they add about 2 points). They buy
confidence you can trust: keep the answers the model is sure of, send the rest to a bigger model.

## Ten tasks, one recipe

[`ten_tasks_table.txt`](ten_tasks_table.txt) · [chart](charts/ten_tasks.png). Ettin 17M, mean of 3 runs, trained on
2,000 messages, or up to 150 per category ("big") where there are many categories. Test sets: 1,000 messages
(TREC 500, banking77 3,076), so ±1.5–2 points.

| task (categories) | Kimi K3 | Jev | Ettin, Kimi's labels | Ettin, human labels |
|---|---|---|---|---|
| LEDGAR, legal clauses (100) | 78.5 | 76.0 | 76.3 | 83.4 |
| banking77 (77) | 81.6 | 79.4 | 79.5 | 91.5 |
| MASSIVE English, assistant (60) | 86.4 | 82.3 | 80.4 | 85.0 |
| MASSIVE French, assistant (60) | 86.0 | 80.0 | 78.0 | 79.4 |
| AG News, topics (4) | 87.4 | 87.0 | 84.8 | 87.3 |
| Financial PhraseBank (3) | 81.5 | 78.6 | 75.3 | 79.3 |
| CLINC150, assistant (151) | 91.1 | 88.2 | 83.5 | 85.2 |
| TREC, question types (50) | 77.4 | 84.2 | 73.9 | 89.9 |
| Tweet hate speech (2) | 71.3 | 75.0 | 58.3 | 49.0 |
| SST-5, sentiment (5) | 53.8 | 59.3 | 39.4 | 36.2 |

- On human labels, Ettin beats Jev on 6 of 10 tasks.
- On Kimi's labels it ties Jev on 2 and is within 5 points on 5 more. It loses where Kimi itself is weaker than
  Jev (TREC, hate speech, SST-5).
- Hate speech: the test set differs from the training set (insults toward women are "hate" 68% of the time in
  training, 42% in the test set); on the dataset's own validation set, the human-label student scores 71%.

## When it pays

[`breakeven.json`](breakeven.json) · [chart](charts/ten_tasks_cost_time_breakeven.png) ·
`uv run recipe/breakeven.py`. Up front: Kimi K3's labels (measured time and cost) and the training on a rented
RTX 3090 at $0.12/hour. Against Jev at its measured price per task:

| task | labels | labelling | labels cost | training | break-even (money) | break-even (time) |
|---|---|---|---|---|---|---|
| banking77 | 9,493 | 28 s | $14.34 | 44 s | 333k messages | 62k |
| CLINC150 | 15,250 | 6.4 min | $28.71 | 85 s | 476k | 395k |
| MASSIVE English | 9,000 | 90 s | $8.86 | 40 s | 268k | 109k |
| MASSIVE French | 9,000 | 3.9 min | $8.70 | 41 s | 261k | 232k |
| TREC | 5,452 | 45 s | $4.29 | 42 s | 151k | 74k |
| AG News | 2,000 | 9 s | $1.04 | 43 s | 69k | 47k |
| SST-5 | 2,000 | 8 s | $0.77 | 42 s | 55k | 44k |
| Financial PhraseBank | 2,000 | 9 s | $0.85 | 43 s | 60k | 46k |
| LEDGAR | 15,000 | 6.4 min | $25.79 | 2.6 min | 504k | 534k |
| Tweet hate speech | 2,000 | 8 s | $0.86 | 42 s | 61k | 45k |

## Reproducing

`recipe/prepare.py` rebuilds the nine non-banking77 tasks byte for byte. banking77 in this repository is the
standard split (10,003 training, 3,080 test messages); the numbers above used 9,493 training messages (500 held
out for calibration) and 3,076 test messages, so expect a difference of a few tenths of a point
(`recipe/train.py data/tasks/banking77 --split train_big` gives 91.6% in 34 s on an RTX 3090).
