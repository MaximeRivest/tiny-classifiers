"""Download the ten public datasets and build one task folder each: data/tasks/<task>/

    task.json          {"name", "instruction", "labels"}: the instruction is one sentence, written once
                       as an analyst would, never tuned on results
    train.jsonl        2,000 random training messages (seed 0)       {"id", "text", "label"}
    train_big.jsonl    up to 150 per category (its first 2,000 rows are train.jsonl)
    test.jsonl         up to 1,000 random test messages (banking77: all 3,080)

    uv run recipe/prepare.py                 # all ten
    uv run recipe/prepare.py agnews trec     # some

Data comes from the Hugging Face Hub's Parquet export (datasets-server), downloaded once into data/raw/.
Each dataset keeps its own license; nothing here redistributes them.
"""
import glob
import json
import os
import random
import sys
import urllib.request
from collections import Counter

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RAW, OUT = os.path.join(ROOT, "raw"), os.path.join(ROOT, "tasks")
HUB = "https://datasets-server.huggingface.co"

# task: (Hub dataset, config, text column, label column, instruction)
TASKS = {
    "banking77": ("legacy-datasets/banking77", "default", "text", "label",
                  "Classify the bank customer's message by what they need."),
    "clinc": ("clinc/clinc_oos", "plus", "text", "intent",
              "Classify the user's request to a virtual assistant by its intent; answer out_of_scope if none fits."),
    "massive_en": ("SetFit/amazon_massive_intent_en-US", "default", "text", "label_text",
                   "Classify the user's command to a voice assistant by its intent."),
    "massive_fr": ("SetFit/amazon_massive_intent_fr-FR", "default", "text", "label_text",
                   "Classify the user's command to a voice assistant (in French) by its intent."),
    "trec": ("SetFit/TREC-QC", "default", "text", "label_text",
             "Classify the question by the type of answer it asks for."),
    "agnews": ("fancyzhx/ag_news", "default", "text", "label", "Classify the news article by its topic."),
    "sst5": ("SetFit/sst5", "default", "text", "label_text", "Classify the sentiment of this movie review excerpt."),
    "fpb": ("atrost/financial_phrasebank", "default", "sentence", "label",
            "Classify the sentiment of this sentence from financial news, from an investor's point of view."),
    "ledgar": ("coastalcph/lex_glue", "ledgar", "text", "label",
               "Classify the contract clause by its type (the heading of its section)."),
    "hate": ("cardiffnlp/tweet_eval", "hate", "text", "label",
             "Classify whether the tweet is hate speech against immigrants or women."),
}
RENAME = {"oos": "out_of_scope"}     # CLINC150's "none of these"
N_TRAIN, N_TEST, PER_LABEL = 2000, 1000, 150


def hub(path):
    return json.load(urllib.request.urlopen(f"{HUB}/{path}"))


def download(name):
    repo, config, *_ = TASKS[name]
    info = f"{RAW}/{name}_info.json"
    if not os.path.exists(info):
        json.dump(hub(f"info?dataset={repo}&config={config}"), open(info, "w"))
    for f in hub(f"parquet?dataset={repo}")["parquet_files"]:
        if f["config"] == config and f["split"] in ("train", "test"):
            path = f"{RAW}/{name}_{f['split']}_{f['filename'].rsplit('.', 1)[0].rsplit('-', 1)[-1]}.parquet"
            if not os.path.exists(path):
                urllib.request.urlretrieve(f["url"], path)


def build(name):
    _, _, tcol, lcol, instruction = TASKS[name]
    feat = json.load(open(f"{RAW}/{name}_info.json"))["dataset_info"]["features"][lcol]
    cls = feat.get("names") if isinstance(feat, dict) else None     # integer labels come with their names

    def rows(split):
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/{name}_{split}_*.parquet"))])
        out = []
        for i, r in enumerate(df.to_dict("records")):
            lab = cls[r[lcol]] if cls else r[lcol]
            out.append({"id": f"{name}-{split}-{i}", "text": str(r[tcol]).strip(), "label": RENAME.get(lab, lab)})
        return out

    tr, te = rows("train"), rows("test")
    labels = sorted({r["label"] for r in tr} | {r["label"] for r in te})
    random.Random(0).shuffle(tr)
    random.Random(0).shuffle(te)
    big = tr[:max(N_TRAIN, min(len(tr), PER_LABEL * len(labels)))]
    tr, te = tr[:N_TRAIN], (te if name == "banking77" else te[:N_TEST])
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    json.dump({"name": name, "instruction": instruction, "labels": labels}, open(f"{d}/task.json", "w"), indent=1)
    for split, rs in (("train", tr), ("test", te), ("train_big", big)):
        with open(f"{d}/{split}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    top = Counter(r["label"] for r in te).most_common(1)[0]
    print(f"{name:10s} {len(labels):3d} labels  train {len(tr):5d}  train_big {len(big):6d}  test {len(te):5d}  "
          f"most common test label {top[1] / len(te):.0%}")


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    for name in sys.argv[1:] or TASKS:
        download(name)
        build(name)
