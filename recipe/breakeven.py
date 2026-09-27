"""After how many messages does fine-tuning your own model pay for itself, against a per-message API?

    uv run recipe/breakeven.py                                     # the measured numbers, ten tasks
    uv run recipe/breakeven.py --labels-usd 14.34 --labels-s 28 --train-s 46 --api-usd-per-million 43

Money: messages = (labels $ + training GPU $) / (API $ per message - own model $ per message)
Time:  messages = (labelling s + training s) / (API s per message - own model s per message)
       (assumes nothing is classified while you label and train: all the upfront time is lost)
Own model: a rented RTX 3090 at $0.12/hour, at the measured messages per second.
API speed: 800 messages/s (Jev, sustained, measured on banking77).
"""
import argparse
import json
import os

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--labels-usd", type=float, help="what the teacher's labels cost")
ap.add_argument("--labels-s", type=float, help="how long labelling took, seconds")
ap.add_argument("--train-s", type=float, default=45, help="loading + training, seconds")
ap.add_argument("--api-usd-per-million", type=float, default=43.12, help="the API's price per million messages")
ap.add_argument("--api-per-s", type=float, default=800, help="the API's sustained messages per second")
ap.add_argument("--own-per-s", type=float, default=18000, help="your model's messages per second (a 3090: ~4,000-21,000)")
ap.add_argument("--gpu-usd-per-hour", type=float, default=0.12)
a = ap.parse_args()


def breakeven(labels_usd, labels_s, train_s, api_usd_m, own_per_s, api_per_s=a.api_per_s, gpu_h=a.gpu_usd_per_hour):
    own_row = gpu_h / 3600 / own_per_s
    money = (labels_usd + gpu_h / 3600 * train_s) / (api_usd_m / 1e6 - own_row)
    time = (labels_s + train_s) / (1 / api_per_s - 1 / own_per_s)
    return money, time


if a.labels_usd is not None:
    m, t = breakeven(a.labels_usd, a.labels_s or 60, a.train_s, a.api_usd_per_million, a.own_per_s)
    print(f"pays back in money after {m:,.0f} messages, in time after {t:,.0f}")
else:
    rows = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "breakeven.json")))
    print(f"{'task':22s} {'labels':>7s} {'label s':>8s} {'label $':>8s} {'train s':>8s} {'API $/M':>8s} "
          f"{'money':>10s} {'time':>10s}   (messages to break even)")
    for r in rows:
        m, t = breakeven(r["label_usd"], r["label_s"], r["load_train_s"], r["jev_usd_per_M"], r["ettin_rows_per_s"])
        print(f"{r['task']:22s} {r['rows']:7,d} {r['label_s']:8.1f} {r['label_usd']:8.2f} {r['train_s']:8.1f} "
              f"{r['jev_usd_per_M']:8.2f} {m:10,.0f} {t:10,.0f}")
