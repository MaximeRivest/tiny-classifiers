"""Label messages with a big "teacher" model (Kimi K3), across several providers at once, as fast as they allow.

    uv run recipe/label.py data/tasks/agnews                    # labels the 2,000 training messages
    uv run recipe/label.py data/tasks/agnews --split test       # the test set: how good is the teacher?
    uv run recipe/label.py data/tasks/banking77 --split train_big --providers openrouter

Keys come from the environment (or --keys FILE with KEY=value lines): FIREWORKS_API_KEY, TOGETHER_API_KEY,
DEEPINFRA_API_KEY, PARASAIL_API_KEY, OPENROUTER_API_KEY. Providers without a key are skipped; one is enough.
Output: labels/<task>-<split>.jsonl, one {"id", "pred", ...} per message (rerunning resumes), and a
.summary.json with time, cost and, when the messages have human labels, the teacher's accuracy.

Before training on a model's answers, check that the model's license and the provider's terms allow it.

How it goes fast without being rate-limited
- One shared queue. Each provider pulls work while it has free slots, so faster providers take more.
- Per-provider adaptive concurrency, like TCP: the number of requests in flight doubles after each
  full round of successes until the first sign of congestion, then grows by one; congestion (429, 503,
  timeouts, "overloaded") multiplies it by 0.7 and pauses that provider for Retry-After or 1 s; if the
  median answer time doubles, it stops growing (a queue is forming on their side).
- A circuit breaker pauses a provider after 15 failures in a row; 401/402/403 disable it for the run.
- Failed items go back to the queue (up to 4 attempts), preferably to another provider.
- Tail hedging: when the queue is empty, a request that is taking too long is also sent elsewhere;
  the first answer wins.
- A budget per provider (USD) stops sending once reached.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from collections import defaultdict, deque

import aiohttp

KEY_NAMES = {"fireworks": "FIREWORKS_API_KEY", "together": "TOGETHER_API_KEY", "deepinfra": "DEEPINFRA_API_KEY",
             "parasail": "PARASAIL_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
# name: url, model, request extras that turn thinking off, price per 1M tokens (input, cached input, output)
ENDPOINTS = {
    "fireworks": ("https://api.fireworks.ai/inference/v1/chat/completions", "accounts/fireworks/models/kimi-k3",
                  {"reasoning_effort": "none"}, (3.0, 0.30, 15.0)),
    "together": ("https://api.together.ai/v1/chat/completions", "moonshotai/Kimi-K3",
                 {"reasoning": {"enabled": False}}, (3.0, 0.30, 15.0)),
    "deepinfra": ("https://api.deepinfra.com/v1/openai/chat/completions", "moonshotai/Kimi-K3",
                  {"reasoning": {"enabled": False}}, (2.85, 0.285, 14.25)),
    "parasail": ("https://api.parasail.io/v1/chat/completions", "parasail-kimi-k3",
                 {"reasoning_effort": "none"}, (3.0, 0.30, 15.0)),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "moonshotai/kimi-k3",
                   {"reasoning": {"enabled": False}}, (3.0, 0.30, 15.0)),
}
PROVIDERS = {}                     # filled in main(): name -> (url, key, model, extras, price)
STATE = os.path.expanduser("~/.cache/tiny-classifiers/limits.json")   # learned in-flight limits, for a warm start
CONGESTION = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL = {401, 402, 403}


class Provider:
    def __init__(self, name, cap, budget, start=0.5):
        self.name = name
        self.url, self.key, self.model, self.extra, self.price = PROVIDERS[name]
        self.cap, self.budget = cap, budget
        learned = (json.load(open(STATE)) if os.path.exists(STATE) else {}).get(name)
        self.limit = max(4.0, min(cap, start * learned)) if learned else 4.0  # start at a fraction of last time
        self.ssthresh = float("inf")
        self.peak_ok_limit = self.limit
        self.inflight, self.ok_since_grow = 0, 0
        self.paused_until, self.last_cut = 0.0, 0.0
        self.consec_fail, self.disabled = 0, None
        self.lat = deque(maxlen=60)
        self.best_med = None
        self.spent = 0.0
        self.n = defaultdict(int)            # ok, congestion, error, invalid, hedged
        self.timeline = defaultdict(int)     # second -> answers
        self.limit_trace = []
        self.session = None                  # aiohttp session, created inside the event loop

    def open(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.cap + 16, ttl_dns_cache=600, keepalive_timeout=60),
            timeout=aiohttp.ClientTimeout(total=25, connect=10))

    def median(self):
        return statistics.median(self.lat) if len(self.lat) >= 10 else None

    def can_send(self, now):
        # budget counts requests already in flight at the average cost so far (~$0.0015 before data)
        per = self.spent / self.n["ok"] if self.n["ok"] >= 20 else 0.0015
        return (not self.disabled and now >= self.paused_until and self.inflight < int(self.limit)
                and self.spent + (self.inflight + 1) * per <= self.budget)

    def on_success(self, latency):
        self.consec_fail = 0
        self.lat.append(latency)
        med = self.median()
        if med is not None:
            self.best_med = med if self.best_med is None else min(self.best_med, med)
        if med is not None and self.best_med and med > 2.0 * self.best_med:
            self.ssthresh = min(self.ssthresh, self.limit)       # latency guard: hold
            return
        self.ok_since_grow += 1
        if self.ok_since_grow >= int(self.limit):
            self.ok_since_grow = 0
            self.limit = min(self.cap, self.limit * 2 if self.limit < self.ssthresh else self.limit + 1)
            self.peak_ok_limit = max(self.peak_ok_limit, self.limit)

    def on_congestion(self, now, retry_after):
        self.consec_fail += 1
        if now - self.last_cut > 2.0:
            self.last_cut = now
            self.limit = max(1.0, self.limit * 0.7)
            self.ssthresh = self.limit
            self.ok_since_grow = 0
        self.paused_until = max(self.paused_until, now + min(30.0, retry_after or 1.0))
        if self.consec_fail >= 15:                                # circuit breaker
            self.paused_until = now + 30.0
            self.limit, self.consec_fail = 2.0, 0


MODELS_URL = {"fireworks": "https://api.fireworks.ai/inference/v1/models", "together": "https://api.together.ai/v1/models",
              "deepinfra": "https://api.deepinfra.com/v1/openai/models", "parasail": "https://api.parasail.io/v1/models",
              "openrouter": "https://openrouter.ai/api/v1/models"}


async def prewarm(providers, system, conns=64):
    """Open `conns` keep-alive connections per provider (cheap GETs) and send one real request per provider
    with the task's instruction, so the prefix is in the provider's cache. Returns seconds spent."""
    t = time.monotonic()

    async def get(p):
        try:
            async with p.session.get(MODELS_URL[p.name], headers={"Authorization": f"Bearer {p.key}"}) as r:
                await r.read()
        except Exception:
            pass

    async def prime(p):
        body = {"model": p.model, "max_tokens": 8, "temperature": 0, **p.extra,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": "hello"}]}
        for _ in range(2):
            try:
                async with p.session.post(p.url, json=body, headers={"Authorization": f"Bearer {p.key}"}) as r:
                    await r.read()
            except Exception:
                pass

    await asyncio.gather(*[get(p) for p in providers for _ in range(conns)], *[prime(p) for p in providers])
    return time.monotonic() - t


def parse(content, labels):
    """Exact label (case-insensitive), else the longest label contained in the answer, else None."""
    s = (content or "").strip().strip("`\"' .*").lower()
    low = {l.lower(): l for l in labels}
    if s in low:
        return low[s]
    first = s.split("\n")[0].strip("`\"'.,*: ")
    if first in low:
        return low[first]
    hits = [l for l in low if l in s]
    return low[max(hits, key=len)] if hits else None


async def run(task, providers, out_path, max_attempts=4, hedge=True, progress_every=1.0, max_tokens=24,
              warm=False, hedge_after=3.0, hedge_mult=3.0, learn=True):
    labels, system = task["labels"], task["instruction"]
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [it for it in task["items"] if it["id"] not in done]
    queue = deque((it, 1, None, 0.0) for it in todo)  # (item, attempt, provider to avoid, time queued)
    by_id = {it["id"]: it for it in todo}
    for p in providers:
        p.open()
    warm_s = await prewarm(providers, system) if warm else 0.0
    if warm:
        print(f"prewarm {warm_s:.1f} s (not counted below)", flush=True)
    inflight = {}                                     # id -> (start time, provider names)
    out = open(out_path, "a")
    total_new = len(todo)
    finished = 0
    wake = asyncio.Event()

    async def one(p, item, attempt, hedged):
        nonlocal finished
        body = {"model": p.model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": item["text"]}],
                **p.extra}
        ts = time.monotonic()
        status, retry_after, err = None, None, None
        try:
            async with p.session.post(p.url, json=body, headers={"Authorization": f"Bearer {p.key}"}) as r:
                status = r.status
                try:
                    retry_after = float(r.headers.get("retry-after", 0) or 0) or None
                except ValueError:
                    retry_after = None
                text = await r.text()
            if status == 200:
                d = json.loads(text)
            else:
                err = text[:200]
                if status == 400 and any(w in err.lower() for w in ("busy", "overload", "capacity")):
                    status = 503
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            status, err = 599, type(e).__name__
        now = time.monotonic()
        p.inflight -= 1
        if status == 200:
            u = d.get("usage") or {}
            pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            p.spent += ((pt - cached) * p.price[0] + cached * p.price[1] + ct * p.price[2]) / 1e6
            p.on_success(now - ts)
            if item["id"] in done:                       # a hedge lost the race
                p.n["hedge_wasted"] += 1
            else:
                content = d["choices"][0]["message"].get("content")
                pred = parse(content, labels)
                p.n["ok"] += 1
                p.n["invalid"] += pred is None
                p.timeline[int(now - t0)] += 1
                rec = {"id": item["id"], "pred": pred, "raw": (content or "")[:80], "provider": p.name,
                       "latency_s": round(now - ts, 3), "t_sent": round(ts - t0, 3), "t_done": round(now - t0, 3),
                       "attempt": attempt, "hedged": hedged,
                       "prompt_tokens": pt, "cached_tokens": cached, "completion_tokens": ct}
                if "label" in item:
                    rec["label"] = item["label"]
                done[item["id"]] = rec
                inflight.pop(item["id"], None)
                out.write(json.dumps(rec) + "\n")
                finished += 1
        elif status in FATAL:
            p.disabled = f"{status} {err}"
            print(f"  !! {p.name} disabled: {p.disabled}", flush=True)
            requeue(item, attempt, p.name, count=False)
        else:
            p.n["congestion" if status in CONGESTION else "error"] += 1
            p.n[f"s{status}"] += 1
            if status in CONGESTION:
                p.on_congestion(now, retry_after)
            else:
                p.consec_fail += 1
            if p.n.get("first_err") is None:
                p.n["first_err"] = 1
                print(f"  .. {p.name} first failure: {status} {err}", flush=True)
            if not hedged:                                 # a failed hedge is simply dropped
                requeue(item, attempt, p.name)
        wake.set()

    def requeue(item, attempt, avoid, count=True):
        nonlocal finished
        if item["id"] in done:
            return
        inflight.pop(item["id"], None)                   # not in flight any more; resent later
        a = attempt + 1 if count else attempt
        if a > max_attempts:
            rec = {"id": item["id"], "pred": None, "failed": True, "provider": avoid, "attempt": attempt}
            if "label" in item:
                rec["label"] = item["label"]
            done[item["id"]] = rec
            inflight.pop(item["id"], None)
            out.write(json.dumps(rec) + "\n")
            finished += 1
        else:
            queue.appendleft((item, a, avoid, time.monotonic()))

    lag = {"max": 0.0}

    async def lag_probe():                              # how late the event loop wakes up
        while True:
            a = time.monotonic()
            await asyncio.sleep(0.05)
            lag["max"] = max(lag["max"], time.monotonic() - a - 0.05)

    async def report():
        last_n, last_t, last_cpu = 0, time.monotonic(), time.process_time()
        while finished < total_new:
            await asyncio.sleep(progress_every)
            now = time.monotonic()
            rate = (finished - last_n) / (now - last_t)
            cpu = (time.process_time() - last_cpu) / (now - last_t)
            loop_lag.append((round(now - t0, 1), round(lag["max"], 3), round(cpu, 2)))
            last_n, last_t, last_cpu = finished, now, time.process_time()
            parts = []
            for p in providers:
                p.limit_trace.append((round(now - t0, 1), int(p.limit), p.inflight))
                state = "OFF" if p.disabled else ("pause" if now < p.paused_until else "")
                parts.append(f"{p.name[:4]} lim {int(p.limit):3d} fly {p.inflight:3d} ok {p.n['ok']:5d} "
                             f"cong {p.n['congestion']:3d} ${p.spent:4.2f}{(' ' + state) if state else ''}")
            print(f"{now - t0:6.1f}s {finished:5d}/{total_new} {rate:6.1f}/s cpu {cpu:4.0%} lag {lag['max']*1000:5.0f}ms | "
                  + " | ".join(parts), flush=True)
            lag["max"] = 0.0
            wake.set()

    loop_lag = []
    async def hedger():                                 # tail hedging, 4 times a second
        while True:
            await asyncio.sleep(0.25)
            if not hedge or queue:
                continue
            now = time.monotonic()
            base = min((q.median() for q in providers if q.median()), default=1.5)
            for iid, (start, names) in list(inflight.items()):
                if iid in done or len(names) >= 3 or now - start < max(hedge_after, hedge_mult * base):
                    continue
                cands = [q for q in providers if q.name not in names and q.can_send(now)]
                if not cands:
                    continue
                q = min(cands, key=lambda q: q.median() or 9)      # fastest right now
                q.inflight += 1
                q.n["hedged"] += 1
                names.append(q.name)
                tasks.add(asyncio.create_task(one(q, by_id[iid], 1, True)))

    t0 = time.monotonic()
    reporter = asyncio.create_task(report())
    hedge_task = asyncio.create_task(hedger())
    prober = asyncio.create_task(lag_probe())
    tasks = set()
    while finished < total_new:
        now = time.monotonic()
        sent = False
        for p in sorted(providers, key=lambda p: p.median() or 1.5):   # fastest providers take work first
            while p.can_send(now) and queue:
                idx = next((i for i, (_, _, av, tq) in enumerate(queue)
                            if av != p.name or now - tq > 0.5), None)
                if idx is None:
                    break
                item, attempt, _, _ = queue[idx]
                del queue[idx]
                if item["id"] in done:
                    continue
                p.inflight += 1
                inflight.setdefault(item["id"], (now, []))[1].append(p.name)
                tasks.add(asyncio.create_task(one(p, item, attempt, False)))
                sent = True
        tasks = {t for t in tasks if not t.done()}
        if not any(p.inflight for p in providers) and not any(p.can_send(time.monotonic() + 60) for p in providers):
            print("  !! every provider is disabled or out of budget; stopping", flush=True)
            break
        if not sent:
            wake.clear()
            try:
                await asyncio.wait_for(wake.wait(), 0.2)
            except asyncio.TimeoutError:
                pass
    reporter.cancel()
    prober.cancel()
    hedge_task.cancel()
    for t in tasks:
        t.cancel()
    for p in providers:
        await p.session.close()
    out.close()
    wall = time.monotonic() - t0
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for p in (providers if learn else []):                     # remember the largest limit reached, capped by the last cut
        if p.n["ok"] >= 50 and not p.disabled:
            state[p.name] = int(min(p.peak_ok_limit, p.ssthresh if p.ssthresh != float("inf") else p.cap))
    json.dump(state, open(STATE, "w"))
    recs = [done[it["id"]] for it in task["items"] if it["id"] in done]
    scored = [r for r in recs if "label" in r and not r.get("failed")]
    summary = {
        "task": task["name"], "items": len(task["items"]), "prewarm_s": round(warm_s, 1), "answered_this_run": finished, "wall_s": round(wall, 1),
        "answers_per_s": round(finished / wall, 1) if wall else None,
        "accuracy": round(sum(r["pred"] == r["label"] for r in scored) / len(scored), 4) if scored else None,
        "failed": sum(1 for r in recs if r.get("failed")),
        "loop_lag_cpu": loop_lag,
        "invalid": sum(1 for r in recs if not r.get("failed") and r["pred"] is None),
        "providers": {p.name: {"answers": p.n["ok"], "spent_usd": round(p.spent, 3), "final_limit": int(p.limit),
                               "median_latency_s": round(p.median(), 2) if p.median() else None,
                               "peak_answers_per_s": max(p.timeline.values(), default=0),
                               "disabled": p.disabled,
                               "counts": {k: v for k, v in p.n.items() if k != "first_err"},
                               "timeline": dict(sorted(p.timeline.items())), "limit_trace": p.limit_trace}
                      for p in providers},
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", help="a task folder: task.json + <split>.jsonl")
    ap.add_argument("--split", default="train", help="train (default), train_big or test")
    ap.add_argument("--n", type=int, help="only the first N messages")
    ap.add_argument("--providers", default="fireworks,together,deepinfra,parasail,openrouter")
    ap.add_argument("--keys", help="a file of KEY=value lines, read before the environment")
    ap.add_argument("--cap", default="fireworks=512,together=320,deepinfra=32,parasail=256,openrouter=256",
                    help="max requests in flight: one number, or name=N,name=N")
    ap.add_argument("--budget", default="5", help="USD per provider for this run: one number, or name=X,name=Y")
    ap.add_argument("--out", help="default: labels/<task>-<split>.jsonl")
    ap.add_argument("--no-hedge", action="store_true")
    a = ap.parse_args()

    keys = {}
    if a.keys:
        keys = dict(l.strip().removeprefix("export ").split("=", 1) for l in open(os.path.expanduser(a.keys))
                    if "=" in l and not l.lstrip().startswith("#"))
    for name, (url, model, extra, price) in ENDPOINTS.items():
        key = (os.environ.get(KEY_NAMES[name]) or keys.get(KEY_NAMES[name], "")).strip().strip('"\'')
        PROVIDERS[name] = (url, key, model, extra, price)
    pick = lambda s, d: ({k: v for k, v in (x.split("=") for x in s.split(","))} if "=" in s else defaultdict(lambda: s)) if s else d
    caps, budgets = pick(a.cap, {}), pick(a.budget, {})

    t = json.load(open(os.path.join(a.task, "task.json")))
    items = [json.loads(l) for l in open(os.path.join(a.task, f"{a.split}.jsonl"))]
    instruction = (t["instruction"] + " Answer with exactly one label from this list and nothing else:\n"
                   + "\n".join(t["labels"]))
    task = {"name": f"{t['name']}-{a.split}", "instruction": instruction, "labels": t["labels"],
            "items": items[:a.n] if a.n else items}
    out = a.out or os.path.join("labels", f"{task['name']}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    provs = [Provider(n, int(caps.get(n, 256)), float(budgets.get(n, 5)), 1.0)
             for n in a.providers.split(",") if PROVIDERS[n][1]]
    if not provs:
        raise SystemExit("no provider has a key: set e.g. OPENROUTER_API_KEY or FIREWORKS_API_KEY (see --help)")
    print(f"{len(task['items'])} messages -> {out}; providers {[p.name for p in provs]}", flush=True)
    summary = asyncio.run(run(task, provs, out, hedge=not a.no_hedge, hedge_after=1.5, hedge_mult=2.0))
    json.dump(summary, open(out.replace(".jsonl", ".summary.json"), "w"), indent=1)
    cost = sum(p["spent_usd"] for p in summary["providers"].values())
    acc = f", teacher accuracy {summary['accuracy']:.1%}" if summary["accuracy"] is not None else ""
    print(f"done: {summary['answered_this_run']} labels in {summary['wall_s']} s, ${cost:.2f}, "
          f"{summary['failed']} failed, {summary['invalid']} not a valid label{acc}")
    for n, p in summary["providers"].items():
        print(f"  {n:10s} {p['answers']:6d} answers  ${p['spent_usd']:.2f}  median {p['median_latency_s']} s")


if __name__ == "__main__":
    main()
