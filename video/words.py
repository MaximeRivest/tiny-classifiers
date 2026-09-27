"""Word timings (and the recordings) for the captions: re-speak each line with Kokoro (same voice, speed, sentence split as the
kokoro-tts server), keep each word's start/end, shift them by the silence voice.mjs trims, and write this take
as work/vo/<id>.pcm (Kokoro varies slightly between takes). Writes work/vo/words.js.
Run in a Python environment with the `kokoro` package and a GPU:  python words.py [line ids]"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np, torch
from kokoro import KModel, KPipeline

here = Path(__file__).parent
script = json.loads(subprocess.check_output(["node", "-e",
    "const g={};new Function('globalThis','window',require('fs').readFileSync('script.js','utf8'))(g);"
    "const v={};new Function('globalThis','window',require('fs').readFileSync('work/vo/durations.js','utf8'))(v);"
    "console.log(JSON.stringify({S:g.SCRIPT,VO:v.VO}))"], cwd=here))
VOICE, SPEED = script["VO"]["voice"], script["VO"]["speed"]
SPLIT = r"(?<=[.!?;:])\s+|\n+"
model = KModel().to("cuda").eval()
pipe = KPipeline(lang_code=VOICE[0], model=model)
only = set(sys.argv[1:])   # re-take only these lines (ids); keep the others
out = {}
if only and (here / "work/vo/words.js").exists():
    t = (here / "work/vo/words.js").read_text(); out = json.loads(t[t.index("r.WORDS = ") + 10:t.rindex("; })")])
for sc in script["S"]:
    for ln in sc["lines"]:
        if only and ln["id"] not in only: continue
        say = ln.get("say") or ln["text"]
        audio, words, off = [], [], 0
        with torch.inference_mode():
            for r in pipe(say, voice=VOICE, speed=SPEED, split_pattern=SPLIT):
                if r.audio is None: continue
                a = r.audio.detach().cpu().numpy()
                for tk in r.tokens or []:
                    if tk.start_ts is None: continue
                    words.append([tk.text, off / 24000 + tk.start_ts, off / 24000 + (tk.end_ts or tk.start_ts), tk.whitespace])
                audio.append(a); off += len(a)
        s = np.clip(np.concatenate(audio) * 32767, -32768, 32767).astype("<i2")
        # the same trim as voice.mjs: first/last sample above 300, keep 960 samples of air
        idx = np.nonzero(np.abs(s) >= 300)[0]
        a0 = max(0, idx[0] - 960)
        b1 = min(len(s), idx[-1] + 960)
        # Kokoro is not bit-for-bit repeatable, so this take becomes the recording: timings then match exactly.
        (here / "work/vo" / f"{ln['id']}.pcm").write_bytes(s[a0:b1].tobytes())
        diff = 0.0
        out[ln["id"]] = [[w, round(t0 - a0 / 24000, 3), round(t1 - a0 / 24000, 3), ws] for w, t0, t1, ws in words]
        print(ln["id"], len(words), "words", file=sys.stderr)
(here / "work/vo/words.js").write_text("// written by words.py\n(function (r) { r.WORDS = " + json.dumps(out) + "; })(typeof window !== 'undefined' ? window : globalThis);\n")
print(len(out), "lines")
