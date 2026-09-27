# The film: tiny classifiers, the long version (6:18, 1080p60, narrated, captioned)

An explainer for the people who asked, after the banking77 / Jev / Ettin posts (Sept 25–26, 2026):
what is Ettin, where does the data come from, and when is fine-tuning worth it.

| File | What |
|---|---|
| `out/tiny-classifiers-long-version.mp4` | The film (not in the repository: rendered from these files) |

## The story

| Scene | The point | Numbers from |
|---|---|---|
| hook | Thumbnail at 0:00: 17M model, 42 s, ≈ Opus + the chart; your questions | `results/banking77_learning_curves.json`, the posts |
| job, ways | What a classifier is; three ways to get one | — |
| result | banking77: Ettin 91.5, Opus 92, Kimi 82, Jev 79.4; cost and speed | `results/charts/gpu_laptop_phone.png` |
| ettin | An encoder: reads all at once, one score per category; the code | `recipe/train.py` |
| anywhere | GPU 42 s, laptop 11 min, phone 21 min, same accuracy | `results/charts/gpu_laptop_phone.png` |
| curve | The labels are the whole game; no curve has flattened | `results/banking77_learning_curves.json` |
| distill | Borrow a teacher: 9,493 Kimi labels in 28 s for $14 over 4 providers; check the terms | `results/breakeven.json` |
| ceiling | A student lands under its teacher (79.5 vs 81.6); human labels 91.5 | `results/banking77_human_vs_kimi_labels.txt` |
| soft | Probabilities: same accuracy, honest confidence (95% sure / 77% right vs 96% on its sure half) | `results/README.md` (probabilities) |
| general | 10 tasks: human labels beat Jev on 6; Kimi labels tie on 2, within 5 on 5 more; why it loses | `results/ten_tasks_table.txt` |
| pays | Break-even vs Jev: 55k–504k messages | `results/breakeven.json` |
| twist | The labels you need to evaluate are the labels you need to train; Jev ≈ Python, fine-tuned ≈ Rust | reply by @maxinel_ai |
| recipe, close | Seven steps; evaluate first, fine-tune what pays | — |

## How it is made

The narration drives everything. `script.js` holds the words; `voice.mjs` speaks each line with Kokoro
(a small Kokoro server on a local RTX 3090: `POST /tts {text, voice, speed}` returning 24 kHz 16-bit PCM; voice `af_heart`, speed 1.0) and measures it; `timeline.js` lays the lines end to end,
and every visual event is pinned to a word (`at('r3', 'Opus')`), so picture, music and voice always agree.

```sh
node voice.mjs                       # speak changed lines (writes the .key files)
python words.py [ids]                # (in an environment with kokoro)   # re-take lines with word timings (captions); no ids = all
node voice.mjs --check               # durations + transcribe each line back (Parakeet) to catch mispronunciations
node music.mjs                       # work/music.wav, from the same timeline
node mix.mjs                         # work/mix.wav: voice over music, music ducked under speech, −16 LUFS
node render.mjs --stills 60,120      # single frames (work/still-*.jpg)
node render.mjs --from 60 --to 90    # preview a part
node render.mjs --jobs 32            # the whole film, about 12 minutes on a 48-core machine
```

- **Change a word:** edit `script.js`, run the four commands. Only changed lines are spoken again; the picture re-times itself.
- **Pronunciation:** a line's `say` overrides what is read; Kokoro takes phonemes like `[Jev](/ʤˈɛv/)`.
- **Your own voice:** record one file per line into `work/vo/<id>.pcm` (24 kHz mono, 16-bit), rerun `voice.mjs`
  without `--check` after removing the `.key` files' mismatch (or just write `durations.js`); timing follows your recording.
- **Picture:** `film.html` + `film.css`. Every frame is a function of time. Open it in a browser to watch it live
  (`film.html?t=150` starts at 2:30).

### Frame 0 and captions

- **Frame 0 is the thumbnail**: headline + the chart of the most-viewed post, fully drawn (nothing fades in).
- **Captions** are burned in for people watching without sound: written words (digits, not spelled numbers),
  timed from Kokoro's own word timings (`work/vo/words.js`), matched to the spoken text phrase by phrase.
  Kokoro is not bit-for-bit repeatable, so `words.py` keeps its take as the recording.
  Footnotes and chart bottoms sit above the caption band (bottom 110 px).

### Two rules that keep the picture clean

1. **Frames are drawn at 2× and shrunk** (`--ss 2`), and nothing uses `will-change`: text never snaps between pixels.
2. **Workers write lossless pieces; one encode at the end.** Encoding each worker's piece separately made static
   text flicker at every seam (each piece starts on a fresh keyframe). Measured after the fix: 0–4/255 on static text.

`--check` transcribes each line back with a speech-to-text server (`STT`, Parakeet here); skip it if you have none.
