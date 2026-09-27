# Train Ettin-17M on a phone's GPU, in the browser

The banking77 recipe (all weights, batch 32, AdamW 1e-4, warmup + cosine, clip 1.0, 6 passes) running
entirely on a phone's graphics chip, in Chrome, through WebGPU and [jax-js](https://github.com/ekzhang/jax-js).
No app, no server doing the math: a web page.

**Measured on a Galaxy S21 Ultra (Adreno 660), plugged in:** 21 min 17 s to train on 9,493 messages,
**91.4%** right on the 3,076 test messages (the RTX 3090 gets 91.5% in 42 s), 39 ms per message afterwards
([results/phone_gpu_run.json](results/phone_gpu_run.json)).

This is research code, kept as it ran. It is honest about the math: `check.html` compares one forward pass,
the gradients and one AdamW step with PyTorch (they agree to about 1e-5), and `traj.html` compares 60 steps.

## Run it

```sh
uv run recipe/prepare.py banking77          # from the repository root: builds data/tasks/banking77
cd phone/prep && uv run prep.py             # tokens, starting weights and a PyTorch reference -> ../web/data
cd ../web && python3 serve.py 8771          # static files + a small log endpoint (logs/<page>.log)
```

Then open `http://localhost:8771/check.html` in a browser with WebGPU (Chrome on Android, or desktop Chrome),
then `train.html` for the full run (`?bench=20` times 20 steps; `?epochs=1` for a short run).

On a phone over USB (`adb`): `scripts/web check.html` forwards the port and opens the page on the phone;
`scripts/web log train` shows what it reported. `scripts/guard` watches the phone's free memory and closes
Chrome before Android starts killing other apps (it happened on the first try).

Without a GPU at all, desktop Chrome can run the check on its software renderer:
`chromium --enable-unsafe-webgpu --enable-unsafe-swiftshader --use-webgpu-adapter=swiftshader`.

## What it took

- **Memory.** jax-js kept every GPU buffer it had ever allocated for reuse, which filled the phone's shared
  memory. `web/lib/jax/` is jax-js 0.1.25 (MIT, see its `LICENSE`) with two local patches (search
  "Local patch"): a cap on the reuse pool, and a fix for routines reading views.
- **Only the words used.** The embedding table is 50,368 words × 256; training only the 3,376 words that
  appear in banking77 gives exactly the same result, with far less memory. The trade-off: a model trained this
  way does not know words it never saw until you merge its rows back into the full table.
- **A WebGPU kernel bug** gave wrong losses at some padded lengths (48, 80, 112); batches are padded around
  them. Found by comparing with PyTorch step by step (`buckets.html`).
- **Why it is slower than a laptop CPU:** each step is small, so the time goes into launching hundreds of tiny
  GPU tasks, not into the arithmetic.

| file | what |
|---|---|
| `prep/prep.py` | tokens, weights (float32, Linear weights transposed) and a one-step PyTorch reference |
| `prep/prep_traj.py`, `prep/prep_buckets.py` | a 60-step reference; losses per padded length |
| `web/ettin.js` | the model, the loss, AdamW, batching |
| `web/check.html`, `traj.html`, `buckets.html` | checks against PyTorch |
| `web/train.html` | the full training run |
