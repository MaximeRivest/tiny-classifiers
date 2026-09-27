// Ettin-17M (ModernBERT) sequence classifier in jax-js: forward pass, AdamW training step, data batching.
// Mirrors transformers' ModernBertForSequenceClassification (mean pooling, exact GELU, no biases except
// the classifier's) and torch.optim.AdamW + clip_grad_norm_(1.0), as used by cpu_train.py.
import { jit, nn, numpy as np, tree, valueAndGrad } from "./lib/jax/index.js";

export const H = 256, N = 4, D = 64, I = 384, LAYERS = 7, EPS = 1e-5, NLAB = 77;
const WINDOW = 64, THETA = 160000, WD = 0.01, B1 = 0.9, B2 = 0.999, ADAM_EPS = 1e-8;
const GLOBAL_ATTN = [true, false, false, true, false, false, true];   // layer_types from the config

// ---- logging back to the server-----------------------------------------------------------------
const T0 = performance.now();
export function logger(name) {
  const lines = [], pre = document.getElementById("o");
  return (s) => {
    const line = `[${((performance.now() - T0) / 1000).toFixed(2)}s] ${s}`;
    lines.push(line); if (pre) pre.textContent = lines.slice(-40).join("\n");
    fetch(`/log/${name}`, { method: "POST", body: line }).catch(() => {});
  };
}

// ---- weights --------------------------------------------------------------------------------
function columns(src, rows, cols, c0, c1) {           // copy columns [c0, c1) of a row-major matrix
  const w = c1 - c0, out = new Float32Array(rows * w);
  for (let r = 0; r < rows; r++) out.set(src.subarray(r * cols + c0, r * cols + c1), r * w);
  return out;
}

// vocab: sorted original token ids that the data uses (see loadData). Only those embedding rows are
// loaded and trained: the other rows get zero gradient, so AdamW would only decay them, and they never
// reach a prediction on this data. Training the compact table gives the same model on banking77.
export async function loadWeights(base = "data", vocab = null) {
  const [index, buf] = await Promise.all([
    fetch(`${base}/weights.json`).then((r) => r.json()),
    fetch(`${base}/weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(buf);
  const get = (name) => {
    const { shape, offset } = index[name];
    return { data: all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1)), shape };
  };
  const arr = (data, shape) => np.array(data, { shape });
  const P = {};
  const put = (k, name) => { const t = get(name); P[k] = arr(t.data, t.shape); };
  if (vocab) {
    const e = get("model.embeddings.tok_embeddings.weight"), rows = new Float32Array(vocab.length * H);
    vocab.forEach((id, r) => rows.set(e.data.subarray(id * H, id * H + H), r * H));
    P.emb = arr(rows, [vocab.length, H]);
  } else put("emb", "model.embeddings.tok_embeddings.weight");
  put("emb_norm", "model.embeddings.norm.weight");
  for (let i = 0; i < LAYERS; i++) {
    const p = `model.layers.${i}.`;
    if (i > 0) put(`L${i}.attn_norm`, p + "attn_norm.weight");
    const qkv = get(p + "attn.Wqkv.weight");            // [256, 768] = [in, q|k|v]
    P[`L${i}.Wq`] = arr(columns(qkv.data, H, 3 * H, 0, H), [H, H]);
    P[`L${i}.Wk`] = arr(columns(qkv.data, H, 3 * H, H, 2 * H), [H, H]);
    P[`L${i}.Wv`] = arr(columns(qkv.data, H, 3 * H, 2 * H, 3 * H), [H, H]);
    put(`L${i}.Wo`, p + "attn.Wo.weight");
    put(`L${i}.mlp_norm`, p + "mlp_norm.weight");
    const wi = get(p + "mlp.Wi.weight");               // [256, 768] = [in, input|gate]
    P[`L${i}.Wi_in`] = arr(columns(wi.data, H, 2 * I, 0, I), [H, I]);
    P[`L${i}.Wi_gate`] = arr(columns(wi.data, H, 2 * I, I, 2 * I), [H, I]);
    put(`L${i}.Wmo`, p + "mlp.Wo.weight");
  }
  put("final_norm", "model.final_norm.weight");
  put("head_dense", "head.dense.weight");
  put("head_norm", "head.norm.weight");
  put("cls_w", "classifier.weight");
  put("cls_b", "classifier.bias");
  return P;
}

export function zerosLike(P) {
  const Z = {};
  for (const k in P) Z[k] = np.zeros(P[k].shape);
  return Z;
}

// ---- rotary position tables ------------------------------------------------------------------
const ropeCache = new Map();
export function ropeTables(L) {                        // cos, sin: [1, L, 1, 64]; R: x @ R == rotate_half(x)
  if (!ropeCache.has(L)) {
    const c = new Float32Array(L * D), s = new Float32Array(L * D);
    for (let l = 0; l < L; l++) for (let j = 0; j < D; j++) {
      const f = l / Math.pow(THETA, (2 * (j % (D / 2))) / D);
      c[l * D + j] = Math.cos(f); s[l * D + j] = Math.sin(f);
    }
    ropeCache.set(L, { c, s });
  }
  const { c, s } = ropeCache.get(L);
  const R = new Float32Array(D * D);
  for (let j = 0; j < D / 2; j++) { R[(j + D / 2) * D + j] = -1; R[j * D + j + D / 2] = 1; }
  return { cos: np.array(c, { shape: [1, L, 1, D] }), sin: np.array(s, { shape: [1, L, 1, D] }),
           R: np.array(R, { shape: [D, D] }) };
}

// ---- model ------------------------------------------------------------------------------------
const ln = (x, w) => nn.standardize(x, -1, { epsilon: EPS }).mul(w);
const gelu = (x) => nn.gelu(x, { approximate: false });

function rope(x, cos, sin, R) {                        // x: [B, L, N, D]
  const shape = x.shape;
  const rot = np.matmul(x.ref.reshape([-1, D]), R).reshape(shape);
  return x.mul(cos).add(rot.mul(sin));
}

// batch = { ids [B,L] int32, kbias [B,1,1,L] (0 real key / -1e9 padding), wbias [1,1,L,L] (0 inside the
// +-64 local window, -1e9 outside), pool [B,L,1] (mask / length), cos, sin, R }. Finite biases instead of
// -inf keep padded query rows finite (they are dropped by the pooling), so no NaN reaches the loss.
export function forward(P, batch) {
  const { ids, kbias, wbias, pool, cos, sin, R } = batch;
  const [B, L] = ids.shape;
  const noMask = !kbias;                               // (diagnostics) attention without any mask
  const biasLocal = noMask ? null : kbias.ref.add(wbias);
  let h = np.take(P.emb ?? batch.emb, ids.reshape([B * L]), 0).reshape([B, L, H]);
  h = ln(h, P.emb_norm);
  for (let i = 0; i < LAYERS; i++) {
    const x = i === 0 ? h.ref : ln(h.ref, P[`L${i}.attn_norm`]);
    const q = rope(np.matmul(x.ref, P[`L${i}.Wq`]).reshape([B, L, N, D]), cos.ref, sin.ref, R.ref);
    const k = rope(np.matmul(x.ref, P[`L${i}.Wk`]).reshape([B, L, N, D]), cos.ref, sin.ref, R.ref);
    const v = np.matmul(x, P[`L${i}.Wv`]).reshape([B, L, N, D]);
    const a = nn.dotProductAttention(q, k, v, noMask ? {} : { bias: GLOBAL_ATTN[i] ? kbias.ref : biasLocal.ref });
    h = h.add(np.matmul(a.reshape([B, L, H]), P[`L${i}.Wo`]));
    const m = ln(h.ref, P[`L${i}.mlp_norm`]);
    const act = gelu(np.matmul(m.ref, P[`L${i}.Wi_in`])).mul(np.matmul(m, P[`L${i}.Wi_gate`]));
    h = h.add(np.matmul(act, P[`L${i}.Wmo`]));
  }
  cos.dispose(); sin.dispose(); R.dispose(); if (!noMask) { kbias.dispose(); biasLocal.dispose(); }
  h = ln(h, P.final_norm);
  const pooled = h.mul(pool).sum(1);                   // masked mean over tokens: [B, H]
  const z = ln(gelu(np.matmul(pooled, P.head_dense)), P.head_norm);
  return np.matmul(z, P.cls_w).add(P.cls_b);          // logits [B, 77]
}

// loss = mean cross-entropy over the real rows (w = 1/n_real on real rows, 0 on filler rows)
export function lossFn(P, batch, yoh, w) {
  return nn.logSoftmax(forward(P, batch), -1).mul(yoh).sum(-1).mul(w).sum().mul(-1);
}

export const lossAndGrad = valueAndGrad(lossFn);

// Sum of squares, reduced one axis at a time: jax-js runs one huge 1-D reduction with little
// parallelism, which on the phone takes seconds for the 12.9M-entry embedding and trips the GPU watchdog.
export const sumSq = (x) => { let y = np.square(x); while (y.ndim > 1) y = y.sum(-1); return y.sum(); };

export const predict = jit((P, batch) => nn.softmax(forward(P, batch), -1));

// One optimizer step. hp = { lr, stepSize = lr / (1 - B1^t), bc2s = sqrt(1 - B2^t) } as 0-d arrays.
export const trainStep = jit((P, M, V, batch, yoh, w, hp) => {
  const [loss, G] = lossAndGrad(tree.ref(P), batch, yoh, w);
  const keys = Object.keys(G);
  let sq = null;
  for (const k of keys) { const s = sumSq(G[k].ref); sq = sq === null ? s : sq.add(s); }
  const gnorm = np.sqrt(sq);
  const coef = np.minimum(np.ones([]).div(gnorm.ref.add(1e-6)), 1);
  const decay = hp.lr.mul(-WD).add(1);
  const P2 = {}, M2 = {}, V2 = {};
  for (const k of keys) {
    const g = G[k].mul(coef.ref);
    const m = M[k].mul(B1).add(g.ref.mul(1 - B1));
    const v = V[k].mul(B2).add(np.square(g).mul(1 - B2));
    const denom = np.sqrt(v.ref).div(hp.bc2s.ref).add(ADAM_EPS);
    P2[k] = P[k].mul(decay.ref).sub(m.ref.div(denom).mul(hp.stepSize.ref));
    M2[k] = m; V2[k] = v;
  }
  coef.dispose(); decay.dispose(); hp.stepSize.dispose(); hp.bc2s.dispose();
  return { P: P2, M: M2, V: V2, loss, gnorm };
});

export function hyper(lr, t) {                         // t = 1-based step count, as in torch.optim.AdamW
  return { lr: np.array(lr), stepSize: np.array(lr / (1 - Math.pow(B1, t))),
           bc2s: np.array(Math.sqrt(1 - Math.pow(B2, t))) };
}

const winCache = new Map();
function windowBias(L) {                                // |i - j| <= 64 visible, as in the reference
  if (!winCache.has(L)) {
    const b = new Float32Array(L * L);
    for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if (Math.abs(i - j) > WINDOW) b[i * L + j] = -1e9;
    winCache.set(L, b);
  }
  return np.array(winCache.get(L), { shape: [1, 1, L, L] });
}

// ---- data -------------------------------------------------------------------------------------
export async function loadData(base = "data") {
  const [meta, buf] = await Promise.all([
    fetch(`${base}/rows.json`).then((r) => r.json()),
    fetch(`${base}/tokens.bin`).then((r) => r.arrayBuffer()),
  ]);
  const raw = new Int32Array(buf);
  const vocab = Int32Array.from(new Set([...raw, meta.pad_id])).sort();
  const pos = new Map(Array.from(vocab, (id, i) => [id, i]));
  const tokens = raw.map((id) => pos.get(id));          // ids into the compact embedding table
  return { ...meta, tokens, vocab, pad_id: pos.get(meta.pad_id), raw_pad_id: meta.pad_id };
}

// Padded length: next multiple of 8, skipping odd multiples of 16 above 16 (48, 80, 112). jax-js's
// WebGPU code for the fused loss+gradient program gives wrong results at those lengths (checked on the
// Adreno 660: forward-only and the wasm backend are right). Extra padding is masked, so math is unchanged.
export const bucket = (n) => { const L = Math.max(8, Math.ceil(n / 8) * 8); return L > 16 && L % 32 === 16 ? L + 8 : L; };

// Rows `idx` of `split`, padded to `B` rows (filler rows get weight 0) and to a length bucket.
export function makeBatch(data, split, idx, B = idx.length) {
  const S = data[split];
  const L = bucket(Math.max(...idx.map((i) => S.lengths[i])));
  const ids = new Int32Array(B * L).fill(data.pad_id), kb = new Float32Array(B * L).fill(-1e9);
  const pool = new Float32Array(B * L), yoh = new Float32Array(B * NLAB), w = new Float32Array(B);
  for (let r = idx.length; r < B; r++) kb[r * L] = 0;   // filler rows: one visible (pad) key, weight 0
  idx.forEach((i, r) => {
    const n = S.lengths[i], o = S.offsets[i];
    ids.set(data.tokens.subarray(o, o + n), r * L);
    for (let t = 0; t < n; t++) { pool[r * L + t] = 1 / n; kb[r * L + t] = 0; }
    yoh[r * NLAB + S.y[i]] = 1;
    w[r] = 1 / idx.length;
  });
  const { cos, sin, R } = ropeTables(L);
  return {
    batch: { ids: np.array(ids, { shape: [B, L] }), kbias: np.array(kb, { shape: [B, 1, 1, L] }),
             wbias: windowBias(L), pool: np.array(pool, { shape: [B, L, 1] }), cos, sin, R },
    yoh: np.array(yoh, { shape: [B, NLAB] }), w: np.array(w, { shape: [B] }), L,
  };
}

// Seeded shuffling (mulberry32), so a run can be repeated.
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function shuffle(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
  return arr;
}
