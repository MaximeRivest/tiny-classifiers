#!/usr/bin/env node
// The soundtrack, synthesised from the same timeline as the picture: work/music.wav
// A dark pulse under the hook, a calm curious groove through the experiments, a breath for the twist, a resolve.
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const { TL } = await import('./load.mjs');
const { SCENES, L: LN, at } = TL;

const SR = 48000, DUR = TL.duration, N = Math.ceil(SR * DUR);
const L = new Float32Array(N), R = new Float32Array(N);        // dry
const RL = new Float32Array(N), RR = new Float32Array(N);      // reverb send
let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const mtof = m => 440 * 2 ** ((m - 69) / 12);
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const smooth = x => { x = clamp(x); return x * x * (3 - 2 * x); };

function put(i, l, r, send = 0) {
  if (i < 0 || i >= N) return;
  L[i] += l; R[i] += r; RL[i] += l * send; RR[i] += r * send;
}

// A plucked, soft electric-piano-like note: a few sine partials, each decaying at its own rate.
function epiano(t0, midi, dur, vel = .5, pan = 0, send = .35) {
  const f = mtof(midi), i0 = Math.floor(t0 * SR), n = Math.floor((dur + 1.6) * SR);
  const parts = [[1, 1, 1.6], [2, .38, 3.2], [3, .12, 5], [4, .06, 7], [7.02, .025, 12]];
  const gl = Math.cos((pan + 1) * Math.PI / 4), gr = Math.sin((pan + 1) * Math.PI / 4);
  for (let k = 0; k < n; k++) {
    const t = k / SR;
    const att = Math.min(1, t / .004);
    const rel = t > dur ? Math.exp(-(t - dur) * 6) : 1;
    let s = 0;
    for (const [m, a, d] of parts) s += a * Math.sin(2 * Math.PI * f * m * t + (m > 1 ? .3 * Math.sin(2 * Math.PI * 5 * t) * .02 : 0)) * Math.exp(-t * d * (1 + f / 2000));
    s *= att * rel * vel * .34;
    put(i0 + k, s * gl, s * gr, send);
  }
}
// A slow pad: detuned saws through a one-pole low-pass, with long attack/release.
function pad(t0, t1, midis, vel = .12, cutoff = 1200, att = 1.5, rel = 2.5, send = .6, cutEnv) {
  const i0 = Math.floor(t0 * SR), n = Math.floor((t1 - t0 + rel) * SR);
  const voices = [];
  for (const m of midis) for (const det of [-.09, .0, .085]) voices.push({ f: mtof(m + det * .12), ph: rnd(), pan: det * 6 });
  let yl = 0, yr = 0;
  for (let k = 0; k < n; k++) {
    const t = k / SR, abs = t0 + t;
    const env = Math.min(1, smooth(t / att)) * (abs > t1 ? Math.exp(-(abs - t1) * 3 / rel) : 1);
    let sl = 0, sr = 0;
    for (const v of voices) {
      v.ph += v.f / SR; if (v.ph >= 1) v.ph -= 1;
      const s = 2 * v.ph - 1;
      sl += s * (1 - v.pan) * .5; sr += s * (1 + v.pan) * .5;
    }
    const c = cutEnv ? cutEnv(abs) : cutoff;
    const a = 1 - Math.exp(-2 * Math.PI * c / SR);
    yl += a * (sl - yl); yr += a * (sr - yr);
    const g2 = env * vel / Math.sqrt(voices.length);
    put(i0 + k, yl * g2, yr * g2, send);
  }
}
function sub(t0, dur, midi, vel = .35) {
  const f = mtof(midi), i0 = Math.floor(t0 * SR), n = Math.floor(dur * SR);
  for (let k = 0; k < n; k++) {
    const t = k / SR;
    const env = Math.min(1, t / .02) * Math.min(1, (dur - t) / .08);
    const s = Math.sin(2 * Math.PI * f * t) * env * vel * .5;
    put(i0 + k, s, s, 0);
  }
}
function kick(t0, vel = .8) {
  const i0 = Math.floor(t0 * SR), n = Math.floor(.45 * SR); let ph = 0;
  for (let k = 0; k < n; k++) {
    const t = k / SR, f = 48 + 90 * Math.exp(-t * 32);
    ph += f / SR;
    const s = (Math.sin(2 * Math.PI * ph) + .25 * Math.sin(4 * Math.PI * ph)) * Math.exp(-t * 7.5) * vel * .55 + (t < .006 ? (rnd() - .5) * vel * .5 : 0);
    put(i0 + k, s, s, .02);
  }
}
function noiseHit(t0, dur, vel, { hp = .6, decay = 40, pan = 0, send = .15, tone = 0 } = {}) {
  const i0 = Math.floor(t0 * SR), n = Math.floor(dur * SR); let prev = 0, lp = 0;
  const gl = Math.cos((pan + 1) * Math.PI / 4), gr = Math.sin((pan + 1) * Math.PI / 4);
  for (let k = 0; k < n; k++) {
    const t = k / SR, w = rnd() * 2 - 1;
    const h = w - prev * hp; prev = w; lp += .5 * (h - lp);
    const s = (tone ? lp : h) * Math.exp(-t * decay) * Math.min(1, t / .001) * vel;
    put(i0 + k, s * gl, s * gr, send);
  }
}
const hat = (t, v = .06, pan = .3) => noiseHit(t, .09, v, { hp: .95, decay: 55, pan, send: .08 });
const shaker = (t, v = .035, pan = -.35) => noiseHit(t, .14, v, { hp: .9, decay: 28, pan, send: .1 });
const snap = (t, v = .16) => { noiseHit(t, .22, v, { hp: .7, decay: 22, send: .35 }); };
// Filtered noise sweep (whoosh), up or down.
function whoosh(t0, dur, vel = .2, up = true) {
  const i0 = Math.floor(t0 * SR), n = Math.floor(dur * SR); let y = 0, y2 = 0;
  for (let k = 0; k < n; k++) {
    const x = k / n, w = rnd() * 2 - 1;
    const c = up ? 200 + 5000 * x * x : 5200 - 5000 * Math.sqrt(x);
    const a = 1 - Math.exp(-2 * Math.PI * c / SR);
    y += a * (w - y); y2 += a * (y - y2);
    const env = Math.sin(Math.PI * x) ** 1.5;
    const s = (y - y2 * .5) * env * vel;
    put(i0 + k, s * (1 - x * .4), s * (.6 + x * .4), .4);
  }
}
function tick(t0, vel = .2, f = 2400) {
  const i0 = Math.floor(t0 * SR), n = Math.floor(.05 * SR);
  for (let k = 0; k < n; k++) { const t = k / SR; const s = Math.sin(2 * Math.PI * f * t) * Math.exp(-t * 90) * vel * .5; put(i0 + k, s, s, .25); }
}
function blip(t0, vel, midi) {
  const f = mtof(midi), i0 = Math.floor(t0 * SR), n = Math.floor(.35 * SR);
  for (let k = 0; k < n; k++) { const t = k / SR; const s = (Math.sin(2 * Math.PI * f * t) + .3 * Math.sin(4 * Math.PI * f * t)) * Math.exp(-t * 14) * Math.min(1, t / .003) * vel * .12; put(i0 + k, s * .8, s, .5); }
}
function typing(t0, t1) {
  for (let t = t0; t < t1; t += .065 + rnd() * .05) noiseHit(t, .03, .05 + rnd() * .03, { hp: .5, decay: 180, pan: -.2, send: .05, tone: 1 });
}

const silence = [-1, -1];
const S_ = id => SCENES[id][0], E_ = id => SCENES[id][1];

/* ------------------ Hook: dark, numbers land like heartbeats ------------------ */
pad(0, S_('job') + .6, [36, 43, 50, 52, 55], .07, 500, 2.5, 2, .6, t => 500 + 1600 * smooth(t / S_('job')));
sub(0.05, S_('job'), 24, .12);
const hits = [LN.h1.a - .1, at('h1', 'Forty-two') - .1, at('h2', 'about as well')];
hits.forEach((h, i) => { kick(h, .7); sub(h, .9, 36 + [0, 3, 7][i], .35); epiano(h + .02, [72, 76, 79][i], 1.4, .34, [-.3, 0, .3][i], .7); });
whoosh(LN.h3.a - .9, 1.1, .18, true);
{ const TQ = at('h3', 'many of you'), span = LN.h3.b - TQ; for (let i = 0; i < 7; i++) blip(TQ + span * i / 7, .5, 84 + [0, 3, 7, 5, 10, 7, 12][i]); }
whoosh(LN.h4.a - .8, 1.0, .22, true);
{ const TT = LN.h4.a + .1; pad(TT, S_('job') + 1.5, [48, 55, 60, 64, 71], .07, 2200, .8, 2.5, .8); [60, 64, 67, 71, 74].forEach((m, i) => epiano(TT + i * .09, m, 2.5, .3, -.4 + i * .2, .7)); sub(TT, 3, 36, .18); }

/* ------------------ The groove ------------------ */
const BPM = 96, BEAT = 60 / BPM, BAR = 4 * BEAT;
const PROG = [
  { root: 36, ch: [48, 55, 59, 62, 64] },      // Cmaj9
  { root: 33, ch: [45, 52, 55, 59, 60] },      // Am9
  { root: 29, ch: [41, 48, 52, 55, 57, 64] },  // Fmaj9
  { root: 31, ch: [43, 50, 55, 57, 62] },      // G6/9
];
const G0 = S_('job') + .2, GEND = S_('close') + .3;
// intensity and whether drums play, by where we are in the story
function feel(t) {
  if (t >= S_('curve') - .3 && t < LN.d2.a - .1) return { k: .45, drums: false, drop: true };        // "the labels are the whole game"
  if (t >= S_('twist') - .2 && t < S_('recipe') - .2) return { k: .55, drums: false, twist: true };   // the dark reflection
  if (t < S_('result')) return { k: .62, drums: t > G0 + 2 * BAR };
  if (t < S_('curve')) return { k: .8, drums: true };
  if (t < S_('pays')) return { k: .95, drums: true, mel: t > S_('general') };
  if (t < S_('twist')) return { k: .88, drums: true };
  return { k: .85, drums: true, mel: true };
}
let bar = 0;
for (let t = G0; t < GEND - .01; t += BAR, bar++) {
  const c = PROG[bar % 4], f = feel(t + .01);
  pad(t, t + BAR, c.ch.slice(1, 4), (f.drop ? .07 : .05) * f.k, f.twist ? 1400 : 2400, .6, 1.2, .6);
  if (f.drop) { epiano(t, c.ch[2] + 12, BAR, .22, 0, .8); continue; }
  const pat = f.twist ? [0, -1, -1, 2, -1, -1, 3, -1] : [0, 2, 1, 3, 2, 4, 3, 1];
  for (let s = 0; s < 8; s++) {
    const tt = t + s * BEAT / 2; if (tt >= GEND || pat[s] < 0) continue;
    const note = c.ch[pat[(s + bar) % 8 < 0 ? 0 : (s + (f.twist ? 0 : bar)) % 8] % c.ch.length] + 12;
    epiano(tt + (s % 2 ? .012 : 0), note, BEAT * .9, (s % 4 === 0 ? .36 : .24) * f.k, s % 2 ? .3 : -.3, f.twist ? .6 : .35);
  }
  if (!f.twist) for (const [b, len] of [[0, 1.5], [1.5, .5], [2, 1.5], [3.5, .5]]) { const tt = t + b * BEAT; if (tt < GEND) sub(tt, len * BEAT * .95, c.root + 12 + (b === 3.5 ? 7 : 0), .16 * f.k); }
  else sub(t, BAR * .95, c.root + 12, .1);
  if (f.drums) for (let bt = 0; bt < 4; bt++) {
    const tt = t + bt * BEAT; if (tt >= GEND) continue;
    if (bt === 0 || bt === 2) kick(tt, .34 * f.k);
    if (bt === 1 || bt === 3) snap(tt, .11 * f.k);
    hat(tt + BEAT / 2, .042 * f.k);
    if (f.k > .9) { shaker(tt + BEAT / 4, .022); shaker(tt + 3 * BEAT / 4, .022); }
  }
  if (f.mel && bar % 2 === 0) for (const [b, m] of [[0, 79], [1.5, 76], [2, 74], [3, 72]]) epiano(t + b * BEAT, m, BEAT * .8, .15, .5, .6);
}
// Close: the resolve chord and a long tail.
const RES = LN.z1.a - .05;
pad(RES, DUR - 1.2, [48, 55, 60, 64, 67, 71], .085, 2200, .8, 3, .8);
[60, 64, 67, 71, 76].forEach((m, i) => epiano(RES + i * .11, m, 4, .3, -.4 + i * .2, .7));
sub(RES, 4.5, 36, .16);

/* ------------------ UI sounds, on the picture's events ------------------ */
const ticks = [
  ...[LN.w1.a, LN.w2.a, LN.w3.a],
  ...['ninety-one', 'Opus', 'Kimi', 'Jev'].map(w => at('r3', w) - .35),
  at('r2', 'forty-two'), at('a2', 'eleven'), at('a2', 'twenty-one'),
  at('d3', 'A hundred'), at('d3', 'A thousand'), at('d3', 'Two thousand'), at('d3', 'Four thousand'),
  at('t3', 'all nine'), at('t3', 'twenty-eight'), at('t3', 'fourteen'), at('t3', 'four providers'),
  at('s2', 'accuracy'), at('s4', 'ninety-six'), at('g2', 'six'), at('g3', 'tied'), at('p3', 'pays back'),
  at('k2', 'Write'), at('k2', 'Take'), at('k2', 'Have people'), LN.k3.a, LN.k4.a, at('k4', 'Compare'), LN.k5.a,
];
ticks.forEach((t, i) => blip(t, .42, 86 + [0, 2, 4, 7][i % 4]));
// job: each message lands in its box
{ const t0 = at('j2', 'card arrival'), t1 = at('j2', 'lost card'), t2 = at('j2', 'refund'); [t0, t1, t2].forEach(t => { tick(t - .05, .25, 2600); blip(t - .05, .35, 91); }); }
// a soft whoosh into every scene
for (const k of TL.order) if (k !== 'hook') whoosh(S_(k) - .3, .8, k === 'twist' || k === 'close' ? .14 : .07, true);
/* ------------------ Reverb and master ------------------ */
function reverb(inp, seedOff) {
  const out = new Float32Array(N);
  const combs = [1557, 1617, 1491, 1422, 1277, 1356].map(d => ({ d: Math.round((d + seedOff) * 1.9), buf: null, i: 0, f: 0 }));
  combs.forEach(c => { c.buf = new Float32Array(c.d); });
  const aps = [556, 441, 341].map(d => ({ d: Math.round((d + seedOff) * 1.9), buf: null, i: 0 }));
  aps.forEach(a => { a.buf = new Float32Array(a.d); });
  const fb = .86, damp = .3;
  for (let n = 0; n < N; n++) {
    const x = inp[n] * .12; let y = 0;
    for (const c of combs) { const o = c.buf[c.i]; c.f = o * (1 - damp) + c.f * damp; c.buf[c.i] = x + c.f * fb; c.i = (c.i + 1) % c.d; y += o; }
    for (const a of aps) { const o = a.buf[a.i]; const v = y + o * .5; a.buf[a.i] = v; y = o - v * .5; a.i = (a.i + 1) % a.d; }
    out[n] = y;
  }
  return out;
}
const WL = reverb(RL, 0), WR = reverb(RR, 23);
const out = new Int16Array(N * 2);
let peak = 0;
const mixL = new Float32Array(N), mixR = new Float32Array(N);
for (let n = 0; n < N; n++) {
  const t = n / SR;
  let gain = 1;
  if (t >= silence[0] && t < silence[1]) gain = 0;
  else if (t >= silence[1] && t < silence[1] + .05) gain = (t - silence[1]) / .05;
  const fadeOut = clamp((DUR - t) / 3.5);
  mixL[n] = (L[n] + WL[n]) * gain * fadeOut; mixR[n] = (R[n] + WR[n]) * gain * fadeOut;
  peak = Math.max(peak, Math.abs(mixL[n]), Math.abs(mixR[n]));
}
// Soft-clip, normalise to about −1 dBFS.
const norm = .89 / Math.tanh(peak * 1.1);
for (let n = 0; n < N; n++) {
  out[2 * n] = Math.round(Math.tanh(mixL[n] * 1.1) * norm * 32767);
  out[2 * n + 1] = Math.round(Math.tanh(mixR[n] * 1.1) * norm * 32767);
}
const hdr = Buffer.alloc(44);
hdr.write('RIFF', 0); hdr.writeUInt32LE(36 + out.byteLength, 4); hdr.write('WAVE', 8); hdr.write('fmt ', 12);
hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(2, 22); hdr.writeUInt32LE(SR, 24);
hdr.writeUInt32LE(SR * 4, 28); hdr.writeUInt16LE(4, 32); hdr.writeUInt16LE(16, 34); hdr.write('data', 36); hdr.writeUInt32LE(out.byteLength, 40);
mkdirSync(join(here, 'work'), { recursive: true });
writeFileSync(join(here, 'work', 'music.wav'), Buffer.concat([hdr, Buffer.from(out.buffer)]));
console.log('wrote work/music.wav', DUR, 's, peak before normalising', peak.toFixed(2));
