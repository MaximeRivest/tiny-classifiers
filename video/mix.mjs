#!/usr/bin/env node
// Mix the narration over the music: work/mix.wav (48 kHz stereo).
// The music ducks under the voice (−11 dB, 0.25 s in, 0.6 s out) and comes back up in the pauses.
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { here, TL, SCRIPT } from './load.mjs';

const SR = 48000, N = Math.ceil(TL.duration * SR);
const wav = f => { const b = readFileSync(f); return new Int16Array(b.buffer, b.byteOffset + 44, (b.length - 44) >> 1); };
const music = wav(join(here, 'work', 'music.wav'));
const voice = new Float32Array(N);
for (const sc of SCRIPT) for (const ln of sc.lines) {
  // Kokoro speaks at 24 kHz: resample to 48 kHz with ffmpeg's soxr and a gentle voice EQ.
  const pcm = execFileSync('ffmpeg', ['-v', 'error', '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', join(here, 'work', 'vo', ln.id + '.pcm'),
    '-af', 'aresample=48000:resampler=soxr,highpass=f=80,equalizer=f=3000:t=q:w=1:g=2,acompressor=threshold=-20dB:ratio=2.5:attack=5:release=80:makeup=2',
    '-f', 's16le', '-'], { maxBuffer: 1 << 28 });
  const s = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.length >> 1), i0 = Math.round(TL.L[ln.id].a * SR);
  for (let k = 0; k < s.length && i0 + k < N; k++) voice[i0 + k] += s[k] / 32768;
}
// Ducking envelope from where lines are, not from the signal: smooth and predictable.
const duck = new Float32Array(N).fill(1), low = 10 ** (-11 / 20);
for (const { a, b } of Object.values(TL.L)) {
  const i0 = Math.round((a - .25) * SR), i1 = Math.round((b + .6) * SR);
  for (let i = Math.max(0, i0); i < Math.min(N, i1); i++) {
    const t = i / SR, g = t < a ? 1 - (1 - low) * (t - (a - .25)) / .25 : t > b ? low + (1 - low) * (t - b) / .6 : low;
    duck[i] = Math.min(duck[i], g);
  }
}
const out = new Int16Array(N * 2), VG = 1.0, MG = .85;
for (let i = 0; i < N; i++) {
  const v = voice[i] * VG;
  for (let c = 0; c < 2; c++) {
    const m = (music[2 * i + c] || 0) / 32768 * MG * duck[i];
    out[2 * i + c] = Math.max(-32768, Math.min(32767, Math.round((v + m) * 32767)));
  }
}
const hdr = Buffer.alloc(44);
hdr.write('RIFF', 0); hdr.writeUInt32LE(36 + out.byteLength, 4); hdr.write('WAVE', 8); hdr.write('fmt ', 12);
hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(2, 22); hdr.writeUInt32LE(SR, 24);
hdr.writeUInt32LE(SR * 4, 28); hdr.writeUInt16LE(4, 32); hdr.writeUInt16LE(16, 34); hdr.write('data', 36); hdr.writeUInt32LE(out.byteLength, 40);
const raw = join(here, 'work', 'mix-raw.wav');
writeFileSync(raw, Buffer.concat([hdr, Buffer.from(out.buffer)]));
// Loudness for the web: −16 LUFS, peaks under −1.5 dB.
execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', raw, '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-ar', '48000', join(here, 'work', 'mix.wav')]);
console.log('wrote work/mix.wav', TL.duration.toFixed(1), 's');
