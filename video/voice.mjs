#!/usr/bin/env node
// Speak every line of script.js with Kokoro, check it by transcribing it back with Parakeet,
// and write work/vo/<id>.pcm (24 kHz mono s16le) and work/vo/durations.js.
//   node voice.mjs [--voice af_heart] [--speed 0.96] [--check]
// Lines are cached by their text, voice and speed: only changed lines are spoken again.
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const VOICE = opt('voice', 'af_heart'), SPEED = +opt('speed', '1.0');
const TTS = process.env.TTS || 'http://127.0.0.1:8880/tts', STT = process.env.STT || 'http://127.0.0.1:8078/transcribe';
const g = {}; new Function('globalThis', 'window', readFileSync(join(here, 'script.js'), 'utf8'))(g, undefined);
const dir = join(here, 'work', 'vo'); mkdirSync(dir, { recursive: true });

// Trim silence at both ends (Kokoro adds a little); keep 40 ms of air.
function trim(pcm) {
  const s = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.length / 2), th = 300, pad = 960;
  let a = 0, b = s.length - 1;
  while (a < s.length && Math.abs(s[a]) < th) a++;
  while (b > a && Math.abs(s[b]) < th) b--;
  a = Math.max(0, a - pad); b = Math.min(s.length, b + pad);
  return Buffer.from(s.slice(a, b).buffer);
}
const words = s => s.toLowerCase().replace(/[’']/g, "'").replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(Boolean);

const durations = {}, report = [];
for (const scene of g.SCRIPT) for (const line of scene.lines) {
  const say = line.say || line.text;
  const key = createHash('sha1').update(`${VOICE}|${SPEED}|${say}`).digest('hex').slice(0, 10);
  const file = join(dir, `${line.id}.pcm`), keyFile = file + '.key';
  if (!existsSync(file) || !existsSync(keyFile) || readFileSync(keyFile, 'utf8') !== key) {
    const res = await fetch(TTS, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: say, voice: VOICE, speed: SPEED }) });
    if (!res.ok) throw new Error(`${line.id}: TTS ${res.status}`);
    writeFileSync(file, trim(Buffer.from(await res.arrayBuffer())));
    writeFileSync(keyFile, key);
    process.stdout.write(`spoke ${line.id}\n`);
  }
  const bytes = readFileSync(file).length;
  durations[line.id] = +(bytes / 2 / 24000).toFixed(3);
  if (args.includes('--check')) {
    const pcm16 = execFileSync('ffmpeg', ['-v', 'error', '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', file, '-ar', '16000', '-f', 's16le', '-'], { maxBuffer: 1 << 28 });
    const heard = await (await fetch(STT, { method: 'POST', headers: { 'content-type': 'audio/L16' }, body: pcm16 })).text();
    const want = words(line.text), got = words(heard);
    const missing = want.filter(w => !got.includes(w) && !/^\d/.test(w));
    report.push(`${line.id.padEnd(3)} ${missing.length ? 'CHECK ' + missing.join(',') : 'ok   '}  heard: ${heard}`);
  }
}
writeFileSync(join(dir, 'durations.js'), `// written by voice.mjs\n(function (r) { r.VO = ${JSON.stringify({ voice: VOICE, speed: SPEED, durations })}; })(typeof window !== 'undefined' ? window : globalThis);\n`);
const total = Object.values(durations).reduce((a, b) => a + b, 0);
console.log(`${Object.keys(durations).length} lines, ${total.toFixed(1)} s of speech, voice ${VOICE} at ${SPEED}`);
if (report.length) console.log(report.join('\n'));
