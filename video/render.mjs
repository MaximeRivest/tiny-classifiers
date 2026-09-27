#!/usr/bin/env node
// Render film.html to video, frame by frame, with headless Chromium and ffmpeg.
//   node render.mjs --stills 3,10.5,27     one PNG per time, in work/
//   node render.mjs [--fps 60] [--jobs 8]  work/seg-*.mp4, then out/ (with music.wav if present)
//   node render.mjs --from 30 --to 40      render only part of the film (preview)
// CHROME=/path/to/chromium chooses the browser (default: nixpkgs' chromium).
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const FPS = +opt('fps', 60), JOBS = +opt('jobs', 8), SS = +opt('ss', 2);  // SS: supersampling factor (draw at 2x, shrink: no text jitter)
const work = join(here, 'work'); mkdirSync(work, { recursive: true });
const outDir = resolve(here, 'out'); mkdirSync(outDir, { recursive: true });
const url = pathToFileURL(join(here, 'film.html')).href + '?capture';

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  return join(execFileSync('nix', ['build', '--no-link', '--print-out-paths', 'nixpkgs#chromium'], { encoding: 'utf8' }).trim(), 'bin/chromium');
}
const CHROME = findChrome();

async function openPage() {
  const profile = mkdtempSync(join(tmpdir(), 'film-'));
  const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run', '--mute-audio',
    '--force-color-profile=srgb', '--font-render-hinting=none', `--user-data-dir=${profile}`, '--remote-debugging-port=0',
    '--allow-file-access-from-files', '--window-size=1920,1080', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  const wsUrl = await new Promise((ok, fail) => {
    let buf = '';
    chrome.stderr.on('data', d => { buf += d; const m = buf.match(/DevTools listening on (ws:\/\/\S+)/); if (m) ok(m[1]); });
    chrome.on('exit', c => fail(new Error('chromium exited ' + c + buf)));
  });
  const ws = new WebSocket(wsUrl);
  await new Promise(r => ws.addEventListener('open', r, { once: true }));
  let id = 0; const pending = new Map();
  ws.addEventListener('message', e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { const { ok, fail } = pending.get(m.id); pending.delete(m.id); m.error ? fail(new Error(m.error.message)) : ok(m.result); }
  });
  const send = (method, params = {}, sessionId) => new Promise((ok, fail) => {
    const i = ++id; pending.set(i, { ok, fail }); ws.send(JSON.stringify({ id: i, method, params, sessionId }));
  });
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  const S = (m, p) => send(m, p, sessionId);
  await S('Page.enable');
  await S('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: SS, mobile: false });
  await S('Page.navigate', { url });
  // wait for the page's own readiness promise
  for (let i = 0; i < 100; i++) {
    const r = await S('Runtime.evaluate', { expression: 'window.ready ? window.ready.then(() => true) : false', awaitPromise: true, returnByValue: true }).catch(() => null);
    if (r?.result?.value === true) break;
    await new Promise(r => setTimeout(r, 200));
  }
  const frame = async t => {
    const r = await S('Runtime.evaluate', { expression: `render(${t})`, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 600));
    const { data } = await S('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true });
    return Buffer.from(data, 'base64');
  };
  const close = () => { try { ws.close(); chrome.kill('SIGKILL'); } catch {} setTimeout(() => rmSync(profile, { recursive: true, force: true }), 500); };
  return { frame, close, S };
}

const { TL } = await import('./load.mjs');
const duration = TL.duration;

if (args.includes('--stills')) {
  const times = opt('stills').split(',').map(Number);
  const p = await openPage();
  for (const t of times) { const f = join(work, `still-${String(t).replace('.', '_')}.png`); writeFileSync(f, await p.frame(t)); if (SS !== 1) execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', f, '-vf', 'scale=1920:1080:flags=lanczos', f.replace('.png', '.jpg')]); console.log('still', t); }
  p.close(); process.exit(0);
}

const from = +opt('from', 0), to = +opt('to', duration);
const f0 = Math.round(from * FPS), f1 = Math.round(to * FPS), total = f1 - f0;
const per = Math.ceil(total / JOBS);
console.log(`${total} frames at ${FPS} fps, ${JOBS} workers, ${CHROME}`);
const t0 = Date.now(); let done = 0;
// Workers write lossless pieces; the one real encode happens at the end, so there are no quality seams between pieces.
const segs = [];
await Promise.all(Array.from({ length: JOBS }, async (_, j) => {
  const a = f0 + j * per, b = Math.min(f1, a + per);
  if (a >= b) return;
  const seg = join(work, `seg-${String(j).padStart(2, '0')}.mp4`); segs[j] = seg;
  const ff = spawn('ffmpeg', ['-v', 'warning', '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'png', '-i', '-',
    '-vf', 'scale=1920:1080:flags=lanczos,format=yuv420p', '-c:v', 'libx264', '-preset', 'ultrafast', '-qp', '0', seg], { stdio: ['pipe', 'inherit', 'inherit'] });
  const p = await openPage();
  for (let f = a; f < b; f++) {
    const png = await p.frame(f / FPS);
    if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once('drain', r));
    if (++done % 120 === 0) { const s = (Date.now() - t0) / 1000; console.log(`${done}/${total}  ${(done / s).toFixed(1)} fps  eta ${Math.round((total - done) / (done / s))} s`); }
  }
  p.close(); ff.stdin.end();
  await new Promise(r => ff.on('close', r));
}));

const list = join(work, 'segs.txt');
writeFileSync(list, segs.filter(Boolean).map(s => `file '${s}'`).join('\n'));
const name = opt('out', from === 0 && to === duration ? 'tiny-classifiers-long-version' : `preview-${from}-${to}`);
const video = join(outDir, name + '.mp4');
const music = join(here, 'work', 'mix.wav');
const audio = existsSync(music) ? ['-ss', String(from), '-t', String(to - from), '-i', music] : [];
execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, ...audio, '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-x264-params', 'ipratio=2.0:aq-mode=0:deblock=-2,-2', '-pix_fmt', 'yuv420p',  // keyframes as sharp as the rest: no text 'tick' every 2 s
  '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-g', '120',
  ...(audio.length ? ['-c:a', 'aac', '-b:a', '256k', '-shortest'] : []), '-movflags', '+faststart', video], { stdio: 'inherit' });
console.log('wrote', video, `in ${Math.round((Date.now() - t0) / 1000)} s`);
