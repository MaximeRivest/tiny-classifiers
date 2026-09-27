// Load the shared timeline in Node: script.js, then the measured voice, then timeline.js.
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
export const here = dirname(fileURLToPath(import.meta.url));
const g = {};
for (const f of ['script.js', 'work/vo/durations.js', 'timeline.js'])
  new Function('globalThis', 'window', readFileSync(join(here, f), 'utf8'))(g, undefined);
export const { SCRIPT, VO, TL } = g;
