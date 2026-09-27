// One timeline for the picture (film.html), the music (music.mjs) and the mix (mix.mjs),
// laid out from the narration: script.js (the words) + work/vo/durations.js (how long each takes).
// Every visual event is pinned to a word: at('r3', 'Opus') is the moment the voice says it.
(function (root) {
  const SCRIPT = root.SCRIPT, D = root.VO.durations;
  const L = {}, SCENES = {}, TEXT = {};
  const OVERLAP = 0.5;  // scenes cross-fade over this long
  let t = 0;
  for (const sc of SCRIPT) {
    const start = t;
    let c = start + sc.lead, lastEnd = c;
    for (const ln of sc.lines) {
      const d = D[ln.id];
      if (d == null) throw new Error(`no recording for line ${ln.id}: run voice.mjs`);
      L[ln.id] = { a: c, b: c + d, d };
      TEXT[ln.id] = ln;
      lastEnd = c + d;
      c += d + (ln.gap ?? 0.45);
    }
    const end = lastEnd + sc.tail;
    SCENES[sc.id] = [start, end];
    t = end - OVERLAP;
  }

  // When the voice reaches `frag` in line `id`, by its share of the spoken characters.
  // Digits are read as words, so fragments are searched in the spoken text first.
  function at(id, frag, off = 0) {
    const ln = TEXT[id];
    if (!ln) throw new Error(`no line ${id}`);
    const spoken = (ln.say || ln.text).replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
    let i = spoken.indexOf(frag), len = spoken.length;
    if (i < 0) { i = ln.text.indexOf(frag); len = ln.text.length; }
    if (i < 0) throw new Error(`"${frag}" is not in line ${id}`);
    return L[id].a + L[id].d * (i / len) + off;
  }
  const last = SCRIPT[SCRIPT.length - 1].id;
  root.TL = { SCENES, L, at, OVERLAP, duration: SCENES[last][1], order: SCRIPT.map(s => s.id) };
})(typeof window !== 'undefined' ? window : globalThis);
