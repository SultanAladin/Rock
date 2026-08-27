/**
 * Integration test for the WebGPU app shell.
 *
 * Boots src/app/gpuMain.js against mock DOM + mock WebGPU, then drives it the
 * way a user would: generate, play frames, scrub, change every slider, switch
 * lithology and joint style, export. Anything that throws, produces a NaN
 * uniform, or leaks GPU buffers fails the run.
 *
 * This exists because there is no browser here, so "it builds" says nothing
 * about whether the app runs.
 */

import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------- mock DOM
const listeners = new Map();
function mkEl(tag = 'div') {
  const e = {
    tagName: tag.toUpperCase(), children: [], style: {}, className: '', dataset: {},
    _text: '', _html: '', value: '', checked: false, disabled: false, title: '',
    clientWidth: 1280, clientHeight: 720, width: 1280, height: 720,
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    prepend(c) { this.children.unshift(c); return c; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); },
    addEventListener(t, f) { (listeners.get(this) || listeners.set(this, {}).get(this))[t] = f; },
    removeEventListener() {},
    setPointerCapture() {}, releasePointerCapture() {},
    getContext(kind) { return kind === 'webgpu' ? gpuContext : null; },
    querySelector() { return mkEl('span'); },
    querySelectorAll() { return { forEach() {} }; },
    classList: { add() {}, remove() {}, toggle() {} },
    focus() {}, click() { this._clicked = true; },
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); },
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); this.children = []; },
  };
  listeners.set(e, {});
  return e;
}
const byId = { view: mkEl('canvas'), panel: mkEl('aside'), stage: mkEl('main') };
globalThis.document = {
  getElementById: (id) => byId[id] || mkEl(),
  createElement: (t) => mkEl(t),
  body: mkEl('body'),
};
globalThis.location = { search: '' };
globalThis.devicePixelRatio = 1;
globalThis.URL = URL;
let exportedText = null;
globalThis.Blob = class {
  constructor(p) { this.parts = p; if (typeof p[0] === 'string') exportedText = p[0]; }
};
globalThis.__BUILD__ = 'test';
let rafQueue = [];
globalThis.requestAnimationFrame = (f) => { rafQueue.push(f); return rafQueue.length; };
globalThis.performance = globalThis.performance || { now: () => Date.now() };

// ---------------------------------------------------------------- mock GPU
const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
globalThis.GPUBufferUsage = U;
globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
globalThis.GPUMapMode = { READ: 1 };

let live = 0, dispatches = 0, draws = 0, mockTick = 0;
const errors = [];
class Buf {
  constructor(d, id) { this.size = d.size; this.usage = d.usage; this.id = id; live++; }
  destroy() { if (!this.destroyed) live--; this.destroyed = true; }
  async mapAsync() {
    this.m = new ArrayBuffer(this.size);
    if (this.size <= 16) {
      // Counter readback: pretend the solid is shrinking slowly, so the
      // survival guard is exercised rather than trivially satisfied.
      new Uint32Array(this.m)[0] = Math.max(1, Math.floor(46000 * Math.pow(0.999, mockTick++)));
      return;
    }
    // Field readback: return a real signed-distance field (a rounded block) so
    // the export path is contouring actual geometry rather than a zero buffer,
    // which would silently "pass" with an empty mesh.
    const f = new Float32Array(this.m);
    const n = Math.round(Math.cbrt(f.length));
    const extent = 0.95, h = (2 * extent) / (n - 1), r = 0.55, round = 0.12;
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = -extent + i * h, y = -extent + j * h, z = -extent + k * h;
      const qx = Math.abs(x) - r, qy = Math.abs(y) - r * 0.8, qz = Math.abs(z) - r * 0.92;
      const ox = Math.max(qx, 0), oy = Math.max(qy, 0), oz = Math.max(qz, 0);
      const outside = Math.hypot(ox, oy, oz);
      const inside = Math.min(Math.max(qx, Math.max(qy, qz)), 0);
      f[(k * n + j) * n + i] = outside + inside - round;
    }
  }
  getMappedRange() { return this.m; }
  unmap() { this.m = null; }
}
let bid = 0;
const pass = (label) => ({
  setPipeline(p) { this.p = p; },
  setBindGroup(i, g) { if (!g) errors.push('null bind group'); },
  dispatchWorkgroups() { dispatches++; },
  draw() { draws++; },
  end() {},
});
const encoder = () => ({
  beginComputePass: () => pass('c'),
  beginRenderPass: (d) => {
    if (!d.colorAttachments?.[0]?.view) errors.push('render pass without view');
    return pass('r');
  },
  copyBufferToBuffer(s, so, dd, doff, size) {
    if (s.destroyed || dd.destroyed) errors.push('copy w/ destroyed buffer');
    if (so + size > s.size || doff + size > dd.size) errors.push('copy overrun');
  },
  clearBuffer() {}, finish: () => ({}),
});
const device = {
  limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 },
  createBuffer: (d) => new Buf(d, bid++),
  createBindGroupLayout: (d) => ({ entries: d.entries }),
  createPipelineLayout: (d) => ({}),
  createShaderModule: (d) => {
    if (d.code.includes('undefined')) errors.push('shader contains "undefined"');
    return {};
  },
  createComputePipeline: () => ({}),
  createRenderPipeline: () => ({}),
  createBindGroup: (d) => {
    const t = new Map(d.layout.entries.map((e) => [e.binding, e.buffer?.type]));
    const r = new Set(), w = new Set();
    for (const e of d.entries) {
      const ty = t.get(e.binding);
      const b = e.resource.buffer;
      if (!b || b.destroyed) errors.push('dead buffer in bind group');
      if (ty === 'storage') w.add(b.id); else if (ty) r.add(b.id);
    }
    for (const x of w) if (r.has(x)) errors.push(`buffer ${x} bound read+write`);
    return {};
  },
  queue: {
    writeBuffer(b, off, data, dOff = 0, sz) {
      if (b.destroyed) errors.push('write to destroyed buffer');
      const n = sz ?? data.byteLength - dOff;
      if (off + n > b.size) errors.push(`writeBuffer overrun on ${b.id}`);
      const view = ArrayBuffer.isView(data) ? data : new Float32Array(data);
      if (view.some && [...view].some((v) => typeof v === 'number' && !Number.isFinite(v))) {
        errors.push('non-finite value written to a GPU buffer');
      }
    },
    submit() {},
  },
  createCommandEncoder: encoder,
  addEventListener() {},
};
const gpuContext = { configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) };
Object.defineProperty(globalThis, 'navigator', { configurable: true, writable: true, value: {
  gpu: {
    requestAdapter: async () => ({
      limits: device.limits,
      info: { vendor: 'mock', architecture: 'test' },
      requestDevice: async () => device,
    }),
    getPreferredCanvasFormat: () => 'bgra8unorm',
  },
  hardwareConcurrency: 8,
} });

// ------------------------------------------------------------------- boot
await import('../src/app/gpuMain.js');
await new Promise((r) => setTimeout(r, 30));

let fail = 0;
const ok = (c, m) => { if (!c) { console.log('[FAIL] ' + m); fail++; } else console.log('[ OK ] ' + m); };

ok(byId.panel.children.length > 3, `UI built (${byId.panel.children.length} root nodes)`);
ok(byId.stage.children.length === 0, 'no fatal error card was shown');
ok(dispatches > 0, `solver dispatched (${dispatches} compute dispatches during boot)`);

// run 120 frames
const runFrames = async (n) => {
  for (let i = 0; i < n; i++) {
    const q = rafQueue; rafQueue = [];
    for (const f of q) await f(performance.now() + i * 16.7);
  }
};
draws = 0;
await runFrames(120);
ok(draws > 0, `render loop drew ${draws} frames`);
ok(errors.length === 0, `no GPU validation errors (${[...new Set(errors)].join('; ')})`);
// Read the live solver readout the user sees, rather than trusting internals.
function findText(node, needle, acc = []) {
  if (node._html && node._html.includes(needle)) acc.push(node._html);
  for (const c of node.children || []) findText(c, needle, acc);
  return acc;
}
const readout = findText(byId.panel, 'iteration');
ok(readout.length > 0, 'solver readout is rendered in the panel');
const m = readout.join(' ').match(/iteration <b>(\d+)<\/b> \/ (\d+)/);
ok(m && +m[1] > 0, `solver advanced past iteration 0 (readout: ${m ? m[1] + '/' + m[2] : 'none'})`);
ok(m && +m[1] === +m[2], `solve ran to completion in the render loop (${m ? m[1] + '/' + m[2] : '?'})`);
ok(findText(byId.panel, 'complete').length > 0, 'readout reports completion, not an early stop');
ok(findText(byId.panel, 'stopped early').length === 0, 'volume guard did not fire spuriously');

// buffers must not grow without bound over a long run
const liveAfter = live;
await runFrames(120);
ok(live <= liveAfter, `no buffer leak across 240 frames (${liveAfter} -> ${live})`);


// ---------------------------------------------------------------- driving
// Walk the whole panel and operate every control, the way a user poking at the
// tool would. Anything that throws or writes a NaN uniform fails.
function walk(node, out = []) {
  out.push(node);
  for (const c of node.children || []) walk(c, out);
  return out;
}
const nodes = walk(byId.panel);
const ranges = nodes.filter((n) => n.tagName === 'INPUT' && n.type === 'range');
const selects = nodes.filter((n) => n.tagName === 'SELECT');
const buttons = nodes.filter((n) => n.tagName === 'BUTTON');
const chips = nodes.filter((n) => n.className && n.className.includes('chip'));

const before = errors.length;
let threw = 0;
// every slider: min, max, midpoint
for (const r of ranges) {
  const lo = parseFloat(r.min), hi = parseFloat(r.max);
  for (const v of [lo, hi, (lo + hi) / 2]) {
    r.value = String(v);
    try { r.oninput && r.oninput(); } catch (e) { threw++; console.log('  slider threw: ' + e.message); }
    try { r.onchange && r.onchange(); } catch (e) { threw++; console.log('  slider threw: ' + e.message); }
  }
  await runFrames(2);
}
ok(threw === 0, `every slider driven to min/mid/max without throwing (${ranges.length} sliders)`);

// every dropdown option (lithology, joint style)
threw = 0;
for (const sel of selects) {
  for (const opt of sel.children) {
    sel.value = opt.value;
    try { sel.onchange && sel.onchange(); } catch (e) { threw++; console.log('  select threw: ' + e.message); }
    await runFrames(3);
  }
}
ok(threw === 0, `every lithology and joint style selected without throwing (${selects.length} dropdowns)`);

// chips: debug view modes, mixes, batch variants
threw = 0;
for (const c of chips) {
  try { c.onclick && c.onclick(); } catch (e) { threw++; console.log('  chip threw: ' + e.message); }
  await runFrames(2);
}
ok(threw === 0, `every chip toggled without throwing (${chips.length} chips)`);

// transport buttons: pause, step, restart
threw = 0;
for (const btn of buttons) {
  if (/OBJ|PLY/i.test(btn.textContent)) continue;   // exports handled below
  try { btn.onclick && btn.onclick(); } catch (e) { threw++; console.log('  button threw: ' + e.message); }
  await runFrames(2);
}
ok(threw === 0, `transport buttons operated without throwing (${buttons.length} buttons)`);

// scrubbing backwards must re-solve, not corrupt state
threw = 0;
const scrubEl = ranges.find((r) => r.className === 'scrub');
if (scrubEl) {
  for (const v of [1000, 250, 900, 0, 500]) {
    scrubEl.value = String(v);
    try { scrubEl.onchange(); } catch (e) { threw++; console.log('  scrub threw: ' + e.message); }
    await runFrames(2);
  }
}
ok(scrubEl && threw === 0, 'scrubbing forwards and backwards is safe');

ok(errors.length === before, `driving the UI produced no GPU errors (${[...new Set(errors.slice(before))].join('; ')})`);

// export path: readback + dual contouring + file write
let exported = null;
globalThis.URL.createObjectURL = () => 'blob:mock';
globalThis.URL.revokeObjectURL = () => {};
const objBtn = buttons.find((b) => b.textContent === 'OBJ');
if (objBtn) {
  try {
    await objBtn.onclick();
    await new Promise((r) => setTimeout(r, 50));
    exported = true;
  } catch (e) { console.log('  export threw: ' + e.message); exported = false; }
}
ok(exported === true, 'OBJ export completes (field readback -> dual contouring -> download)');
ok(typeof exportedText === 'string' && exportedText.startsWith('# boulder'),
   'exported OBJ has a header');
const vCount = (exportedText.match(/^v /gm) || []).length;
const fCount = (exportedText.match(/^f /gm) || []).length;
ok(vCount > 500, `exported mesh has real geometry (${vCount} vertices, ${fCount} faces)`);
const statLine = (exportedText.match(/^# volume.*$/m) || [''])[0];
const sphM = statLine.match(/sphericity ([\d.]+)/);
ok(sphM && +sphM[1] > 0.6 && +sphM[1] <= 1.0,
   `exported sphericity is physical: ${statLine.trim()}`);
ok(byId.stage.children.length === 0, 'still no fatal error card after driving everything');

console.log(`\ntotals: ${dispatches} dispatches, ${draws} draws, ${live} live buffers`);
console.log(`driven: ${ranges.length} sliders, ${selects.length} selects, ${chips.length} chips, ${buttons.length} buttons`);
if (errors.length) {
  console.log('errors:'); for (const e of new Set(errors)) console.log('  - ' + e);
}
console.log(fail ? `\nFAIL (${fail})` : '\nPASS');
process.exit(fail ? 1 : 0);
