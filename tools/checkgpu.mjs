/**
 * Headless driver test for ErosionEngine.
 *
 * There is no GPU in this environment, so instead of pretending the driver is
 * verified we run it against a mock WebGPU device that enforces the parts of
 * the spec that actually bite:
 *
 *   - a bind group may not expose the same buffer as read-only-storage and
 *     storage simultaneously;
 *   - a buffer bound as storage must have been created with STORAGE usage,
 *     uniform with UNIFORM, copy src/dst likewise;
 *   - bind group entries must match their layout's binding numbers exactly;
 *   - writeBuffer must stay inside the destination;
 *   - copyBufferToBuffer must stay inside both buffers;
 *   - a destroyed buffer may not be used.
 *
 * On top of that it records the pass sequence, so we can assert the solver
 * actually does what the design claims: log2(n) jump-flood steps, a redistance
 * every `redistanceEvery` erosion steps, and so on.
 */

import { ErosionEngine, PARAMS_FLOATS } from '../src/gpu/erosionEngine.js';

let fail = 0;
const problems = [];
function check(cond, msg) { if (!cond) { problems.push(msg); fail++; } }

// ----------------------------------------------------------------- mock GPU
const U = {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
  INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256,
};
globalThis.GPUBufferUsage = U;
globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };

const log = [];
let liveBuffers = 0;

/**
 * Queue/encoder ordering model.
 *
 * queue.writeBuffer is a QUEUE operation: it takes effect in submission order,
 * NOT at the point in an encoder where you happen to call it. So a uniform
 * written between two dispatches recorded in the SAME command buffer is seen by
 * BOTH of them with the final value. That is invisible in a mock that only
 * checks validity, and it silently broke the jump flood (every stride read the
 * last jfaStep written, collapsing an O(log n) flood to single-cell
 * propagation). We model real ordering: uniform state is snapshotted when a
 * command buffer is FINISHED, and dispatches record the value they will
 * actually observe.
 */
const uniformState = new Map();     // buffer id -> { offset -> value }
const observed = [];                // { pipe, jfaStep }


class MockBuffer {
  constructor(desc, id) {
    this.size = desc.size; this.usage = desc.usage; this.id = id;
    this.destroyed = false; this._mapped = null;
    liveBuffers++;
  }
  destroy() { if (!this.destroyed) liveBuffers--; this.destroyed = true; }
  async mapAsync() {
    check(!this.destroyed, `mapAsync on destroyed buffer ${this.id}`);
    check(this.usage & U.MAP_READ, `mapAsync on buffer ${this.id} without MAP_READ`);
    this._mapped = new ArrayBuffer(this.size);
  }
  getMappedRange() { check(this._mapped, `getMappedRange without map (${this.id})`); return this._mapped; }
  unmap() { this._mapped = null; }
}

class MockPass {
  constructor(kind) { this.kind = kind; this.groups = {}; }
  setPipeline(p) { this.pipeline = p; }
  setBindGroup(i, g) {
    check(g, `setBindGroup(${i}) with null group`);
    this.groups[i] = g;
    // every buffer referenced must be alive
    for (const e of g.entries) {
      check(!e.resource.buffer.destroyed,
        `pass ${this.pipeline?.label}: binding ${e.binding} uses destroyed buffer`);
    }
  }
  dispatchWorkgroups(x, y, z) {
    const rec = { op: 'dispatch', pipe: this.pipeline?.label, x, y, z, _enc: this._enc };
    log.push(rec);
    this._enc._recorded.push(rec);
  }
  draw(n) { log.push({ op: 'draw', n }); }
  end() {}
}

class MockEncoder {
  constructor() { this._recorded = []; }
  beginComputePass() { const p = new MockPass('compute'); p._enc = this; return p; }
  beginRenderPass(d) {
    check(d.colorAttachments?.[0]?.view, 'render pass without a view');
    return new MockPass('render');
  }
  copyBufferToBuffer(src, so, dst, doff, size) {
    check(!src.destroyed && !dst.destroyed, 'copy with destroyed buffer');
    check(src.usage & U.COPY_SRC, `copy source ${src.id} lacks COPY_SRC`);
    check(dst.usage & U.COPY_DST, `copy dest ${dst.id} lacks COPY_DST`);
    check(so + size <= src.size, `copy overruns source ${src.id}: ${so}+${size} > ${src.size}`);
    check(doff + size <= dst.size, `copy overruns dest ${dst.id}: ${doff}+${size} > ${dst.size}`);
    log.push({ op: 'copy', size });
  }
  clearBuffer(b, off, size) {
    check(!b.destroyed, 'clearBuffer on destroyed buffer');
    check(off + size <= b.size, 'clearBuffer overruns');
  }
  finish() {
    // Snapshot the uniform values these commands will actually see: the state
    // as of submission, not as of recording.
    const snap = new Map();
    for (const [id, m] of uniformState) snap.set(id, new Map(m));
    return { _cmd: true, _recorded: this._recorded, _snap: snap };
  }
}

let nextId = 0;
let pipeLabel = 'compute';
const device = {
  limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 },
  createBuffer: (d) => new MockBuffer(d, `buf${nextId++}`),
  createBindGroupLayout: (d) => {
    // Per-layout resource counts, checked against the guaranteed minimums.
    let storage = 0, uniform = 0;
    for (const e of d.entries) {
      if (!e.buffer) continue;
      if (e.buffer.type === 'uniform') uniform++; else storage++;
    }
    return { _layout: true, entries: d.entries, storage, uniform };
  },
  createPipelineLayout: (d) => {
    // WebGPU's guaranteed minimums, summed across ALL bind groups in the
    // layout, because the limit is per SHADER STAGE not per group. Exceeding
    // one makes pipeline creation fail, and the only symptom you get later is
    // "Invalid ComputePipeline ... is invalid due to a previous error" at
    // setPipeline time -- which names neither the limit nor the pipeline.
    const LIMITS = { storage: 8, uniform: 12 };
    let storage = 0, uniform = 0;
    for (const g of d.bindGroupLayouts) { storage += g.storage; uniform += g.uniform; }
    check(storage <= LIMITS.storage,
      `pipeline needs ${storage} storage buffers per stage, guaranteed max is ${LIMITS.storage}`);
    check(uniform <= LIMITS.uniform,
      `pipeline needs ${uniform} uniform buffers per stage, guaranteed max is ${LIMITS.uniform}`);
    return { _pl: true, groups: d.bindGroupLayouts, storage, uniform };
  },
  createShaderModule: (d) => {
    check(typeof d.code === 'string' && d.code.length > 500, 'shader module code too short');
    check(!d.code.includes('undefined'), 'shader source contains the literal "undefined"');
    pipeLabel = d.label || 'compute';
    return { _mod: true, code: d.code, label: d.label };
  },
  createComputePipeline: (d) => ({ label: pipeLabel, layout: d.layout }),
  createRenderPipeline: (d) => ({ label: 'render', layout: d.layout }),
  createBindGroup: (d) => {
    const layout = d.layout;
    const want = new Set(layout.entries.map((e) => e.binding));
    const got = new Set(d.entries.map((e) => e.binding));
    check(want.size === got.size && [...want].every((b) => got.has(b)),
      `bind group binding mismatch: layout [${[...want]}] vs group [${[...got]}]`);

    const byBinding = new Map(layout.entries.map((e) => [e.binding, e.buffer?.type]));
    const reads = new Set(), writes = new Set();
    for (const e of d.entries) {
      const type = byBinding.get(e.binding);
      const buf = e.resource.buffer;
      check(buf && !buf.destroyed, `binding ${e.binding} references a dead buffer`);
      if (type === 'uniform') {
        check(buf.usage & U.UNIFORM, `binding ${e.binding} needs UNIFORM usage`);
      } else if (type) {
        check(buf.usage & U.STORAGE, `binding ${e.binding} needs STORAGE usage`);
        if (type === 'storage') writes.add(buf.id); else reads.add(buf.id);
      }
    }
    for (const w of writes) {
      check(!reads.has(w), `bind group aliases ${w} as both read-only-storage and storage`);
    }
    return { entries: d.entries, layout };
  },
  queue: {
    writeBuffer: (b, off, data, dOff = 0, sz) => {
      if (b.usage & U.UNIFORM) {
        let m = uniformState.get(b.id);
        if (!m) { m = new Map(); uniformState.set(b.id, m); }
        const view = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
        const u32 = new Uint32Array(view.buffer, view.byteOffset + dOff,
          Math.floor((sz ?? (view.byteLength - dOff)) / 4));
        for (let i = 0; i < u32.length; i++) m.set(off + i * 4, u32[i]);
      }
      check(!b.destroyed, 'writeBuffer on destroyed buffer');
      check(b.usage & U.COPY_DST, `writeBuffer target ${b.id} lacks COPY_DST`);
      const bytes = sz ?? (data.byteLength - dOff);
      check(off + bytes <= b.size,
        `writeBuffer overruns ${b.id}: ${off}+${bytes} > ${b.size}`);
    },
    submit: (cmds) => {
      check(Array.isArray(cmds) && cmds.length, 'empty submit');
      for (const c of cmds) {
        for (const rec of c._recorded || []) {
          // jfaStep lives at byte offset 8 of Params
          const paramsId = [...(c._snap?.keys() || [])][0];
          const v = c._snap?.get(paramsId)?.get(8);
          if (rec.pipe === 'JFA_STEP') observed.push(v);
        }
      }
    },
  },
  createCommandEncoder: () => new MockEncoder(),
  addEventListener: () => {},
};

// -------------------------------------------------------------------- run
const engine = new ErosionEngine(device, null, 'bgra8unorm');

const P = {
  seed: 3, lithology: 'biotite-granite', jointStyle: 'orthogonal',
  resolution: 64, size: 1.0, aspectVariation: 0.28, jointRoughness: 1.0,
  hurst: 0.8, sheetingCurvature: 0.0, microReliefAmount: 1.0,
  weathering: { years: 0.7 },
};

log.length = 0;
const info = engine.reset(P);
const initLog = log.slice();

const jfaSteps = initLog.filter((e) => e.op === 'dispatch' && e.pipe === 'compute').length;
const expectedJfa = Math.ceil(Math.log2(64)) + 1;   // strides 64..1
check(info.totalSteps > 0, 'totalSteps must be positive');
check(info.faces >= 6, `expected >=6 joint faces, got ${info.faces}`);
check(Number.isFinite(info.dt) && info.dt > 0, `dt must be finite and positive, got ${info.dt}`);

// dispatch grid must cover the domain
const disp = initLog.find((e) => e.op === 'dispatch');
check(disp && disp.x === 16 && disp.y === 16 && disp.z === 16,
  `dispatch should be 16^3 for n=64/wg=4, got ${disp && [disp.x, disp.y, disp.z]}`);

// uniform buffer must be exactly the reflected struct size
check(engine.params.length === PARAMS_FLOATS, 'params array size mismatch');
check(engine.buf.params.size === PARAMS_FLOATS * 4, 'params buffer size mismatch');

// every parameter must be finite -- a single NaN poisons the whole shader
const bad = [];
engine.params.forEach((v, i) => { if (!Number.isFinite(v)) bad.push(i); });
check(bad.length === 0, `non-finite params at float indices [${bad}]`);

// the modal CDF must be monotone and end at 1
const cdf = Array.from({ length: 6 }, (_, i) => engine.params[128 + i * 4]);
let mono = true;
for (let i = 1; i < 6; i++) if (cdf[i] < cdf[i - 1] - 1e-6) mono = false;
check(mono, `modeCDF not monotone: ${cdf.map((v) => v.toFixed(3))}`);
check(Math.abs(cdf[5] - 1) < 1e-6, `modeCDF must end at 1, got ${cdf[5]}`);

// ---- resource limits ----------------------------------------------------
// (createPipelineLayout above asserts these as each pipeline is built.)

// ---- jump flood must see every stride -----------------------------------
// This is the bug that produced a black screen: all strides collapsed to 1.
{
  const strides = observed.filter((v) => v !== undefined);
  const uniq = [...new Set(strides)].sort((a, b) => b - a);
  const wantStrides = [];
  for (let st = 1 << Math.ceil(Math.log2(64)); st >= 1; st >>= 1) wantStrides.push(st);
  check(uniq.length === wantStrides.length,
    `jump flood must run each stride with its own jfaStep: saw [${uniq}], expected [${wantStrides}]`);
  check(uniq[0] === 64 && uniq[uniq.length - 1] === 1,
    `jump flood strides must span 64..1, got [${uniq}]`);
}

// ---- stepping -----------------------------------------------------------
log.length = 0;
const taken = engine.advance(24);
check(taken === Math.min(24, info.totalSteps), `advance(24) took ${taken}`);
const stepDispatches = log.filter((e) => e.op === 'dispatch').length;
check(stepDispatches > taken, 'expected redistance dispatches interleaved with steps');

// run to completion; must terminate and must not exceed the budget
let guard = 0;
while (!engine.done && guard++ < 5000) engine.advance(64);
check(engine.done, 'solver did not terminate');
check(engine.step <= info.totalSteps, `overran budget: ${engine.step} > ${info.totalSteps}`);
const finalStep = engine.step;
check(finalStep === info.totalSteps,
  `solver should reach the full budget without the volume guard firing: ${finalStep}/${info.totalSteps}`);

// ---- render -------------------------------------------------------------
log.length = 0;
engine.updateCamera({
  invViewProj: new Float32Array(16).fill(0.5), eye: [1, 1, 1],
  sun: [0.45, 0.78, 0.44], aspect: 0.002, time: 0, maxSteps: 160,
});
engine.render({ _view: true });
check(log.some((e) => e.op === 'draw' && e.n === 3), 'render must draw a 3-vertex fullscreen triangle');

// ---- resolution change must not leak ------------------------------------
const before = liveBuffers;
engine.reset({ ...P, resolution: 48 });
check(engine.n === 48, 'resolution change not applied');
check(liveBuffers <= before + 1, `buffer leak on resize: ${before} -> ${liveBuffers}`);
engine.destroy();

// ---- all lithologies and joint styles must pack cleanly ------------------
for (const litho of ['biotite-granite', 'pink-porphyritic', 'granodiorite',
                     'leucogranite', 'gneissic-granite']) {
  for (const style of ['orthogonal', 'sheeting', 'columnar', 'polyhedral', 'conjugate']) {
    const e2 = new ErosionEngine(device, null, 'bgra8unorm');
    try {
      const r = e2.reset({ ...P, resolution: 32, lithology: litho, jointStyle: style });
      const nan = [...e2.params].some((v) => !Number.isFinite(v));
      check(!nan, `${litho}/${style}: non-finite param`);
      check(r.faces >= 6, `${litho}/${style}: only ${r.faces} faces`);
      e2.advance(4);
    } catch (err) {
      check(false, `${litho}/${style}: ${err.message}`);
    }
    e2.destroy();
  }
}

// ------------------------------------------------------------------ report
if (problems.length) {
  console.log('FAILURES:');
  for (const p of [...new Set(problems)]) console.log('  - ' + p);
} else {
  console.log(`[ OK ] reset: ${info.faces} faces, ${info.totalSteps} steps, dt=${info.dt.toExponential(3)}`);
  console.log(`[ OK ] jump flood: ${expectedJfa} strides for n=64`);
  console.log(`[ OK ] params: ${PARAMS_FLOATS} floats, all finite, CDF monotone to 1`);
  console.log(`[ OK ] solver ran to completion: ${finalStep}/${info.totalSteps} steps`);
  console.log('[ OK ] 25 lithology x joint-style combinations packed and stepped');
  console.log('[ OK ] no bind-group aliasing, no buffer overruns, no use-after-destroy');
}
console.log(problems.length ? `\nFAIL (${problems.length})` : '\nPASS');
process.exit(problems.length ? 1 : 0);
