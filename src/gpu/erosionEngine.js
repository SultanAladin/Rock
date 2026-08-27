/**
 * WebGPU erosion engine.
 *
 * The whole point of this class is that phi never leaves the GPU. The compute
 * passes write it, the render pass raymarches it, and the only readbacks are
 * (a) a 4-byte atomic cell count for the survival guard and the live volume
 * readout, and (b) an explicit, user-triggered export.
 *
 * That is what buys the realtime behaviour: a step is a few hundred
 * microseconds of compute and the display cost is one fullscreen pass, so we
 * can run N steps per frame and still present at vsync. There is no bake, no
 * mesh rebuild, and nothing blocking on the main thread.
 *
 * PASS ORDER PER SOLVE
 * --------------------
 *   INIT                        analytic joint block -> phi, phi0
 *   JFA_SEED, JFA_STEP xlog2(n), JFA_RESOLVE      -> metric phi
 *   SHELTER                                        -> occlusion
 *   [ loop ]  STEP  (x stepsPerFrame)
 *             every redistanceEvery: JFA_SEED/STEP/RESOLVE, SHELTER
 *             every 8: COUNT (survival guard)
 *   RETREAT   (on demand, for shading)
 */

import { INIT_WGSL, JFA_SEED_WGSL, JFA_STEP_WGSL, JFA_RESOLVE_WGSL,
         SHELTER_WGSL, STEP_WGSL, COUNT_WGSL, RETREAT_WGSL } from './wgsl/erode.wgsl.js';
import { RAYMARCH_WGSL } from './wgsl/raymarch.wgsl.js';
import { MINERAL_LIST, LITHOLOGIES, buildModeCDF } from '../core/petrology.js';
import { buildJointBlock } from '../core/joints.js';
import { DEFAULT_WEATHERING } from '../core/weathering.js';

const WG = 4;                       // workgroup_size(4,4,4)
export const PARAMS_FLOATS = 152;   // 608 bytes; pinned by tools/checkwgsl.mjs

// ---------------------------------------------------------------- offsets
// Byte offsets from the WGSL Params struct. Mirrored here rather than guessed;
// tools/checkwgsl.mjs re-derives them from the shader and fails on drift.
const O = {
  n: 0, seed: 1, jfaStep: 2, flags: 3,
  h: 4, origin: 5, extent: 6, dt: 7,
  spheroidal: 8, spheroidalPower: 9, cavernous: 10, cavernousPower: 11,
  uniformRate: 12, roundingRadius: 13, weakMin: 14, weakMax: 15,
  insolation: 16, moistureGradient: 17, buriedFraction: 18, grussification: 19,
  rindlet: 20, rindletSpacing: 21, heterogeneity: 22, heteroScale: 23,
  shelterRadius: 24, bandWidth: 25, minVolFrac: 26,
  sunDir: 28,
  cellSize: 32, cellSize2: 33, grainSigma: 34, seriate: 35,
  foliation: 36, phenFrac: 37, phenSize: 38, phenId: 39,
  weatherAge: 40, lichen: 41, caseHardening: 42, dust: 43,
  wetness: 44, microRelief: 45, retreatScale: 46, stainStrength: 47,
  stainColor: 48,
  exposure: 52, debugMode: 53,
  minAlbedoRough: 56, minProps: 80, minExtra: 104, modeCDF: 128,
};

const WEAK_MIN = 0.35, WEAK_MAX = 1.8;   // matches core/weathering.js

export class ErosionEngine {
  /**
   * @param {GPUDevice} device
   * @param {GPUCanvasContext} context
   * @param {GPUTextureFormat} format
   */
  constructor(device, context, format) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.n = 0;
    this.step = 0;
    this.totalSteps = 0;
    this.done = false;
    this.stoppedEarly = false;
    this.initialInside = 0;
    this.lastInside = 0;
    this._buildPipelines();
  }

  // ------------------------------------------------------------- pipelines
  _buildPipelines() {
    const d = this.device;

    // Compute layout: one group with every buffer the passes might touch, so a
    // single bind group serves all of them and pass switching is free.
    const ro = { type: 'read-only-storage' };
    const rw = { type: 'storage' };
    this.computeLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: ro },
      ],
    });
    this.faceLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const mk = (code, layouts) => d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: layouts }),
      compute: { module: d.createShaderModule({ code }), entryPoint: 'main' },
    });
    const only = [this.computeLayout];
    this.pipe = {
      init:    mk(INIT_WGSL, [this.computeLayout, this.faceLayout]),
      seed:    mk(JFA_SEED_WGSL, only),
      jfa:     mk(JFA_STEP_WGSL, only),
      resolve: mk(JFA_RESOLVE_WGSL, only),
      shelter: mk(SHELTER_WGSL, only),
      step:    mk(STEP_WGSL, only),
      count:   mk(COUNT_WGSL, only),
      retreat: mk(RETREAT_WGSL, only),
    };

    // Render layout
    this.renderLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: ro },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: ro },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: ro },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const rmod = d.createShaderModule({ code: RAYMARCH_WGSL });
    this.renderPipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
      vertex: { module: rmod, entryPoint: 'vs' },
      fragment: { module: rmod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  // --------------------------------------------------------------- buffers
  _allocate(n) {
    if (this.n === n) return;
    this._free();
    const d = this.device;
    const N3 = n * n * n;
    const S = GPUBufferUsage.STORAGE;
    const f32 = N3 * 4;

    this.n = n;
    this.buf = {
      params:  d.createBuffer({ size: PARAMS_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      camera:  d.createBuffer({ size: 7 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      phiA:    d.createBuffer({ size: f32, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
      phiB:    d.createBuffer({ size: f32, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
      phi0:    d.createBuffer({ size: f32, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
      shelter: d.createBuffer({ size: f32, usage: S | GPUBufferUsage.COPY_SRC }),
      retreat: d.createBuffer({ size: f32, usage: S | GPUBufferUsage.COPY_SRC }),
      seedA:   d.createBuffer({ size: N3 * 16, usage: S }),
      seedB:   d.createBuffer({ size: N3 * 16, usage: S }),
      counter: d.createBuffer({ size: 16, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
      readback: d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
    this.params = new Float32Array(PARAMS_FLOATS);
    this.paramsU32 = new Uint32Array(this.params.buffer);
    this.camera = new Float32Array(28);
    this.wg = Math.ceil(n / WG);
  }

  _free() {
    if (!this.buf) return;
    for (const b of Object.values(this.buf)) b.destroy?.();
    this.buf = null;
    this.n = 0;
  }

  destroy() {
    this._free();
    if (this.exportBuf) { this.exportBuf.destroy(); this.exportBuf = null; }
  }

  /**
   * Compute bind group. Every pass declares all eight bindings, so we just
   * choose which buffer plays which role for the pass at hand.
   */
  _bind({ phiIn, phiOut, seedIn, seedOut, aux, auxOut }) {
    const b = this.buf;
    phiIn  = phiIn  || b.phiA;  phiOut  = phiOut  || b.phiB;
    seedIn = seedIn || b.seedA; seedOut = seedOut || b.seedB;
    aux    = aux    || b.phi0;  auxOut  = auxOut  || b.shelter;

    // WebGPU rejects a bind group that exposes one buffer as both
    // read-only-storage and storage. That is easy to do by accident with
    // defaulted slots, and the failure is a validation error at draw time
    // rather than anything visible, so assert it here where it is debuggable.
    const reads = [phiIn, seedIn, aux, b.phi0];
    const writes = [phiOut, seedOut, auxOut, b.counter];
    for (const w of writes) {
      if (reads.includes(w)) {
        throw new Error('ErosionEngine: bind group aliases a buffer as both read and write');
      }
    }

    return this.device.createBindGroup({
      layout: this.computeLayout,
      entries: [
        { binding: 0, resource: { buffer: b.params } },
        { binding: 1, resource: { buffer: phiIn } },
        { binding: 2, resource: { buffer: phiOut } },
        { binding: 3, resource: { buffer: seedIn } },
        { binding: 4, resource: { buffer: seedOut } },
        { binding: 5, resource: { buffer: aux } },
        { binding: 6, resource: { buffer: auxOut } },
        { binding: 7, resource: { buffer: b.counter } },
        { binding: 8, resource: { buffer: b.phi0 } },
      ],
    });
  }

  _dispatch(enc, pipeline, groups, extra) {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, groups);
    if (extra) pass.setBindGroup(1, extra);
    pass.dispatchWorkgroups(this.wg, this.wg, this.wg);
    pass.end();
  }

  // ------------------------------------------------------------ parameters
  /**
   * Pack the Params uniform. Everything the compute passes and the raymarcher
   * need lives in one struct so a slider change is a single 608-byte upload.
   */
  _packParams(P) {
    const f = this.params, u = this.paramsU32;
    const litho = LITHOLOGIES[P.lithology] || LITHOLOGIES['biotite-granite'];
    const W = { ...DEFAULT_WEATHERING, ...(P.weathering || {}) };
    const n = P.resolution;
    const extent = P.size * 0.95;
    const h = (2 * extent) / (n - 1);

    u[O.n] = n;
    u[O.seed] = (P.seed * 7919 + 13) >>> 0;
    u[O.jfaStep] = 1;
    u[O.flags] = 0;
    f[O.h] = h;
    f[O.origin] = -extent;
    f[O.extent] = extent;

    f[O.spheroidal] = W.spheroidal;
    f[O.spheroidalPower] = W.spheroidalPower;
    f[O.cavernous] = W.cavernous;
    f[O.cavernousPower] = W.cavernousPower;
    f[O.uniformRate] = W.uniform;
    f[O.roundingRadius] = W.roundingRadius;
    f[O.weakMin] = WEAK_MIN;
    f[O.weakMax] = WEAK_MAX;
    f[O.insolation] = W.insolation;
    f[O.moistureGradient] = W.moistureGradient;
    f[O.buriedFraction] = W.buriedFraction;
    f[O.grussification] = W.grussification;
    f[O.rindlet] = W.rindlet;
    f[O.rindletSpacing] = W.rindletSpacing;
    f[O.heterogeneity] = W.heterogeneity;
    f[O.heteroScale] = W.heteroScale;
    f[O.shelterRadius] = W.shelterRadius;
    f[O.bandWidth] = W.bandWidth;
    f[O.minVolFrac] = W.minVolumeFraction;

    const s = W.sunDir, sl = Math.hypot(s[0], s[1], s[2]) || 1;
    f[O.sunDir] = s[0] / sl; f[O.sunDir + 1] = s[1] / sl; f[O.sunDir + 2] = s[2] / sl;

    // ---- timestep + budget: identical derivation to core/weathering.js -----
    // The rate law saturates, so Fmax is a real parameter-independent bound and
    // the advective CFL is honest. The parabolic bound is evaluated at the
    // curvature where the diffusivity actually peaks (k ~ 1), not at grid
    // resolution, where it has already decayed by 1/k^2.
    const L = extent;
    const Fmax = (W.spheroidal + W.cavernous + W.uniform) * WEAK_MAX;
    const dtAdv = (0.45 * h) / Math.max(1e-6, Fmax);
    const p = W.spheroidalPower;
    const kStar = p > 1 ? Math.pow((p - 1) / (p + 1), 1 / p) : 1.0;
    const k = Math.max(1e-3, kStar);
    const kp2 = Math.pow(k, p);
    const bSup = (p * Math.pow(k, p - 1)) / ((1 + kp2) * (1 + kp2));
    const bEff = W.spheroidal * bSup * (L * W.roundingRadius) * WEAK_MAX;
    const dtDiff = (0.25 * h * h) / Math.max(1e-9, bEff);
    const dt = Math.min(dtAdv, dtDiff);
    f[O.dt] = dt;

    // `years` is a RETREAT DISTANCE, not a time. Converting through the corner
    // speed is what keeps the requested rounding independent of the rate knobs.
    const Fcorner = Math.max(1e-6, W.spheroidal * 0.5 + W.cavernous * 0.15 + W.uniform);
    const budget = (W.years * L * 0.12) / Fcorner;
    this.totalSteps = Math.max(1, Math.min(W.maxSteps, Math.ceil(budget / dt)));
    this.redistanceEvery = W.redistanceEvery;

    // ---- crystal aggregate -------------------------------------------------
    const cell = litho.grain * 1.15;
    f[O.cellSize] = cell;
    f[O.cellSize2] = cell * 0.42;
    f[O.grainSigma] = litho.grainSigma;
    f[O.seriate] = litho.seriate;
    f[O.foliation] = litho.foliation || 0;
    const phen = litho.phenocryst || { frac: 0, size: 0.02, mineral: 'kfeldspar' };
    f[O.phenFrac] = phen.frac;
    f[O.phenSize] = phen.size;
    const pm = MINERAL_LIST.find((m) => m.name === phen.mineral);
    f[O.phenId] = pm ? pm.id : 1;

    // ---- shading -----------------------------------------------------------
    const S = P.shading || {};
    f[O.weatherAge] = S.weatherAge ?? Math.min(1, W.years);
    f[O.lichen] = S.lichen ?? 0.25;
    f[O.caseHardening] = S.caseHardening ?? 0.4;
    f[O.dust] = S.dust ?? 0.3;
    f[O.wetness] = S.wetness ?? 0.0;
    f[O.microRelief] = S.microRelief ?? (P.microReliefAmount ?? 1.0);
    f[O.retreatScale] = S.retreatScale ?? 8.0;
    f[O.stainStrength] = litho.stainStrength ?? 0.7;
    const st = litho.stain || [0.42, 0.24, 0.10];
    f[O.stainColor] = st[0]; f[O.stainColor + 1] = st[1]; f[O.stainColor + 2] = st[2];
    f[O.exposure] = S.exposure ?? 1.0;
    f[O.debugMode] = S.debugMode ?? 0;

    // ---- mineral tables (vec4 x 6) -----------------------------------------
    for (const m of MINERAL_LIST) {
      const o = m.id * 4;
      f[O.minAlbedoRough + o + 0] = m.albedo[0];
      f[O.minAlbedoRough + o + 1] = m.albedo[1];
      f[O.minAlbedoRough + o + 2] = m.albedo[2];
      f[O.minAlbedoRough + o + 3] = m.rough;
      f[O.minProps + o + 0] = m.spec;
      f[O.minProps + o + 1] = m.hardness;
      f[O.minProps + o + 2] = m.durability;
      f[O.minProps + o + 3] = m.cleavage;
      f[O.minExtra + o + 0] = m.fe;
      f[O.minExtra + o + 1] = m.translucency;
    }
    // CDF: the shader scans .x of six vec4s, so expand the 256-entry LUT back
    // into per-mineral cumulative thresholds.
    const cdf = buildModeCDF(litho.mode);
    const cum = cdfThresholds(cdf, litho.mode);
    for (let i = 0; i < 6; i++) f[O.modeCDF + i * 4] = cum[i];

    this.device.queue.writeBuffer(this.buf.params, 0, this.params);
    this.meta = { n, h, extent, dt, litho, W, size: P.size };
  }

  setUniform(name, value) {
    if (!(name in O) || !this.buf) return;
    this.params[O[name]] = value;
    this.device.queue.writeBuffer(this.buf.params, O[name] * 4,
      this.params.buffer, O[name] * 4, 4);
  }

  _setU32(name, value) {
    this.paramsU32[O[name]] = value >>> 0;
    this.device.queue.writeBuffer(this.buf.params, O[name] * 4,
      this.params.buffer, O[name] * 4, 4);
  }

  // ------------------------------------------------------------------ init
  /** Build the fresh block and put a metric SDF on the GPU. Sub-millisecond. */
  reset(P) {
    this._allocate(P.resolution);
    this._packParams(P);

    // Joint faces: geometry comes from the same CPU generator as before (it is
    // a handful of planes, not a grid, so there is nothing to gain from moving
    // it), but the SDF *evaluation* over 64^3 cells happens on the GPU.
    const litho = this.meta.litho;
    const rngSeed = P.seed;
    const av = P.aspectVariation ?? 0.28;
    const aspect = aspectFor(rngSeed, av);
    const block = buildJointBlock({
      seed: rngSeed,
      style: P.jointStyle,
      size: P.size,
      aspect,
      jointRoughness: P.jointRoughness,
      hurst: P.hurst,
      grainSize: litho.grain,
      sheetingCurvature: P.sheetingCurvature,
    });
    this._uploadFaces(block.faces);

    const enc = this.device.createCommandEncoder();
    // INIT writes phi -> phiA and phi0.
    // NB: phiIn must NOT be phiA here. WebGPU forbids binding one buffer as
    // both read-only-storage and storage in the same bind group, so the unused
    // read slot gets the scratch buffer.
    // INIT writes the fresh block into phiA. phi0 is bound read-only at slot 8
    // for every pass, so it cannot also be a write target here; we snapshot it
    // with a device-side copy, which is cheaper than a second dispatch anyway.
    this._dispatch(enc, this.pipe.init,
      this._bind({ phiIn: this.buf.phiB, phiOut: this.buf.phiA, auxOut: this.buf.shelter }),
      this.faceGroup);
    enc.copyBufferToBuffer(this.buf.phiA, 0, this.buf.phi0, 0, P.resolution ** 3 * 4);
    this.device.queue.submit([enc.finish()]);

    this.redistance();
    this.computeShelter();

    this.step = 0;
    this.done = false;
    this.stoppedEarly = false;
    this.initialInside = 0;
    this.lastInside = 0;
    this._countPending = false;

    // Count the fresh block NOW. The survival guard compares against this
    // number, so without an actual dispatch here the first readback returns
    // whatever was in the buffer and the guard can fire on step one and
    // silently end the solve before anything erodes.
    const enc2 = this.device.createCommandEncoder();
    enc2.clearBuffer(this.buf.counter, 0, 16);
    this._dispatch(enc2, this.pipe.count,
      this._bind({ phiIn: this.buf.phiA, aux: this.buf.phi0, auxOut: this.buf.retreat }));
    enc2.copyBufferToBuffer(this.buf.counter, 0, this.buf.readback, 0, 4);
    this.device.queue.submit([enc2.finish()]);
    this.requestCount(true);
    return { faces: block.faces.length, totalSteps: this.totalSteps, dt: this.meta.dt };
  }

  _uploadFaces(faces) {
    const d = this.device;
    const stride = 16; // 4 x vec4
    const arr = new Float32Array(faces.length * stride);
    faces.forEach((fc, i) => {
      const o = i * stride;
      arr[o + 0] = fc.n[0]; arr[o + 1] = fc.n[1]; arr[o + 2] = fc.n[2]; arr[o + 3] = fc.d;
      arr[o + 4] = fc.u[0]; arr[o + 5] = fc.u[1]; arr[o + 6] = fc.u[2]; arr[o + 7] = fc.rough;
      arr[o + 8] = fc.v[0]; arr[o + 9] = fc.v[1]; arr[o + 10] = fc.v[2]; arr[o + 11] = fc.grain;
      arr[o + 12] = fc.hurst; arr[o + 13] = fc.so >>> 0; arr[o + 14] = fc.lac; arr[o + 15] = 0;
    });
    if (this.faceBuf) this.faceBuf.destroy();
    this.faceBuf = d.createBuffer({
      size: Math.max(64, arr.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.faceBuf, 0, arr);
    if (!this.faceCountBuf) {
      this.faceCountBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    d.queue.writeBuffer(this.faceCountBuf, 0, new Uint32Array([faces.length, 0, 0, 0]));
    this.faceGroup = d.createBindGroup({
      layout: this.faceLayout,
      entries: [
        { binding: 0, resource: { buffer: this.faceBuf } },
        { binding: 1, resource: { buffer: this.faceCountBuf } },
      ],
    });
  }

  // ----------------------------------------------------------- redistancing
  /**
   * Jump-flood redistancing. log2(n) passes, all fully parallel -- this is the
   * pass that replaces the sequential fast sweep, and it is why the solver can
   * re-metrise mid-frame without a hitch.
   */
  redistance(enc0) {
    const enc = enc0 || this.device.createCommandEncoder();
    const b = this.buf;

    this._dispatch(enc, this.pipe.seed,
      this._bind({ phiIn: b.phiA, seedIn: b.seedB, seedOut: b.seedA }));

    let src = b.seedA, dst = b.seedB;
    for (let s = 1 << Math.ceil(Math.log2(this.n)); s >= 1; s >>= 1) {
      this._setU32('jfaStep', s);
      this._dispatch(enc, this.pipe.jfa,
        this._bind({ phiIn: b.phiA, seedIn: src, seedOut: dst }));
      const t = src; src = dst; dst = t;
    }
    this._dispatch(enc, this.pipe.resolve,
      this._bind({ phiIn: b.phiA, phiOut: b.phiB, seedIn: src, seedOut: dst }));
    // resolve wrote into phiB; copy back so phiA is always the live field
    enc.copyBufferToBuffer(b.phiB, 0, b.phiA, 0, this.n ** 3 * 4);

    if (!enc0) this.device.queue.submit([enc.finish()]);
  }

  computeShelter(enc0) {
    const enc = enc0 || this.device.createCommandEncoder();
    this._dispatch(enc, this.pipe.shelter,
      this._bind({ phiIn: this.buf.phiA, aux: this.buf.phi0, auxOut: this.buf.shelter }));
    if (!enc0) this.device.queue.submit([enc.finish()]);
  }

  computeRetreat(enc0) {
    const enc = enc0 || this.device.createCommandEncoder();
    this._dispatch(enc, this.pipe.retreat,
      this._bind({ phiIn: this.buf.phiA, aux: this.buf.phi0, auxOut: this.buf.retreat }));
    if (!enc0) this.device.queue.submit([enc.finish()]);
  }

  // ------------------------------------------------------------------ solve
  /**
   * Advance the erosion by up to `count` steps. Returns the number actually
   * taken. Cheap enough to call every animation frame.
   */
  advance(count = 1) {
    if (this.done || !this.buf) return 0;
    const b = this.buf;
    const enc = this.device.createCommandEncoder();
    let taken = 0;

    for (let i = 0; i < count && this.step < this.totalSteps; i++) {
      // STEP reads phiA, writes phiB, then we swap the *bindings* by copying.
      // A true ping-pong would avoid the copy, but the redistance pass and the
      // renderer both want a single canonical "phiA = current field", and at
      // 64^3 the copy is 1 MB of on-device bandwidth -- far below the cost of
      // the step itself.
      // aux = shelter (read); auxOut must be a DIFFERENT buffer or the group
      // aliases shelter as both readable and writable, which is invalid.
      this._dispatch(enc, this.pipe.step,
        this._bind({ phiIn: b.phiA, phiOut: b.phiB, aux: b.shelter, auxOut: b.retreat }));
      enc.copyBufferToBuffer(b.phiB, 0, b.phiA, 0, this.n ** 3 * 4);
      this.step++;
      taken++;

      if (this.step % this.redistanceEvery === 0) {
        this.redistance(enc);
        this.computeShelter(enc);
      }
    }

    // Survival guard / live volume: one atomic count, read back asynchronously
    // so the frame never blocks on the GPU.
    if (this.step % 8 < taken || this.step >= this.totalSteps) {
      enc.clearBuffer(b.counter, 0, 16);
        this._dispatch(enc, this.pipe.count,
        this._bind({ phiIn: b.phiA, aux: b.phi0, auxOut: b.retreat }));
      enc.copyBufferToBuffer(b.counter, 0, b.readback, 0, 4);
      this._readbackQueued = true;
    }

    if (this.step >= this.totalSteps) {
      this.redistance(enc);
      this.computeRetreat(enc);
      this.done = true;
    } else {
      this.computeRetreat(enc);
    }

    this.device.queue.submit([enc.finish()]);
    if (this._readbackQueued) { this._readbackQueued = false; this.requestCount(); }
    return taken;
  }

  /** Non-blocking interior-cell count. */
  async requestCount(isInitial = false) {
    if (this._countPending || !this.buf) return;
    this._countPending = true;
    try {
      await this.buf.readback.mapAsync(GPUMapMode.READ);
      const v = new Uint32Array(this.buf.readback.getMappedRange().slice(0))[0];
      this.buf.readback.unmap();
      if (isInitial) {
        // A fresh block that counts zero interior cells means the block never
        // intersected the grid -- a parameter error, not an eroded rock. Leave
        // the guard disabled rather than instantly "finishing".
        this.initialInside = v;
        this.lastInside = v;
        return;
      }
      this.lastInside = v;
      // Guard only once we have a trustworthy baseline.
      if (this.initialInside > 0) {
        const frac = v / this.initialInside;
        if (frac < this.params[O.minVolFrac]) {
          this.stoppedEarly = true;
          this.done = true;
        }
      }
    } catch (e) {
      /* buffer destroyed mid-flight during a reset; harmless */
    } finally {
      this._countPending = false;
    }
  }

  get volumeFraction() {
    return this.initialInside ? this.lastInside / this.initialInside : 1;
  }

  // ----------------------------------------------------------------- render
  updateCamera({ invViewProj, eye, sun, aspect, time, maxSteps = 160 }) {
    const c = this.camera;
    c.set(invViewProj, 0);
    // layout: mat4 [0..63], eye [64..79], sun [80..95], misc [96..111]
    c[16] = eye[0]; c[17] = eye[1]; c[18] = eye[2]; c[19] = 1;
    c[20] = sun[0]; c[21] = sun[1]; c[22] = sun[2]; c[23] = 0;
    c[24] = aspect; c[25] = time; c[26] = maxSteps; c[27] = 0;
    this.device.queue.writeBuffer(this.buf.camera, 0, c);
  }

  render(view) {
    const b = this.buf;
    const group = this.device.createBindGroup({
      layout: this.renderLayout,
      entries: [
        { binding: 0, resource: { buffer: b.params } },
        { binding: 1, resource: { buffer: b.phiA } },
        { binding: 2, resource: { buffer: b.shelter } },
        { binding: 3, resource: { buffer: b.retreat } },
        { binding: 4, resource: { buffer: b.camera } },
      ],
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.05, g: 0.06, b: 0.07, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(this.renderPipe);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Pull a grid-sized f32 buffer back to the CPU. Only used by Export -- never
   * in the loop, because a readback stalls the pipeline.
   */
  async readField(which = 'phi') {
    const src = { phi: this.buf.phiA, phi0: this.buf.phi0,
                  shelter: this.buf.shelter, retreat: this.buf.retreat }[which];
    if (!src) throw new Error(`readField: unknown field "${which}"`);
    // One staging buffer is reused across calls, so overlapping readbacks
    // would race on the same mapping. Serialise rather than leaving it to the
    // caller to remember.
    if (this._readInFlight) await this._readInFlight;
    let release;
    this._readInFlight = new Promise((r) => { release = r; });
    try {
      return await this._readFieldInner(src);
    } finally {
      this._readInFlight = null;
      release();
    }
  }

  async _readFieldInner(src) {
    const bytes = this.n ** 3 * 4;
    if (!this.exportBuf || this.exportBuf.size !== bytes) {
      this.exportBuf?.destroy();
      this.exportBuf = this.device.createBuffer({
        size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, this.exportBuf, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await this.exportBuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.exportBuf.getMappedRange().slice(0));
    this.exportBuf.unmap();
    return out;
  }

  /** shelter and retreat are only STORAGE; give them COPY_SRC for export. */
  readPhi() { return this.readField('phi'); }
}

// -------------------------------------------------------------- helpers
/** Per-mineral cumulative thresholds, recovered from the modal composition. */
function cdfThresholds(_lut, mode) {
  const out = new Array(6).fill(1);
  let total = 0;
  for (const m of MINERAL_LIST) total += mode[m.name] || 0;
  total = total || 1;
  let acc = 0;
  for (const m of MINERAL_LIST) {
    acc += (mode[m.name] || 0) / total;
    out[m.id] = acc;
  }
  out[5] = 1;
  return out;
}

/** Deterministic aspect ratio, matching generator.js's RNG draw order. */
function aspectFor(seed, av) {
  let s = (seed * 2654435761 + 1013904223) >>> 0;
  const nx = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const nrm = () => {
    const u = Math.max(1e-9, nx()), v = nx();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return [1.0, Math.exp(nrm() * av * 0.8) * 0.85, Math.exp(nrm() * av * 0.8)];
}
