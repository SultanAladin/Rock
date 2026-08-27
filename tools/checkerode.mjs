/**
 * Erosion acceptance tests.
 *
 * These pin the *observable outcome* of weathering, which every structural
 * check we had was blind to: the shaders compiled, the pipelines were valid,
 * the dispatch order was right -- and the rock still came out with sharp
 * corners and a sphericity that FELL with age.
 */
import { Field3, redistance } from '../src/core/grid.js';
import { buildJointBlock } from '../src/core/joints.js';
import { GrainField, LITHOLOGIES } from '../src/core/petrology.js';
import { weather, DEFAULT_WEATHERING } from '../src/core/weathering.js';
import { dualContour, largestComponent } from '../src/core/mesher.js';
import { blockAxes } from '../src/core/generator.js';
import { RNG } from '../src/core/rng.js';

let fails = [];
const ok  = (c, m) => { console.log(`[ ${c ? 'OK' : '!!'} ] ${m}`); if (!c) fails.push(m); };

// ---------------------------------------------------------------- 1. the law
// Spheroidal weathering is DEFINED by corners > edges > faces. Start from an
// exact cube so the only thing under test is the rate law.
{
  const n = 56, L = 0.95, h = 2 * L / (n - 1), a = 0.55;
  const f = new Field3(n, L);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = -L + i * h, y = -L + j * h, z = -L + k * h;
    const dx = Math.abs(x) - a, dy = Math.abs(y) - a, dz = Math.abs(z) - a;
    const ox = Math.max(dx, 0), oy = Math.max(dy, 0), oz = Math.max(dz, 0);
    f.data[f.idx(i, j, k)] = Math.hypot(ox, oy, oz) + Math.min(Math.max(dx, Math.max(dy, dz)), 0);
  }
  const grains = { durability: () => 1, durabilityCell: () => 1 };
  const r = weather(f, grains, { ...DEFAULT_WEATHERING, years: 1.5, grussification: 0,
    heterogeneity: 0, rindlet: 0, insolation: 0, moistureGradient: 0, buriedFraction: 0 }, () => {});
  const phi = r.phi;
  const tri = (x, y, z) => {
    const fi = (x + L) / h, fj = (y + L) / h, fk = (z + L) / h;
    const i = Math.min(n - 2, Math.max(0, Math.floor(fi))), j = Math.min(n - 2, Math.max(0, Math.floor(fj))), k = Math.min(n - 2, Math.max(0, Math.floor(fk)));
    const tx = fi - i, ty = fj - j, tz = fk - k, d = phi.data, X = (p, q, s) => phi.idx(p, q, s);
    const c = (p, q, t) => p * (1 - t) + q * t;
    const c00 = c(d[X(i, j, k)], d[X(i + 1, j, k)], tx), c10 = c(d[X(i, j + 1, k)], d[X(i + 1, j + 1, k)], tx);
    const c01 = c(d[X(i, j, k + 1)], d[X(i + 1, j, k + 1)], tx), c11 = c(d[X(i, j + 1, k + 1)], d[X(i + 1, j + 1, k + 1)], tx);
    return c(c(c00, c10, ty), c(c01, c11, ty), tz);
  };
  const surf = (dx, dy, dz) => {
    const l = Math.hypot(dx, dy, dz); dx /= l; dy /= l; dz /= l;
    let prev = tri(0, 0, 0), pt = 0;
    for (let t = h * 0.02; t < 1.5; t += h * 0.02) {
      const v = tri(dx * t, dy * t, dz * t);
      if (prev < 0 && v >= 0) return pt + (0 - prev) / (v - prev) * (t - pt);
      prev = v; pt = t;
    }
    return NaN;
  };
  const rf = a - surf(1, 0, 0);
  const re = a * Math.SQRT2 - surf(1, 1, 0);
  const rc = a * Math.sqrt(3) - surf(1, 1, 1);
  console.log(`       retreat face ${rf.toFixed(4)} edge ${re.toFixed(4)} corner ${rc.toFixed(4)}`);
  ok(rc > re && re > rf, `spheroidal ordering corner > edge > face (${rc.toFixed(3)} > ${re.toFixed(3)} > ${rf.toFixed(3)})`);
  ok(rc / Math.max(1e-6, rf) > 3, `corners retreat >3x faster than faces (got ${(rc / rf).toFixed(1)}x)`);
}

// ------------------------------------------------- 2. rounding, on a real block
// Sphericity must RISE with exposure age. It fell (0.84 -> 0.02) because the
// domain was a fixed 0.95 * size while joint spacing is anisotropic and random:
// the block overruns that by up to 3.2x, so it was clipped by the grid wall,
// the solver froze the boundary shell, and dualContour could not close the
// surface -- making the divergence-theorem volume (and sphericity) garbage.
// Sweep styles and seeds, because a single lucky aspect ratio fits.
{
  const n = 44;
  const cases = [];
  for (const style of ['orthogonal', 'polyhedral', 'columnar']) {
    for (const seed of [5, 1337]) cases.push({ style, seed });
  }
  let wallHits = 0, openHits = 0, rose = 0, shrank = 0;
  for (const { style, seed } of cases) {
    const rng = new RNG(seed), av = 0.28;
    const aspect = [1, Math.exp(rng.normal() * av * 0.8) * 0.85, Math.exp(rng.normal() * av * 0.8)];
    const block = buildJointBlock({ seed, style, size: 1, aspect, jointRoughness: 1,
      hurst: 0.75, grainSize: LITHOLOGIES['biotite-granite'].grain, sheetingCurvature: 0 });
    const axes = blockAxes(block.sdf, 1);
    const extent = axes.max * 1.06;
    const measure = (years) => {
      const f = new Field3(n, extent); f.fill(block.sdf); redistance(f, 2, f.h * 8);
      const grains = new GrainField(LITHOLOGIES['biotite-granite'], seed * 7919 + 13);
      const phi = weather(f, grains, { ...DEFAULT_WEATHERING, years, blockRadius: axes.min }, () => {}).phi;
      const h = phi.h;
      let V = 0, A = 0, touch = 0; const edge = new Map();
      for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const v = phi.data[phi.idx(i, j, k)];
        const onWall = !(i && j && k && i < n - 1 && j < n - 1 && k < n - 1);
        if (v < 0) { V += h * h * h; if (onWall) touch++; }
        const e = 1.5 * h;
        if (Math.abs(v) < e) A += (0.5 / e) * (1 + Math.cos(Math.PI * v / e)) * h * h * h;
      }
      const m = largestComponent(dualContour(phi, { sharpness: 0.85 }));
      for (let t = 0; t < m.indices.length; t += 3)
        for (const [x, y] of [[m.indices[t], m.indices[t + 1]], [m.indices[t + 1], m.indices[t + 2]], [m.indices[t + 2], m.indices[t]]]) {
          const kk = x < y ? `${x}:${y}` : `${y}:${x}`; edge.set(kk, (edge.get(kk) || 0) + 1);
        }
      let open = 0; for (const c of edge.values()) if (c === 1) open++;
      return { sph: Math.cbrt(Math.PI) * Math.pow(6 * V, 2 / 3) / A, V, touch, open };
    };
    const a = measure(0.0), b = measure(1.2);
    if (a.touch || b.touch) wallHits++;
    if (a.open || b.open) openHits++;
    if (b.sph > a.sph) rose++;
    if (b.V < a.V) shrank++;
    console.log(`       ${style}/${seed}: sph ${a.sph.toFixed(3)} -> ${b.sph.toFixed(3)}  V ${a.V.toFixed(3)} -> ${b.V.toFixed(3)}  wall ${a.touch + b.touch} open ${a.open + b.open}`);
  }
  ok(wallHits === 0, `no block touches the domain wall (${wallHits}/${cases.length} did)`);
  ok(openHits === 0, `every contoured mesh is closed (${openHits}/${cases.length} were open)`);
  ok(rose === cases.length, `sphericity RISES with age in all cases (${rose}/${cases.length})`);
  ok(shrank === cases.length, `weathering removes volume in all cases (${shrank}/${cases.length})`);
}

// -------------------------------------------- 3. the speed field is not noise
// Cell-scale mineral variation must be a converged average, not sampling error.
{
  const g = new GrainField(LITHOLOGIES['biotite-granite'], 137);
  const h = 2 * 0.95 / 63;
  const v = []; for (let i = 0; i < 300; i++) v.push(g.durabilityCell(-0.4 + i * h, 0.013, 0.021, h));
  const m = v.reduce((a, b) => a + b) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
  // converged reference
  const brute = (x, y, z) => { let inv = 0, c = 0, S = 14;
    for (let k = 0; k < S; k++) for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
      inv += 1 / g.durability(x + (i / S - 0.5 + 0.5 / S) * h, y + (j / S - 0.5 + 0.5 / S) * h, z + (k / S - 0.5 + 0.5 / S) * h); c++; }
    return c / inv; };
  const w = []; for (let i = 0; i < 60; i++) w.push(brute(-0.4 + i * h, 0.013, 0.021));
  const mw = w.reduce((a, b) => a + b) / w.length;
  const sdw = Math.sqrt(w.reduce((a, b) => a + (b - mw) ** 2, 0) / w.length);
  console.log(`       durabilityCell sd ${sd.toFixed(4)} vs converged ${sdw.toFixed(4)}`);
  ok(sd < 4 * sdw + 0.02, `cell durability is an average, not sampling noise (sd ${sd.toFixed(4)} vs true ${sdw.toFixed(4)})`);
  ok(Math.abs(m - mw) < 0.05, `cell durability mean is unbiased (${m.toFixed(4)} vs ${mw.toFixed(4)})`);
}

console.log(fails.length ? `\nFAIL (${fails.length})\n` + fails.map(f => '  - ' + f).join('\n') : '\nPASS');
process.exit(fails.length ? 1 : 0);
