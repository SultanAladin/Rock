/**
 * Level-set weathering solver for granitoid boulders.
 * =====================================================================
 * We advect the rock/air interface phi = 0 into the rock with a normal speed F
 * that encodes the actual geomorphic processes. The level-set equation for
 * motion in the normal direction is
 *
 *      d(phi)/dt + F |grad phi| = 0,
 *
 * and with our sign convention (phi < 0 inside the solid) *erosion* means F < 0
 * and phi increases. We integrate explicitly with Godunov upwinding, and
 * re-distance periodically so phi stays metric.
 *
 * THE SPEED FUNCTION
 * ------------------
 *   F = R * [ A_sph * sat(k_hat, p)          spheroidal / granular attack
 *           + A_cav * sat(-k_hat, q) * s^2   cavernous (tafoni) attack
 *           + A_uni ]                        background surface lowering
 *       * weakness(x)
 *
 * with k_hat = kappa * L (curvature made dimensionless by the block size) and
 *
 *      sat(k, p) = max(k,0)^p / (1 + max(k,0)^p)
 *
 * The saturating form matters and is not cosmetic. A raw power law in curvature
 * is unbounded: at grid resolution a fresh joint arris has kappa ~ 1/h, so the
 * rate blows up, the CFL bound collapses to nothing, and the solver spends its
 * whole step budget nibbling one voxel off a corner while the rest of the rock
 * never moves. That failure mode is exactly what makes most "SDF erosion" demos
 * produce a shape indistinguishable from the input. Saturation also happens to
 * be the physically correct asymptote: a corner is attacked from at most three
 * joint faces, so its rate is bounded by ~3x the face rate no matter how sharp
 * it is. Bounded F gives a predictable dt and a retreat budget you can state in
 * millimetres.
 *
 * 1. SPHEROIDAL TERM. A joint block is attacked simultaneously from every
 *    bounding joint face. A corner is exposed to three, an edge to two, a face
 *    to one, so the local attack rate scales with the solid angle of exposure,
 *    which to first order is the mean curvature. This is the textbook mechanism
 *    of corestone rounding and it is the same curvature-driven flow used for
 *    weathering concave rock in the graphics literature (Jones/Beardall et al.,
 *    spheroidal + cavernous rates as functions of mean curvature on a voxel
 *    grid).
 *
 * 2. CAVERNOUS TERM. Concavities retain moisture and salt and are sheltered
 *    from rainwash, so a hollow, once started, accelerates. That positive
 *    feedback is what produces tafoni and honeycomb weathering. We gate it on a
 *    shelter integral (squared, so it is genuinely selective) - without the
 *    gate this term simply dissolves the whole rock.
 *
 * 3. WEAKNESS FIELD. Built from the crystal aggregate (petrology.js), so it is
 *    a *volume* property: as the surface retreats it exposes new crystals with
 *    their own resistance. Quartz is effectively inert; biotite oxidises first
 *    (Fe(II)->Fe(III) with a positive delta-V) and that expansion is the
 *    documented trigger for rindlet spalling and grussification in granitoids.
 *    Because the same Laguerre field feeds the shader, quartz physically stands
 *    proud where the shader draws quartz. The micro-relief is emergent.
 *
 * 4. MOISTURE. Basal/buried rock stays damp far longer, so it weathers faster.
 *    This is why corestones are rounder at the base and why granite inselbergs
 *    develop flared slopes and basal notches.
 *
 * 5. INSOLATION. Sun-facing surfaces see harder thermal and wet-dry cycling; a
 *    mild directional bias, deliberately not a dominant term.
 *
 * 6. RINDLETS. Spheroidal weathering produces concentric 35-50 mm shells that
 *    spall off. We inject a periodic weakness phased on the *initial* distance
 *    field, so the shells are parallel to the original joint surface - which is
 *    what rindlets physically are - giving scaly onion-skin relief rather than
 *    a bland ellipsoid.
 *
 * PERFORMANCE
 * -----------
 * Everything expensive is restricted to a narrow band around the interface
 * (|phi| < bandWidth * h). Curvature, speed assembly, the update, the shelter
 * integral and the durability sampling all run over the band index list, which
 * is a few percent of the volume. The band is rebuilt whenever we re-distance.
 * Durability is cached per lattice site and computed lazily the first time a
 * site enters the band, so a site is never sampled twice.
 */

import { Field3, meanCurvatureField, godunovNormGrowth, godunovNormGrowthFast, redistance } from './grid.js';
import { fbm3 } from './noise.js';

export const DEFAULT_WEATHERING = {
  /** Retreat budget as a fraction of block radius. 1.0 ~ a well-rounded corestone. */
  years: 0.7,
  spheroidal: 1.0,        // A_sph  (relative rate at a fully convex corner)
  spheroidalPower: 1.25,  // p      (corner-vs-face contrast)
  cavernous: 0.35,        // A_cav
  cavernousPower: 1.6,    // q
  uniform: 0.06,          // A_uni
  grussification: 0.55,   // strength of mineral-selective attack
  moistureGradient: 0.55,
  buriedFraction: 0.25,
  insolation: 0.25,
  sunDir: [0.45, 0.78, 0.44],
  rindlet: 0.35,
  rindletSpacing: 0.042,  // m; field measurement on granite corestones: 35-50 mm
  shelterRadius: 0.22,
  maxSteps: 1200,         // safety cap; the budget normally terminates first
  redistanceEvery: 12,
  /** Rock radius in metres, for the retreat budget. Set by the caller. */
  blockRadius: 0,
  bandWidth: 3.0,         // in cells
  /**
   * Rounding radius as a fraction of the block radius L. This is the length
   * scale that non-dimensionalises curvature: k_hat = kappa * (roundingRadius*L).
   *
   * It is the radius at which corner rounding stops being accelerated. A corner
   * sharper than this is attacked at close to the saturated maximum rate; once
   * it has been blunted to this radius the curvature term falls away and the
   * uniform term takes over. Field corestones round their arrises to a radius
   * of order a tenth to a fifth of the block, which is where the default sits.
   *
   * It is also the single knob that sets solver cost. Explicit curvature flow
   * needs dt ~ h^2 / (A_sph * l), so the step count goes as L*l/h^2: normalising
   * on the block radius (l = L) instead of the rounding radius costs ~6x the
   * steps for a rounding that is visually identical, because the extra work all
   * goes into diffusing modes longer than the block.
   */
  roundingRadius: 0.14,

  /**
   * Floor on the surviving solid, as a fraction of the fresh block volume.
   * Curvature-driven erosion of a finite body is self-accelerating at the end,
   * so without this an aggressive age returns an empty grid.
   */
  minVolumeFraction: 0.12,

  /**
   * Mesoscale (decimetre) compositional heterogeneity: mafic schlieren, aplite
   * and quartz veins, alteration haloes. Unlike the crystal field this survives
   * grid-cell averaging, so it is what legitimately drives differential relief
   * at solver resolution -- knobs, ribs and recessive bands.
   */
  heterogeneity: 0.45,
  heteroScale: 0.35,      // correlation length as a fraction of block radius
};

/**
 * Ambient shelter over a set of band cells. 1 = deeply sheltered pocket,
 * 0 = fully exposed. This is a short-range occlusion integral, which is the
 * right proxy for "does water sit here and does the sun miss it".
 */
function computeShelter(phi, band, radius, out, samples = 12, steps = 4) {
  const dirs = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < samples; i++) {
    const z = 1 - (2 * i + 1) / samples;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const t = ga * i;
    dirs.push(r * Math.cos(t), r * Math.sin(t), z);
  }
  const I = band.idx, JK = band.ijk;
  for (let b = 0; b < band.length; b++) {
    const id = I[b];
    const x = phi.coord(JK[b * 3]), y = phi.coord(JK[b * 3 + 1]), z = phi.coord(JK[b * 3 + 2]);
    let occ = 0;
    for (let dI = 0; dI < dirs.length; dI += 3) {
      const dx = dirs[dI], dy = dirs[dI + 1], dz = dirs[dI + 2];
      for (let s = 1; s <= steps; s++) {
        const t = (s / steps) * radius;
        if (phi.sample(x + dx * t, y + dy * t, z + dz * t) < 0) { occ++; break; }
      }
    }
    out[id] = occ / samples;
  }
}

/**
 * Collect cells within `w` cells of the interface, with their (i,j,k) decoded
 * once. Decoding the index inside the step loop costs two integer divisions per
 * cell per pass, three passes per step, several hundred steps -- it measurably
 * dominates once the arithmetic is tightened.
 *
 * Boundary cells (within one cell of the grid edge) are excluded: the clamped
 * stencil there is not a valid difference operator, and letting the front reach
 * the domain wall produces a flat-sided rock.
 */
function buildBand(phi, w) {
  const lim = w * phi.h;
  const d = phi.data, n = phi.n, nn = n * n;
  const idxs = [], ijk = [];
  for (let k = 1; k < n - 1; k++)
    for (let j = 1; j < n - 1; j++) {
      const row = k * nn + j * n;
      for (let i = 1; i < n - 1; i++) {
        const p = row + i;
        const v = d[p];
        if (v > -lim && v < lim * 0.8) { idxs.push(p); ijk.push(i, j, k); }
      }
    }
  return { idx: new Int32Array(idxs), ijk: new Int32Array(ijk), length: idxs.length };
}

/** Trilinear sample of a bare Float32Array laid out on `phi`'s lattice. */
function sampleArr(arr, phi, x, y, z) {
  const n = phi.n, h = phi.h;
  let fx = (x - phi.origin) / h, fy = (y - phi.origin) / h, fz = (z - phi.origin) / h;
  fx = Math.min(n - 1.0001, Math.max(0, fx));
  fy = Math.min(n - 1.0001, Math.max(0, fy));
  fz = Math.min(n - 1.0001, Math.max(0, fz));
  const i = fx | 0, j = fy | 0, k = fz | 0;
  const tx = fx - i, ty = fy - j, tz = fz - k;
  const s = (a, b, c) => arr[(c * n + b) * n + a];
  const c00 = s(i, j, k) * (1 - tx) + s(i + 1, j, k) * tx;
  const c10 = s(i, j + 1, k) * (1 - tx) + s(i + 1, j + 1, k) * tx;
  const c01 = s(i, j, k + 1) * (1 - tx) + s(i + 1, j, k + 1) * tx;
  const c11 = s(i, j + 1, k + 1) * (1 - tx) + s(i + 1, j + 1, k + 1) * tx;
  const c0 = c00 * (1 - ty) + c10 * ty;
  const c1 = c01 * (1 - ty) + c11 * ty;
  return c0 * (1 - tz) + c1 * tz;
}

/**
 * @param {Field3} phi         fresh block SDF, modified in place
 * @param {GrainField} grains  crystal aggregate supplying durability
 * @param {object} opts
 * @param {(f:number,label:string)=>void} [onProgress]
 */
export function weather(phi, grains, opts = {}, onProgress) {
  const P = { ...DEFAULT_WEATHERING, ...opts };
  const n = phi.n, h = phi.h;
  const N3 = n * n * n;
  // Characteristic radius of the ROCK, which is what `years` is calibrated
  // against. NOT phi.extent: the domain is padded around the block by a margin
  // that varies with joint style (1.5x to 3.2x size), so tying the budget to it
  // made `years` mean 14% retreat on one block and 46% on another. Callers pass
  // the measured block radius; falling back to the extent preserves the old
  // behaviour for callers that don't.
  const L = P.blockRadius || phi.extent;

  const phi0 = phi.clone();

  // ---- lazily-cached static fields --------------------------------------
  const weak = new Float32Array(N3);          // durability -> weakness multiplier
  const weakDone = new Uint8Array(N3);
  const shelter = new Float32Array(N3);
  const kappa = new Float32Array(N3);
  const speed = new Float32Array(N3);

  // Weakness envelope. This is not a safety clamp, it sets the physics: the
  // *fastest* cell retreats WEAK_MAX times the nominal budget, so a loose bound
  // lets the softest patches consume the entire rock before the corners have
  // finished rounding. Measured contrast between fresh granite and its most
  // altered mesoscale bands is well under 2x in bulk retreat rate. It also sets
  // the timestep, since bEff scales with it.
  const WEAK_MIN = 0.5, WEAK_MAX = 1.8;
  const twoPiOverS = (2 * Math.PI) / Math.max(1e-4, P.rindletSpacing);
  const yMin = -L, yMax = L;
  const buriedY = yMin + (yMax - yMin) * P.buriedFraction;

  /** Combined static weakness at a lattice site: mineralogy x moisture x rindlet. */
  function weaknessAt(id, x, y, z) {
    if (weakDone[id]) return weak[id];

    // Mineralogy, UPSCALED to the cell. See GrainField.durabilityCell for why
    // point-sampling the crystal field at grid resolution is wrong (it is white
    // noise at 30 mm and it grows grid-scale fuzz instead of rounding the rock).
    const dur = grains.durabilityCell(x, y, z, h);
    let m = 1 + P.grussification * (1 / Math.max(0.10, dur) - 1);
    // renormalise so grussification changes contrast, not overall rate
    m /= 1 + P.grussification * 1.4;

    // Mesoscale heterogeneity: the grid-scale variation that IS real. Mafic
    // schlieren and biotite-rich bands weather recessively; aplite and quartz
    // veins stand out as resistant ribs. Correlation length is a few tens of
    // centimetres, so it survives cell averaging where the crystal field cannot.
    if (P.heterogeneity > 0) {
      const s = 1 / Math.max(0.02, P.heteroScale * L);
      m *= 1 + P.heterogeneity * 0.55 * fbm3(x * s, y * s, z * s, 1777, 0.85, 2.1, 3);
    }

    // moisture: damp at the base, drying upward
    const t = Math.max(0, Math.min(1, (y - buriedY) / (0.55 * (yMax - buriedY) + 1e-6)));
    m *= 1 + P.moistureGradient * (1 - t * t * (3 - 2 * t));

    // rindlets: concentric shells parallel to the original joint surface
    if (P.rindlet > 0) {
      const jit = 0.35 * P.rindletSpacing * fbm3(x * 5, y * 5, z * 5, 991, 0.9, 2.1, 3);
      m *= 1 + P.rindlet * 0.5 * (1 + Math.sin((phi0.data[id] + jit) * twoPiOverS));
    }
    // Clamp. The three factors above are independent worst cases that never
    // physically co-occur, and an unclamped product both exaggerates the
    // mineral contrast beyond anything measured (biotite does not retreat ten
    // times faster than the bulk in a continuum sense) and destroys the
    // timestep. WEAK_MAX is the bound the CFL estimate relies on.
    m = Math.max(WEAK_MIN, Math.min(WEAK_MAX, m));
    weak[id] = m; weakDone[id] = 1;
    return m;
  }

  /**
   * Weakness evaluated at an arbitrary (interface) point, cached on the lattice
   * and trilinearly reconstructed. Caching on the lattice rather than
   * recomputing per query keeps the crystal-aggregate cost at one evaluation
   * per cell for the whole run, while the interpolation makes the field
   * continuous -- a piecewise-constant weakness would reintroduce exactly the
   * cell-scale rate discontinuities that velocity extension exists to remove.
   */
  function weaknessExtended(x, y, z) {
    const nn = n;
    let fx = (x - phi.origin) / h, fy = (y - phi.origin) / h, fz = (z - phi.origin) / h;
    fx = Math.min(nn - 1.0001, Math.max(0, fx));
    fy = Math.min(nn - 1.0001, Math.max(0, fy));
    fz = Math.min(nn - 1.0001, Math.max(0, fz));
    const i = fx | 0, j = fy | 0, k = fz | 0;
    const tx = fx - i, ty = fy - j, tz = fz - k;
    const w = (a, b, c) => {
      const id = (c * nn + b) * nn + a;
      return weakDone[id] ? weak[id] : weaknessAt(id, phi.coord(a), phi.coord(b), phi.coord(c));
    };
    const c00 = w(i, j, k) * (1 - tx) + w(i + 1, j, k) * tx;
    const c10 = w(i, j + 1, k) * (1 - tx) + w(i + 1, j + 1, k) * tx;
    const c01 = w(i, j, k + 1) * (1 - tx) + w(i + 1, j, k + 1) * tx;
    const c11 = w(i, j + 1, k + 1) * (1 - tx) + w(i + 1, j + 1, k + 1) * tx;
    const c0 = c00 * (1 - ty) + c10 * ty;
    const c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
  }

  const sun = (() => { const s = P.sunDir; const l = Math.hypot(s[0], s[1], s[2]) || 1; return [s[0] / l, s[1] / l, s[2] / l]; })();

  // Retreat budget, in metres of maximum face retreat. Scaling by L means
  // `years` is scale-invariant: 1.0 rounds a 10 cm cobble and a 3 m boulder to
  // the same degree.
  // `years` is specified as a RETREAT DISTANCE (fraction of block radius that
  // the fastest-attacked corner loses), which is the only formulation an artist
  // or a geologist can reason about. Integration happens in time, so convert:
  //   t_total = distance / F_corner,
  // where F_corner is the speed at a fully saturated convex corner in average
  // rock. Forgetting this conversion means the budget is in time units while
  // the speed multiplies it, so the actual retreat scales with the rate
  // parameters -- turning up A_sph then dissolves the rock instead of rounding
  // it faster, and `years = 2` asks for 2.4 m of retreat on a 1 m boulder.
  //
  // Calibration: age 1.0 = 12% of the block radius lost at the corners. That is
  // the point at which a fresh joint block reads as a properly rounded
  // corestone -- measured Wadell sphericity climbs from ~0.83 (fresh block) into
  // the 0.85-0.90 band that field corestones occupy. Beyond ~2.5 there is
  // simply not much rock left, which is physically true but rarely what anyone
  // wants, so the solver refuses to consume the solid entirely (below).
  const Fcorner = Math.max(1e-6, P.spheroidal * 0.5 + P.cavernous * 0.15 + P.uniform);
  const targetRetreat = P.years * L * 0.12;
  const budget = targetRetreat / Fcorner;
  // TIMESTEP.
  // Because the rate law saturates, F is bounded by (A_sph+A_cav+A_uni)*WEAK_MAX
  // regardless of how sharp the geometry gets, so the advective CFL is a real,
  // parameter-independent bound:  dt <= C h / Fmax.
  //
  // Crucially we do NOT impose the usual parabolic dt <= h^2 / (6 b) bound for
  // pure mean-curvature flow. That bound applies when the speed is *linear* in
  // kappa, F = b kappa, where the scheme is a diffusion and the h^2 restriction
  // is genuine. Here F saturates: dF/dkappa -> 0 as kappa grows, so the
  // equation is not parabolic at the sharp features that would set the bound,
  // and the effective diffusion coefficient near an arris is vanishing rather
  // than maximal. Enforcing h^2 there costs three orders of magnitude in steps
  // and buys no stability. We keep a mild parabolic term evaluated with the
  // *saturated* derivative (which decays as 1/kappa^p) and take the min.
  const Fmax = (P.spheroidal + P.cavernous + P.uniform) * WEAK_MAX;
  const dtAdv = (0.45 * h) / Math.max(1e-6, Fmax);
  // Parabolic bound. F is only *asymptotically* flat in kappa; the equation is
  // still a diffusion at moderate curvature, and the diffusivity
  //   b(k) = dF/dkappa = A_sph p k^(p-1) / (1+k^p)^2 * (L * roundingRadius)
  // is maximised near k = 1, NOT at the grid-resolution curvature k = L/h.
  // Evaluating the bound out at k = L/h (where b has already decayed by ~1/k^2)
  // is what lets high-frequency modes on the band grow: the surface sprouts
  // grid-aligned fuzz, the triangle count explodes and sphericity collapses.
  // So take the supremum over k.
  const bSup = (() => {
    const p = P.spheroidalPower;
    // d/dk of k^(p-1)/(1+k^p)^2 vanishes at k^p = (p-1)/(p+1); for p<=1 the
    // supremum is at k -> 0+ for p<1 and at k=0..1 for p=1, so clamp to k=1.
    const kStar = p > 1 ? Math.pow((p - 1) / (p + 1), 1 / p) : 1.0;
    const k = Math.max(1e-3, kStar);
    const kp2 = Math.pow(k, p);
    return p * Math.pow(k, p - 1) / ((1 + kp2) * (1 + kp2));
  })();
  const bEff = P.spheroidal * bSup * (L * P.roundingRadius) * WEAK_MAX;
  const dtDiff = (0.15 * h * h) / Math.max(1e-9, bEff);
  const dt = Math.min(dtAdv, dtDiff);
  const stepsNeeded = Math.ceil(budget / dt);
  const steps = Math.max(1, Math.min(P.maxSteps, stepsNeeded));

  let band = buildBand(phi, P.bandWidth);
  computeShelter(phi, band, P.shelterRadius, shelter);

  let elapsed = 0;
  let stoppedEarly = false;
  let initialInside = 0;
  for (let p = 0; p < N3; p++) if (phi.data[p] < 0) initialInside++;
  const grad = [0, 0, 0];
  const cs = L * P.roundingRadius;
  const origin = phi.origin;
  const nsq = n * n;
  const inv2h = 1 / (2 * h);

  for (let step = 0; step < steps; step++) {
    // Curvature is only meaningful (and only needed) in the band.
    meanCurvatureField(phi, kappa, band);

    // Velocity extension (Adalsteinsson & Sethian). The speed is a property of
    // the SURFACE, not of the volume, so for an off-interface band cell we must
    // evaluate F at the closest point ON the interface, not at the cell itself.
    //
    // Skipping this is the classic narrow-band failure. Evaluating F pointwise
    // makes it vary along the normal; the level sets then move at different
    // speeds, phi stops being a distance function, |grad phi| drifts away from
    // 1, and the Godunov term amplifies exactly the high-frequency modes it is
    // meant to control. Symptom: surface AREA rises as the rock erodes and the
    // triangle count climbs, i.e. the boulder grows fuzz instead of rounding.
    // With extension, F is constant along each normal by construction, phi stays
    // metric between re-distancings, and the front stays smooth.
    //
    // Since phi is (approximately) signed distance, the closest interface point
    // is simply p - phi(p) * n.
    const BI = band.idx, BJK = band.ijk;
    const src = phi.data.slice();
    for (let b = 0; b < band.length; b++) {
      const id = BI[b];
      const i = BJK[b * 3], j = BJK[b * 3 + 1], k = BJK[b * 3 + 2];
      let x_ = origin + i * h, y_ = origin + j * h, z_ = origin + k * h;

      const sd = phi.data;
      const gx = (sd[id + 1] - sd[id - 1]) * inv2h;
      const gy = (sd[id + n] - sd[id - n]) * inv2h;
      const gz = (sd[id + nsq] - sd[id - nsq]) * inv2h;
      const glen = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1;
      const nx = gx / glen, ny = gy / glen, nz = gz / glen;
      const d0 = src[id];
      // project onto the interface
      const px = x_ - d0 * nx, py = y_ - d0 * ny, pz = z_ - d0 * nz;

      const kh = sampleArr(kappa, phi, px, py, pz) * cs;
      const conv = Math.max(0, kh), conc = Math.max(0, -kh);
      const cp = Math.pow(conv, P.spheroidalPower);
      const sph = P.spheroidal * (cp / (1 + cp));
      const cq = Math.pow(conc, P.cavernousPower);
      const shl = sampleArr(shelter, phi, px, py, pz);
      const cav = P.cavernous * (cq / (1 + cq)) * shl * shl;

      let f = sph + cav + P.uniform;

      if (P.insolation > 0) {
        const cosA = nx * sun[0] + ny * sun[1] + nz * sun[2];
        f *= 1 + P.insolation * (0.5 * cosA + 0.25 * Math.max(0, -cosA));
      }

      // Weakness is also extended: sampled at the interface point, and cached
      // on a lattice keyed by the *interface* location rather than the cell.
      // explicit update: phi += dt * F * |grad phi|_Godunov
      const F = f * weaknessExtended(px, py, pz);
      phi.data[id] = src[id] + dt * F * godunovNormGrowthFast(src, n, nsq, h, id);
    }

    elapsed += dt;

    // Survival guard. A curvature-driven front on a finite solid is
    // self-accelerating in the endgame: as the body shrinks its curvature rises
    // everywhere, so the last 10% of the rock disappears in a small fraction of
    // the total time and the run can end with an empty grid. Rather than let a
    // high age silently return nothing, stop once the solid has shrunk to
    // `minVolumeFraction` of its original volume. Physically this is the point
    // where the corestone has become a cobble and would have been transported
    // away rather than continuing to weather in place.
    if ((step & 7) === 0) {
      let inside = 0;
      const dd = phi.data;
      for (let p = 0; p < N3; p++) if (dd[p] < 0) inside++;
      if (inside < initialInside * P.minVolumeFraction) { stoppedEarly = true; break; }
    }

    if ((step + 1) % P.redistanceEvery === 0 || step === steps - 1) {
      redistance(phi, 2, (P.bandWidth + 3) * h);
      band = buildBand(phi, P.bandWidth);
      computeShelter(phi, band, P.shelterRadius, shelter);
    }
    if (onProgress && (step % 8 === 0)) onProgress((step + 1) / steps, 'weathering');
  }

  redistance(phi, 2);
  // final shelter over a fresh band, for the baked vertex attribute
  band = buildBand(phi, P.bandWidth + 2);
  computeShelter(phi, band, P.shelterRadius, shelter, 18, 6);

  // Surface retreat: how much rock was removed here. Drives rind colour (fresh
  // interiors grey, long-exposed skin Fe-stained) and the roughness split
  // between spalled and case-hardened surfaces.
  const retreat = new Float32Array(N3);
  for (let p = 0; p < N3; p++) retreat[p] = phi.data[p] - phi0.data[p];

  return {
    phi, retreat, shelter, weakness: weak, elapsed, budget,
    steps, dt, truncated: stepsNeeded > P.maxSteps, stoppedEarly,
  };
}

/**
 * Post-weathering micro-relief: differential etching at the grain scale.
 *
 * A 72^3 grid over a 1 m boulder has 14 mm cells and simply cannot resolve a
 * 3 mm crystal, so grain-scale relief is applied at mesh level, displacing
 * vertices along the normal by the local hardness contrast. On a weathered
 * granite quartz stands proud by a few tenths of a millimetre and biotite pits
 * recede; grain boundaries are etched grooves regardless of species, because
 * hydrolysis fronts run along them.
 */
export function microRelief(positions, normals, grains, amount = 1.0) {
  const out = new Float32Array(positions.length);
  const nv = positions.length / 3;
  for (let v = 0; v < nv; v++) {
    const x = positions[v * 3], y = positions[v * 3 + 1], z = positions[v * 3 + 2];
    const s = grains.sample(x, y, z);
    const rel = (s.mineral.hardness - 5.6) / 4.5;     // relative to feldspar
    const groove = -(1 - s.boundary) * 0.35;
    const d = (rel + groove) * s.size * 0.16 * amount;
    out[v * 3] = x + normals[v * 3] * d;
    out[v * 3 + 1] = y + normals[v * 3 + 1] * d;
    out[v * 3 + 2] = z + normals[v * 3 + 2] * d;
  }
  return out;
}
