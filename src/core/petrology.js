/**
 * Granitoid petrology: a modal mineralogy + crystal-aggregate solid texture.
 *
 * WHY THIS AND NOT NOISE
 * ----------------------
 * Granite is a holocrystalline, equigranular/seriate aggregate of interlocking
 * crystals that nucleated and grew until they impinged on each other. The
 * correct geometric idealisation of that process is a *Laguerre (radical
 * Voronoi) tessellation* of nucleation sites with per-crystal weights -- not
 * fBm noise. fBm gives you scale-free clouds; a real granite gives you
 * polygonal, sharply bounded grains of a log-normal size distribution with
 * ~120-degree triple junctions, and every visual property (colour mottling,
 * specular cleavage flashes, differential micro-relief, grussification) keys
 * off *which grain you are standing on*, not off a smooth field.
 *
 * So the base texture here is:
 *   1. a jittered lattice of nucleation sites -> Laguerre cells = crystals;
 *   2. per-crystal mineral species drawn from a modal composition (QAP);
 *   3. per-crystal orientation (cleavage plane normal) and colour jitter;
 *   4. a *second*, finer aggregate blended in to produce seriate (continuous
 *      grain-size) rather than perfectly equigranular fabric, plus optional
 *      K-feldspar phenocrysts (porphyritic texture);
 *   5. only *then* a small amount of band-limited noise, used strictly for
 *      sub-grain effects that really are stochastic: sericite clouding on
 *      plagioclase, perthitic exsolution lamellae, micro-fracture staining.
 *
 * The identical field is evaluated in GLSL (gpu/glsl/grain.glsl.js) so the
 * erosion solver and the renderer agree on where every crystal is.
 *
 * MODAL DATA
 * ----------
 * Modes below are volume fractions in the QAP sense for common granitoids;
 * grain sizes follow the standard igneous classes (fine 0.2-1 mm, medium
 * 1-5 mm, coarse 5-15 mm). Colours are approximate linear-sRGB diffuse
 * albedos for clean, dry, freshly broken surfaces.
 */

import { hash3i, hashU32 } from './rng.js';

/**
 * Mineral table.
 *  albedo     : linear RGB base colour of a fresh cleavage/fracture surface
 *  rough      : base GGX roughness of a fracture surface
 *  spec       : dielectric F0 scale (proxy for refractive index; quartz 1.55,
 *               feldspar 1.53, biotite 1.6 but micas read as sheen)
 *  hardness   : Mohs. Drives micro-relief: harder grains stand proud.
 *  durability : resistance to chemical weathering in the Goldich sense.
 *               Quartz ~ inert; biotite oxidises first and is the trigger for
 *               spheroidal fracturing and grussification in granitoids.
 *  cleavage   : 0 = conchoidal fracture (quartz), 1 = one perfect plane (mica),
 *               2 = two planes at ~90 deg (feldspar)
 *  fe         : ferrous iron content -> drives goethite/limonite staining.
 */
export const MINERALS = {
  quartz:      { id: 0, albedo: [0.42, 0.43, 0.44], rough: 0.34, spec: 0.55, hardness: 7.0, durability: 1.00, cleavage: 0, fe: 0.0, translucency: 0.55 },
  kfeldspar:   { id: 1, albedo: [0.62, 0.40, 0.34], rough: 0.30, spec: 0.50, hardness: 6.0, durability: 0.62, cleavage: 2, fe: 0.02, translucency: 0.12 },
  plagioclase: { id: 2, albedo: [0.70, 0.69, 0.66], rough: 0.36, spec: 0.50, hardness: 6.0, durability: 0.45, cleavage: 2, fe: 0.03, translucency: 0.15 },
  biotite:     { id: 3, albedo: [0.045, 0.038, 0.032], rough: 0.18, spec: 0.62, hardness: 2.75, durability: 0.10, cleavage: 1, fe: 0.85, translucency: 0.02 },
  muscovite:   { id: 4, albedo: [0.58, 0.56, 0.50], rough: 0.12, spec: 0.58, hardness: 2.5, durability: 0.35, cleavage: 1, fe: 0.05, translucency: 0.30 },
  hornblende:  { id: 5, albedo: [0.055, 0.070, 0.055], rough: 0.28, spec: 0.58, hardness: 5.5, durability: 0.28, cleavage: 2, fe: 0.70, translucency: 0.02 },
};

export const MINERAL_LIST = Object.keys(MINERALS).map((k) => ({ name: k, ...MINERALS[k] }))
  .sort((a, b) => a.id - b.id);

/**
 * Named granitoid lithologies. `mode` entries are volume fractions and are
 * renormalised on use. `grain` is the median long axis in metres.
 */
export const LITHOLOGIES = {
  'biotite-granite': {
    label: 'Biotite granite (Cape/Sierra type)',
    mode: { quartz: 0.28, kfeldspar: 0.33, plagioclase: 0.27, biotite: 0.09, muscovite: 0.01, hornblende: 0.02 },
    grain: 0.0035, grainSigma: 0.45, seriate: 0.55,
    phenocryst: { frac: 0.05, size: 0.022, mineral: 'kfeldspar' },
    // Fe released by biotite/hornblende breakdown -> ochre rinds
    stain: [0.42, 0.24, 0.10], stainStrength: 0.7,
  },
  'pink-porphyritic': {
    label: 'Porphyritic pink granite',
    mode: { quartz: 0.26, kfeldspar: 0.42, plagioclase: 0.21, biotite: 0.08, muscovite: 0.01, hornblende: 0.02 },
    grain: 0.005, grainSigma: 0.5, seriate: 0.7,
    phenocryst: { frac: 0.14, size: 0.035, mineral: 'kfeldspar' },
    stain: [0.45, 0.22, 0.09], stainStrength: 0.6,
  },
  'granodiorite': {
    label: 'Granodiorite (Rio Blanco type)',
    mode: { quartz: 0.22, kfeldspar: 0.12, plagioclase: 0.44, biotite: 0.13, muscovite: 0.0, hornblende: 0.09 },
    grain: 0.003, grainSigma: 0.4, seriate: 0.45,
    phenocryst: { frac: 0.02, size: 0.014, mineral: 'plagioclase' },
    stain: [0.38, 0.22, 0.11], stainStrength: 0.9,
  },
  'leucogranite': {
    label: 'Two-mica leucogranite',
    mode: { quartz: 0.33, kfeldspar: 0.32, plagioclase: 0.28, biotite: 0.03, muscovite: 0.04, hornblende: 0.0 },
    grain: 0.0025, grainSigma: 0.35, seriate: 0.35,
    phenocryst: { frac: 0.0, size: 0.01, mineral: 'kfeldspar' },
    stain: [0.48, 0.36, 0.20], stainStrength: 0.25,
  },
  'gneissic-granite': {
    label: 'Gneissose granite (foliated)',
    mode: { quartz: 0.27, kfeldspar: 0.28, plagioclase: 0.28, biotite: 0.13, muscovite: 0.02, hornblende: 0.02 },
    grain: 0.004, grainSigma: 0.55, seriate: 0.6,
    phenocryst: { frac: 0.06, size: 0.026, mineral: 'kfeldspar' },
    foliation: 0.62,          // anisotropy: grains flattened in the S-plane
    stain: [0.40, 0.23, 0.10], stainStrength: 0.8,
  },
};

/** Build a 256-entry CDF lookup so mineral assignment is a single hash + scan. */
export function buildModeCDF(mode) {
  const entries = MINERAL_LIST.map((m) => [m.id, mode[m.name] || 0]);
  const total = entries.reduce((s, e) => s + e[1], 0) || 1;
  let acc = 0;
  const cdf = [];
  for (const [id, w] of entries) { acc += w / total; cdf.push([acc, id]); }
  cdf[cdf.length - 1][0] = 1.0;
  return cdf;
}

function pickFromCDF(cdf, u) {
  for (let i = 0; i < cdf.length; i++) if (u <= cdf[i][0]) return cdf[i][1];
  return cdf[cdf.length - 1][1];
}

/**
 * Crystal-aggregate evaluator.
 *
 * Returns, for a world point, the crystal it belongs to and that crystal's
 * mineral properties, plus the distance to the nearest grain boundary
 * (needed for boundary-preferential etching -- weathering fronts propagate
 * along grain boundaries, not through crystal interiors).
 */
export class GrainField {
  /**
   * @param {object} litho entry from LITHOLOGIES
   * @param {number} seed
   */
  constructor(litho, seed = 1) {
    this.litho = litho;
    this.seed = seed >>> 0;
    this.cdf = buildModeCDF(litho.mode);
    // Lattice cell size chosen so that a jittered site per cell reproduces the
    // requested median grain diameter.
    this.cell = litho.grain * 1.15;
    this.cell2 = this.cell * 0.42;       // fine sub-population for seriate fabric
    this.foliation = litho.foliation || 0;
    this.phen = litho.phenocryst || { frac: 0, size: 0.02, mineral: 'kfeldspar' };
    this.phenCell = this.phen.size * 1.3;
  }

  /** Anisotropic metric implementing tectonic foliation (flattened grains). */
  _warp(x, y, z) {
    const f = this.foliation;
    if (f <= 0) return [x, y, z];
    // S-plane is the XZ plane; grains flattened along Y, stretched along X.
    return [x / (1 + 0.55 * f), y * (1 + 1.35 * f), z / (1 + 0.15 * f)];
  }

  /**
   * Laguerre cell lookup on a jittered lattice.
   * Returns { id, d1, d2, site } where d1,d2 are the two smallest *weighted*
   * distances; (d2 - d1) is the grain-boundary proximity.
   */
  _cellAt(x, y, z, cellSize, salt) {
    const cs = cellSize;
    const gx = Math.floor(x / cs), gy = Math.floor(y / cs), gz = Math.floor(z / cs);
    let best = 1e30, second = 1e30, bestKey = 0, bx = 0, by = 0, bz = 0;
    for (let k = -1; k <= 1; k++) for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
      const cx = gx + i, cy = gy + j, cz = gz + k;
      const h0 = hash3i(cx, cy, cz, this.seed ^ salt);
      const h1 = hashU32(h0 ^ 0x9e3779b9);
      const h2 = hashU32(h1 ^ 0x85ebca6b);
      const h3 = hashU32(h2 ^ 0xc2b2ae35);
      const px = (cx + (h0 / 4294967296)) * cs;
      const py = (cy + (h1 / 4294967296)) * cs;
      const pz = (cz + (h2 / 4294967296)) * cs;
      // Laguerre weight: log-normal crystal radius. sigma from the lithology.
      const u = (h3 / 4294967296) * 2 - 1;
      const w = Math.exp(this.litho.grainSigma * u) * cs * 0.5;
      const dx = px - x, dy = py - y, dz = pz - z;
      const dd = Math.sqrt(dx * dx + dy * dy + dz * dz) - w;   // radical distance
      if (dd < best) {
        second = best; best = dd; bestKey = h0; bx = px; by = py; bz = pz;
      } else if (dd < second) second = dd;
    }
    return { key: bestKey, d1: best, d2: second, sx: bx, sy: by, sz: bz };
  }

  /**
   * Full sample. `x,y,z` in object space, metres.
   * @returns {{mineral:object, mineralId:number, boundary:number, cid:number,
   *            axis:number[], jitter:number, size:number}}
   */
  sample(x, y, z) {
    const [wx, wy, wz] = this._warp(x, y, z);

    // Phenocrysts: sparse large crystals that override the groundmass.
    if (this.phen.frac > 0) {
      const pc = this._cellAt(wx, wy, wz, this.phenCell, 0x51ed27);
      const occupy = (hashU32(pc.key ^ 0x1b873593) / 4294967296) < this.phen.frac;
      const r = this.phen.size * 0.5 * (0.7 + 0.6 * (hashU32(pc.key ^ 0x27d4eb2f) / 4294967296));
      const dist = Math.hypot(pc.sx - wx, pc.sy - wy, pc.sz - wz);
      if (occupy && dist < r) {
        const m = MINERALS[this.phen.mineral];
        return this._pack(m, pc.key, Math.min(1, (r - dist) / (0.25 * r)), this.phen.size, true);
      }
    }

    // Two grain populations blended by a hashed coin per coarse cell gives a
    // seriate (continuous size) fabric rather than an unnaturally uniform one.
    const coarse = this._cellAt(wx, wy, wz, this.cell, 0);
    let cellInfo = coarse, size = this.cell;
    if (this.litho.seriate > 0) {
      const coin = hashU32(coarse.key ^ 0x7ed55d16) / 4294967296;
      if (coin < this.litho.seriate * 0.45) {
        cellInfo = this._cellAt(wx, wy, wz, this.cell2, 0x2545f4);
        size = this.cell2;
      }
    }
    const u = hashU32(cellInfo.key ^ 0xa5a5a5a5) / 4294967296;
    const mid = pickFromCDF(this.cdf, u);
    const m = MINERAL_LIST[mid];
    const boundary = Math.max(0, Math.min(1, (cellInfo.d2 - cellInfo.d1) / (0.35 * size)));
    return this._pack(m, cellInfo.key, boundary, size, false);
  }

  _pack(m, key, boundary, size, isPheno) {
    const h1 = hashU32(key ^ 0x165667b1) / 4294967296;
    const h2 = hashU32(key ^ 0x9e3779b1) / 4294967296;
    const h3 = hashU32(key ^ 0x27d4eb2d) / 4294967296;
    // cleavage-plane normal in object space; feldspar/mica flash specularly
    // when this aligns with the halfvector, which is the single most
    // recognisable optical cue of a coarse granite in sunlight.
    const th = Math.acos(2 * h1 - 1), ph = h2 * Math.PI * 2;
    const axis = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    return {
      mineral: m, mineralId: m.id, boundary, cid: key, axis,
      jitter: h3, size, phenocryst: isPheno,
    };
  }

  /**
   * Scalar durability in [0,1] used by the erosion solver. Combines mineral
   * resistance with grain-boundary weakness: hydrolysis fronts run along
   * boundaries and along biotite cleavage, so boundaries are softer than
   * crystal interiors regardless of species.
   */
  durability(x, y, z) {
    const s = this.sample(x, y, z);
    const boundaryWeak = 1 - 0.45 * (1 - s.boundary);
    return Math.max(0.02, s.mineral.durability * boundaryWeak);
  }

  /**
   * Durability UPSCALED to a finite cell of side `cell`.
   *
   * This distinction is not pedantry, it is the difference between a boulder
   * and a lump of noise. The erosion grid has ~30 mm cells; crystals are ~3 mm.
   * Point-sampling `durability()` per cell therefore returns an essentially
   * uncorrelated draw from the modal distribution at every cell -- white noise
   * at the grid scale. Feeding that to the solver makes neighbouring cells
   * retreat at wildly different rates, and the surface grows grid-scale fuzz:
   * volume falls while surface area stays flat, which is the exact opposite of
   * what weathering does to a rock (a weathering boulder loses area faster than
   * volume, that is what "rounding" means).
   *
   * The physically correct grid-scale quantity is the volume-averaged
   * resistance of the crystals inside the cell -- a Reuss-style harmonic mean,
   * since the weathering front advances fastest through the weakest connected
   * path. Averaging collapses the crystal-scale variance by ~1/sqrt(N) where N
   * is the number of crystals per cell, which is exactly the physical statement
   * that a rock mass containing thousands of crystals per litre does not erode
   * at the rate of its softest crystal.
   *
   * Genuine grid-scale variation comes from *mesoscale* heterogeneity instead:
   * mafic schlieren, aplite veins, alteration haloes along microfractures.
   * Those are added as a separate low-frequency field by the caller.
   *
   * Crystal-scale relief is not lost -- it reappears at mesh level in
   * microRelief(), where the resolution can actually carry it.
   */
  durabilityCell(x, y, z, cell) {
    // QUADRATURE, not physics. A 30 mm cell holds ~640 crystals of 3.5 mm, so
    // the converged Reuss mean varies very little from cell to cell (measured
    // sd 0.016). A sparse estimate does NOT reproduce that: the sampling error
    // of an M-sample mean falls only as sqrt(M/N), so the old 2x2x2 stencil
    // returned sd 0.133 -- 8.2x the real variation, and *uncorrelated* between
    // neighbouring cells because each cell drew a different set of crystals.
    //
    // That is white noise in the speed field. It pits the surface at exactly
    // the grid frequency the level set cannot resolve, adding area while the
    // corners are still trying to round: sphericity fell with age instead of
    // rising. The cure is to converge the estimator, not to clamp the result.
    // 4x4x4 stratified cuts the sampling error to ~1/3 for 8x the samples, and
    // this is evaluated once per band cell per redistance interval, not per
    // step, so the cost is not on the critical path.
    const S = 4;
    const step = cell / S;
    const base = -0.5 * cell + 0.5 * step;
    let inv = 0;
    for (let k = 0; k < S; k++)
      for (let j = 0; j < S; j++)
        for (let i = 0; i < S; i++) {
          inv += 1 / this.durability(x + base + i * step,
                                     y + base + j * step,
                                     z + base + k * step);
        }
    return (S * S * S) / inv;   // harmonic (Reuss) mean
  }
}
