/**
 * Uniform scalar field on a regular grid, used as the signed-distance
 * substrate for the weathering solver.
 *
 * Convention: phi < 0 inside the rock, phi > 0 in air, |grad phi| = 1 when the
 * field is a true distance field. Outward surface retreat (erosion) therefore
 * corresponds to *increasing* phi.
 */
export class Field3 {
  /**
   * @param {number} n      resolution per axis (cubic grid)
   * @param {number} extent world-space half-size of the domain, metres
   */
  constructor(n, extent) {
    this.n = n;
    this.extent = extent;
    this.h = (2 * extent) / (n - 1);
    this.origin = -extent;
    this.data = new Float32Array(n * n * n);
  }
  idx(i, j, k) { const n = this.n; return (k * n + j) * n + i; }
  at(i, j, k) { return this.data[(k * this.n + j) * this.n + i]; }
  set(i, j, k, v) { this.data[(k * this.n + j) * this.n + i] = v; }
  /** world coordinate of lattice site along one axis */
  coord(i) { return this.origin + i * this.h; }

  clone() {
    const f = new Field3(this.n, this.extent);
    f.data.set(this.data);
    return f;
  }

  /** Fill from an analytic function f(x,y,z). */
  fill(fn) {
    const n = this.n;
    for (let k = 0; k < n; k++) {
      const z = this.coord(k);
      for (let j = 0; j < n; j++) {
        const y = this.coord(j);
        for (let i = 0; i < n; i++) {
          this.data[(k * n + j) * n + i] = fn(this.coord(i), y, z);
        }
      }
    }
    return this;
  }

  /** Trilinear sample at a world position, clamped at the boundary. */
  sample(x, y, z) {
    const n = this.n, h = this.h;
    let fx = (x - this.origin) / h, fy = (y - this.origin) / h, fz = (z - this.origin) / h;
    fx = Math.min(n - 1.0001, Math.max(0, fx));
    fy = Math.min(n - 1.0001, Math.max(0, fy));
    fz = Math.min(n - 1.0001, Math.max(0, fz));
    const i = fx | 0, j = fy | 0, k = fz | 0;
    const tx = fx - i, ty = fy - j, tz = fz - k;
    const d = this.data;
    const s = (ii, jj, kk) => d[(kk * n + jj) * n + ii];
    const c00 = s(i, j, k) * (1 - tx) + s(i + 1, j, k) * tx;
    const c10 = s(i, j + 1, k) * (1 - tx) + s(i + 1, j + 1, k) * tx;
    const c01 = s(i, j, k + 1) * (1 - tx) + s(i + 1, j, k + 1) * tx;
    const c11 = s(i, j + 1, k + 1) * (1 - tx) + s(i + 1, j + 1, k + 1) * tx;
    const c0 = c00 * (1 - ty) + c10 * ty;
    const c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
  }

  /** Central-difference gradient at a lattice site (clamped stencil). */
  gradAt(i, j, k, out = [0, 0, 0]) {
    const n = this.n, d = this.data, h2 = 2 * this.h;
    const c = (a) => Math.min(n - 1, Math.max(0, a));
    const g = (ii, jj, kk) => d[(c(kk) * n + c(jj)) * n + c(ii)];
    out[0] = (g(i + 1, j, k) - g(i - 1, j, k)) / h2;
    out[1] = (g(i, j + 1, k) - g(i, j - 1, k)) / h2;
    out[2] = (g(i, j, k + 1) - g(i, j, k - 1)) / h2;
    return out;
  }

  /** Numerical gradient at an arbitrary world point. */
  gradient(x, y, z, eps = this.h * 0.75, out = [0, 0, 0]) {
    out[0] = (this.sample(x + eps, y, z) - this.sample(x - eps, y, z)) / (2 * eps);
    out[1] = (this.sample(x, y + eps, z) - this.sample(x, y - eps, z)) / (2 * eps);
    out[2] = (this.sample(x, y, z + eps) - this.sample(x, y, z - eps)) / (2 * eps);
    return out;
  }
}

/**
 * Mean curvature of the level set, kappa = div(grad phi / |grad phi|).
 *
 * Expanded for a general (not necessarily unit-gradient) phi so it stays valid
 * between re-distancing passes:
 *
 *   kappa = ( (phi_yy+phi_zz) phi_x^2 + (phi_xx+phi_zz) phi_y^2 + (phi_xx+phi_yy) phi_z^2
 *             - 2 phi_x phi_y phi_xy - 2 phi_x phi_z phi_xz - 2 phi_y phi_z phi_yz )
 *           / |grad phi|^3
 *
 * Sign convention: for phi<0 inside, a convex protrusion (corner of a block)
 * has kappa > 0. This is exactly the quantity that drives spheroidal
 * weathering -- corners see three faces of attack, edges two, faces one -- and
 * it is the term Beardall et al. / Jones et al. use for curvature-controlled
 * weathering of concave rock.
 */
export function meanCurvatureField(phi, out, band = null) {
  const n = phi.n, h = phi.h, d = phi.data;
  out = out || new Float32Array(n * n * n);
  const cl = (a) => Math.min(n - 1, Math.max(0, a));
  const g = (i, j, k) => d[(cl(k) * n + cl(j)) * n + cl(i)];
  const h2 = 2 * h, hh = h * h, h4 = 4 * h * h;

  /**
   * Fast path for band cells, which buildBand guarantees are interior
   * (1 <= i,j,k <= n-2). The full 19-point stencil is then addressable by
   * constant offsets from the centre index, with no per-access clamping or
   * index arithmetic. The clamped `kernel` below stays for the whole-grid case.
   */
  const nn = n * n;
  const kernelFast = (id) => {
    const c = d[id];
    const px = (d[id + 1] - d[id - 1]) / h2;
    const py = (d[id + n] - d[id - n]) / h2;
    const pz = (d[id + nn] - d[id - nn]) / h2;
    const m2 = px * px + py * py + pz * pz;
    if (m2 < 1e-10) { out[id] = 0; return; }
    const pxx = (d[id + 1] - 2 * c + d[id - 1]) / hh;
    const pyy = (d[id + n] - 2 * c + d[id - n]) / hh;
    const pzz = (d[id + nn] - 2 * c + d[id - nn]) / hh;
    const pxy = (d[id + 1 + n] - d[id + 1 - n] - d[id - 1 + n] + d[id - 1 - n]) / h4;
    const pxz = (d[id + 1 + nn] - d[id + 1 - nn] - d[id - 1 + nn] + d[id - 1 - nn]) / h4;
    const pyz = (d[id + n + nn] - d[id + n - nn] - d[id - n + nn] + d[id - n - nn]) / h4;
    const num = (pyy + pzz) * px * px + (pxx + pzz) * py * py + (pxx + pyy) * pz * pz
      - 2 * (px * py * pxy + px * pz * pxz + py * pz * pyz);
    out[id] = num / (m2 * Math.sqrt(m2));
  };

  const kernel = (i, j, k) => {
        const c = g(i, j, k);
        const px = (g(i + 1, j, k) - g(i - 1, j, k)) / h2;
        const py = (g(i, j + 1, k) - g(i, j - 1, k)) / h2;
        const pz = (g(i, j, k + 1) - g(i, j, k - 1)) / h2;
        const m2 = px * px + py * py + pz * pz;
        if (m2 < 1e-10) { out[(k * n + j) * n + i] = 0; return; }
        const pxx = (g(i + 1, j, k) - 2 * c + g(i - 1, j, k)) / hh;
        const pyy = (g(i, j + 1, k) - 2 * c + g(i, j - 1, k)) / hh;
        const pzz = (g(i, j, k + 1) - 2 * c + g(i, j, k - 1)) / hh;
        const pxy = (g(i + 1, j + 1, k) - g(i + 1, j - 1, k) - g(i - 1, j + 1, k) + g(i - 1, j - 1, k)) / h4;
        const pxz = (g(i + 1, j, k + 1) - g(i + 1, j, k - 1) - g(i - 1, j, k + 1) + g(i - 1, j, k - 1)) / h4;
        const pyz = (g(i, j + 1, k + 1) - g(i, j + 1, k - 1) - g(i, j - 1, k + 1) + g(i, j - 1, k - 1)) / h4;
        const num = (pyy + pzz) * px * px + (pxx + pzz) * py * py + (pxx + pyy) * pz * pz
          - 2 * (px * py * pxy + px * pz * pxz + py * pz * pyz);
        out[(k * n + j) * n + i] = num / Math.pow(m2, 1.5);
  };

  if (band) {
    const BI = band.idx;
    for (let b = 0; b < band.length; b++) kernelFast(BI[b]);
  } else {
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) kernel(i, j, k);
  }
  return out;
}

/**
 * Godunov upwind approximation of |grad phi| for a motion in the +normal
 * (erosive) direction. Using the correct one-sided differences here is what
 * keeps a retreating front from developing the staircase artefacts that
 * central differencing produces on sharp joint edges.
 */
/**
 * Direct-index Godunov, for interior cells only (as guaranteed by buildBand).
 * Same maths as godunovNormGrowth, without the clamping closure -- this is the
 * innermost operation of the whole solver.
 */
export function godunovNormGrowthFast(d, n, nn, h, id) {
  const c = d[id];
  const dxm = (c - d[id - 1]) / h, dxp = (d[id + 1] - c) / h;
  const dym = (c - d[id - n]) / h, dyp = (d[id + n] - c) / h;
  const dzm = (c - d[id - nn]) / h, dzp = (d[id + nn] - c) / h;
  const a = dxm > 0 ? dxm : 0, b = dxp < 0 ? dxp : 0;
  const cc = dym > 0 ? dym : 0, dd = dyp < 0 ? dyp : 0;
  const e = dzm > 0 ? dzm : 0, f = dzp < 0 ? dzp : 0;
  const ax = a * a > b * b ? a * a : b * b;
  const ay = cc * cc > dd * dd ? cc * cc : dd * dd;
  const az = e * e > f * f ? e * e : f * f;
  return Math.sqrt(ax + ay + az);
}

export function godunovNormGrowth(d, n, h, i, j, k) {
  const cl = (a) => Math.min(n - 1, Math.max(0, a));
  const g = (ii, jj, kk) => d[(cl(kk) * n + cl(jj)) * n + cl(ii)];
  const c = g(i, j, k);
  const dxm = (c - g(i - 1, j, k)) / h, dxp = (g(i + 1, j, k) - c) / h;
  const dym = (c - g(i, j - 1, k)) / h, dyp = (g(i, j + 1, k) - c) / h;
  const dzm = (c - g(i, j, k - 1)) / h, dzp = (g(i, j, k + 1) - c) / h;
  const a = Math.max(dxm, 0), b = Math.min(dxp, 0);
  const cc = Math.max(dym, 0), dd = Math.min(dyp, 0);
  const e = Math.max(dzm, 0), f = Math.min(dzp, 0);
  return Math.sqrt(Math.max(a * a, b * b) + Math.max(cc * cc, dd * dd) + Math.max(e * e, f * f));
}

/**
 * Re-distancing by fast sweeping (Zhao 2005): solve the Eikonal equation
 * |grad phi| = 1 keeping the zero level set fixed. Called periodically during
 * the erosion run, because the curvature term steepens/flattens the field and
 * a non-metric phi biases both the curvature estimate and the CFL bound.
 */
/**
 * Scratch buffers for redistance(). The solver calls this every few steps, and
 * allocating two N^3 arrays per call (2 MB at 64^3) made the GC, not the
 * arithmetic, the dominant cost. Keyed by size so a batch of mixed resolutions
 * still reuses correctly.
 */
const _rdCache = new Map();
function rdScratch(size) {
  let s = _rdCache.get(size);
  if (!s) { s = { out: new Float32Array(size), frozen: new Uint8Array(size) }; _rdCache.set(size, s); }
  return s;
}

export function redistance(field, sweeps = 2, maxDist = Infinity) {
  const n = field.n, h = field.h, d = field.data;
  const LARGE = 1e6;
  // We only ever need phi to be metric inside the narrow band; capping the
  // solved distance lets the sweep bail out early on the ~75% of cells that are
  // deep inside or far outside the rock, which is the difference between a
  // 1 second re-distance and a 0.2 second one.
  const cap = Number.isFinite(maxDist) ? maxDist : LARGE;
  const _s = rdScratch(n * n * n);
  const out = _s.out, frozen = _s.frozen;
  frozen.fill(0);

  // Seed: cells adjacent to the interface get a first-order subcell distance.
  const idx = (i, j, k) => (k * n + j) * n + i;
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const id = idx(i, j, k);
    const c = d[id];
    let isBoundary = false, minD = LARGE;
    const neigh = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    for (const [a, b, cc2] of neigh) {
      const ii = i + a, jj = j + b, kk = k + cc2;
      if (ii < 0 || jj < 0 || kk < 0 || ii >= n || jj >= n || kk >= n) continue;
      const v = d[idx(ii, jj, kk)];
      if ((c <= 0 && v > 0) || (c > 0 && v <= 0)) {
        isBoundary = true;
        const t = Math.abs(c) / Math.max(1e-9, Math.abs(c - v));
        minD = Math.min(minD, t * h);
      }
    }
    if (isBoundary) { out[id] = minD; frozen[id] = 1; }
    else out[id] = LARGE;
  }

  const solve = (i, j, k) => {
    const id = idx(i, j, k);
    if (frozen[id]) return;
    const gx = Math.min(i > 0 ? out[idx(i - 1, j, k)] : LARGE, i < n - 1 ? out[idx(i + 1, j, k)] : LARGE);
    const gy = Math.min(j > 0 ? out[idx(i, j - 1, k)] : LARGE, j < n - 1 ? out[idx(i, j + 1, k)] : LARGE);
    const gz = Math.min(k > 0 ? out[idx(i, j, k - 1)] : LARGE, k < n - 1 ? out[idx(i, j, k + 1)] : LARGE);
    if (gx >= cap && gy >= cap && gz >= cap) return;
    const a = [gx, gy, gz].sort((p, q) => p - q);
    // 1D
    let x = a[0] + h;
    if (x > a[1]) {
      // 2D
      const s = a[0] + a[1];
      const disc = 2 * h * h - (a[0] - a[1]) * (a[0] - a[1]);
      if (disc >= 0) x = 0.5 * (s + Math.sqrt(disc));
      if (x > a[2]) {
        // 3D
        const s3 = a[0] + a[1] + a[2];
        const q = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        const disc3 = s3 * s3 - 3 * (q - h * h);
        if (disc3 >= 0) x = (s3 + Math.sqrt(disc3)) / 3;
      }
    }
    if (x > cap) x = cap;
    if (x < out[id]) out[id] = x;
  };

  // When the distance is capped we only need to sweep the slab the wavefront
  // can actually reach: the interface bounding box grown by `cap`. On a boulder
  // occupying part of the domain this is a large constant-factor saving, and it
  // is exact -- cells outside cannot be reached by a capped wavefront anyway.
  let lo0 = 0, lo1 = 0, lo2 = 0, hi0 = n - 1, hi1 = n - 1, hi2 = n - 1;
  if (Number.isFinite(maxDist)) {
    let bi0 = n, bj0 = n, bk0 = n, bi1 = -1, bj1 = -1, bk1 = -1;
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) {
      const row = k * n * n + j * n;
      for (let i = 0; i < n; i++) {
        if (!frozen[row + i]) continue;
        if (i < bi0) bi0 = i; if (i > bi1) bi1 = i;
        if (j < bj0) bj0 = j; if (j > bj1) bj1 = j;
        if (k < bk0) bk0 = k; if (k > bk1) bk1 = k;
      }
    }
    if (bi1 >= 0) {
      const pad = Math.ceil(cap / h) + 2;
      lo0 = Math.max(0, bi0 - pad); hi0 = Math.min(n - 1, bi1 + pad);
      lo1 = Math.max(0, bj0 - pad); hi1 = Math.min(n - 1, bj1 + pad);
      lo2 = Math.max(0, bk0 - pad); hi2 = Math.min(n - 1, bk1 + pad);
    }
  }

  for (let s = 0; s < sweeps; s++) {
    for (let dir = 0; dir < 8; dir++) {
      const xi = (dir & 1) ? -1 : 1, yi = (dir & 2) ? -1 : 1, zi = (dir & 4) ? -1 : 1;
      const i0 = xi > 0 ? lo0 : hi0, i1 = xi > 0 ? hi0 + 1 : lo0 - 1;
      const j0 = yi > 0 ? lo1 : hi1, j1 = yi > 0 ? hi1 + 1 : lo1 - 1;
      const k0 = zi > 0 ? lo2 : hi2, k1 = zi > 0 ? hi2 + 1 : lo2 - 1;
      for (let k = k0; k !== k1; k += zi)
        for (let j = j0; j !== j1; j += yi)
          for (let i = i0; i !== i1; i += xi) solve(i, j, k);
    }
  }

  // Cells the capped wavefront never reached stay at LARGE; clamp them to the
  // cap so the field is a bounded plateau outside the band rather than garbage.
  for (let p = 0; p < d.length; p++) {
    const v = Math.min(out[p], cap);
    d[p] = d[p] <= 0 ? -v : v;
  }
  return field;
}
