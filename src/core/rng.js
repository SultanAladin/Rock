/**
 * Deterministic 32-bit integer hashing + PRNG.
 *
 * The same hash is implemented bit-for-bit in GLSL (see gpu/glsl/hash.glsl.js)
 * so that the CPU-side grain field used for durability / vertex colours and the
 * GPU-side grain field used for shading agree exactly. That agreement is the
 * whole point: the crystal that erodes is the crystal you see.
 */

const U32 = 0xffffffff;

/** PCG-style output permutation on a single uint32. */
export function hashU32(x) {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x >>> 0;
}

export function hash3i(x, y, z, seed = 0) {
  let h = (Math.imul(x | 0, 0x8da6b343) ^
           Math.imul(y | 0, 0xd8163841) ^
           Math.imul(z | 0, 0xcb1ab31f) ^
           Math.imul(seed | 0, 0x165667b1)) >>> 0;
  return hashU32(h);
}

/** Three independent floats in [0,1) from an integer lattice site. */
export function hash3f(x, y, z, seed = 0) {
  const h0 = hash3i(x, y, z, seed);
  const h1 = hashU32(h0 ^ 0x9e3779b9);
  const h2 = hashU32(h1 ^ 0x85ebca6b);
  return [h0 / 4294967296, h1 / 4294967296, h2 / 4294967296];
}

/** Small, fast, seedable stream PRNG (xoshiro128**-lite / sfc32). */
export class RNG {
  constructor(seed = 1) {
    let s = hashU32(seed >>> 0) || 1;
    this.a = s;
    this.b = hashU32(s ^ 0xdeadbeef) || 2;
    this.c = hashU32(s ^ 0x1234567) || 3;
    this.d = hashU32(s ^ 0xfeedface) || 4;
    for (let i = 0; i < 12; i++) this.next();
  }
  next() {
    // sfc32
    const t = (this.a + this.b) >>> 0;
    this.a = (this.b ^ (this.b >>> 9)) >>> 0;
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = ((this.c << 21) | (this.c >>> 11)) >>> 0;
    this.c = (this.c + t) >>> 0;
    this.d = (this.d + 1) >>> 0;
    const r = (t + this.d) >>> 0;
    return r / 4294967296;
  }
  /** Uniform in [lo,hi). */
  range(lo, hi) { return lo + (hi - lo) * this.next(); }
  int(n) { return Math.min(n - 1, (this.next() * n) | 0); }
  /** Standard normal via Box-Muller (cached pair). */
  normal() {
    if (this._spare !== undefined) { const v = this._spare; this._spare = undefined; return v; }
    let u = 0, v = 0, s = 0;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt(-2 * Math.log(s) / s);
    this._spare = v * f;
    return u * f;
  }
  /** Log-normal with given median and sigma of the underlying normal. */
  logNormal(median, sigma) { return median * Math.exp(sigma * this.normal()); }
  /** Unit vector, uniform on the sphere. */
  unitVector() {
    const z = this.next() * 2 - 1;
    const t = this.next() * Math.PI * 2;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    return [r * Math.cos(t), r * Math.sin(t), z];
  }
  /**
   * Fisher (von Mises-Fisher on S2) distributed direction about `axis` with
   * concentration kappa. Joint-set pole dispersion in structural geology is
   * conventionally reported as a Fisher K value, so this is the right sampler
   * for perturbing joint normals.
   */
  fisher(axis, kappa) {
    if (kappa < 1e-3) return this.unitVector();
    const u = this.next();
    const w = 1 + Math.log(u + (1 - u) * Math.exp(-2 * kappa)) / kappa;
    const ang = this.next() * Math.PI * 2;
    const s = Math.sqrt(Math.max(0, 1 - w * w));
    // build orthonormal basis around axis
    const a = axis;
    let t0 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let e1 = cross(t0, a); normalize(e1);
    let e2 = cross(a, e1);
    return [
      s * Math.cos(ang) * e1[0] + s * Math.sin(ang) * e2[0] + w * a[0],
      s * Math.cos(ang) * e1[1] + s * Math.sin(ang) * e2[1] + w * a[1],
      s * Math.cos(ang) * e1[2] + s * Math.sin(ang) * e2[2] + w * a[2],
    ];
  }
}

export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  v[0] /= l; v[1] /= l; v[2] /= l;
  return v;
}
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export { U32 };
