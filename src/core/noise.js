/**
 * Band-limited gradient noise and a *spectrally correct* self-affine fBm.
 *
 * Note the distinction from ordinary "turbulence": for a self-affine surface
 * with Hurst exponent H the amplitude of octave i must scale as
 * lacunarity^(-i*H), and the resulting surface has a power spectral density
 * ~ k^(-(2H+1)) in 1D profile terms. Granite mode-I fracture surfaces measure
 * H ~ 0.75-0.85 across five decades of scale, so this is not an aesthetic
 * choice -- it is the measured statistic of the thing we are drawing.
 */

import { hash3i, hashU32 } from './rng.js';

function grad2(ix, iy, seed) {
  const h = hash3i(ix, iy, 0, seed);
  const a = (h / 4294967296) * Math.PI * 2;
  return [Math.cos(a), Math.sin(a)];
}
function grad3(ix, iy, iz, seed) {
  const h = hash3i(ix, iy, iz, seed);
  const h2 = hashU32(h ^ 0x9e3779b9);
  const z = (h / 4294967296) * 2 - 1;
  const t = (h2 / 4294967296) * Math.PI * 2;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(t), r * Math.sin(t), z];
}
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

/** Perlin-style gradient noise in 2D, output roughly in [-1,1]. */
export function valueNoise2(x, y, seed = 0) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const u = fade(fx), v = fade(fy);
  const d = (gx, gy) => {
    const g = grad2(ix + gx, iy + gy, seed);
    return g[0] * (fx - gx) + g[1] * (fy - gy);
  };
  const n00 = d(0, 0), n10 = d(1, 0), n01 = d(0, 1), n11 = d(1, 1);
  return ((n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v) * 1.4142;
}

/** Perlin-style gradient noise in 3D. */
export function valueNoise3(x, y, z, seed = 0) {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const u = fade(fx), v = fade(fy), w = fade(fz);
  const d = (gx, gy, gz) => {
    const g = grad3(ix + gx, iy + gy, iz + gz, seed);
    return g[0] * (fx - gx) + g[1] * (fy - gy) + g[2] * (fz - gz);
  };
  const lerp = (a, b, t) => a + (b - a) * t;
  const n000 = d(0, 0, 0), n100 = d(1, 0, 0), n010 = d(0, 1, 0), n110 = d(1, 1, 0);
  const n001 = d(0, 0, 1), n101 = d(1, 0, 1), n011 = d(0, 1, 1), n111 = d(1, 1, 1);
  const x00 = lerp(n000, n100, u), x10 = lerp(n010, n110, u);
  const x01 = lerp(n001, n101, u), x11 = lerp(n011, n111, u);
  return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * 1.1547;
}

/**
 * Self-affine 2D fBm with Hurst exponent H. Unit-variance normalised so the
 * caller's `rough` parameter really is an RMS amplitude in metres.
 */
export function fbmSelfAffine(x, y, seed, H = 0.8, lac = 2.07, octaves = 5) {
  let sum = 0, amp = 1, norm = 0, f = 1;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise2(x * f, y * f, seed + i * 1013);
    norm += amp * amp;
    amp *= Math.pow(lac, -H);
    f *= lac;
  }
  return sum / Math.sqrt(norm || 1);
}

/** 3D counterpart, for volumetric fields (alteration fronts, joint haloes). */
export function fbm3(x, y, z, seed, H = 0.85, lac = 2.03, octaves = 4) {
  let sum = 0, amp = 1, norm = 0, f = 1;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise3(x * f, y * f, z * f, seed + i * 7919);
    norm += amp * amp;
    amp *= Math.pow(lac, -H);
    f *= lac;
  }
  return sum / Math.sqrt(norm || 1);
}
