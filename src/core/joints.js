/**
 * Structural stage: cut a fresh joint block out of the rock mass.
 *
 * Real boulders are not "a sphere plus noise". A granitoid boulder starts life
 * as a *block bounded by joints* -- planar discontinuities produced by cooling
 * contraction, unloading (sheeting/exfoliation joints, which are curved and
 * roughly parallel to the topographic surface), and tectonic conjugate sets.
 * Three roughly orthogonal sets give the equidimensional blocks that later
 * round into corestones; widely spaced parallel sets give slabs; a dominant
 * columnar set gives prisms.
 *
 * We therefore build the fresh block as a CSG intersection of half-spaces
 * (planar joints) and shallow spherical caps (sheeting joints), each with a
 * *rough* fracture surface. Joint-surface roughness is not white noise: natural
 * rock joints are self-affine, with a JRC-like roughness whose power spectrum
 * goes as k^(-2H-1) with Hurst exponent H ~ 0.8 (mode I tensile fracture
 * surfaces in granite are famously self-affine with H ~ 0.75-0.85). We
 * reproduce that with a spectrally-weighted sum of gradient-noise octaves at
 * exactly that exponent, and we scale its amplitude with grain size, because
 * fracture roughness in granite scales with the grain diameter (transgranular
 * vs intergranular crack paths).
 */

import { RNG, normalize, cross, dot } from './rng.js';
import { valueNoise3, fbmSelfAffine } from './noise.js';

/**
 * A joint set: mean pole direction, Fisher concentration, mean spacing.
 */
export class JointSet {
  constructor({ pole, kappa = 60, spacing = 0.8, spacingCV = 0.35, persistence = 1.0, roughness = 1.0 }) {
    this.pole = normalize(pole.slice());
    this.kappa = kappa;
    this.spacing = spacing;
    this.spacingCV = spacingCV;
    this.persistence = persistence;
    this.roughness = roughness;
  }
}

/** Standard structural configurations. */
export const JOINT_STYLES = {
  orthogonal: {
    label: 'Orthogonal 3-set (equidimensional blocks)',
    build: (rng, s) => [
      new JointSet({ pole: [1, 0, 0], kappa: 40, spacing: s * rng.range(0.85, 1.3) }),
      new JointSet({ pole: [0, 1, 0], kappa: 40, spacing: s * rng.range(0.85, 1.3) }),
      new JointSet({ pole: [0, 0, 1], kappa: 40, spacing: s * rng.range(0.85, 1.3) }),
    ],
  },
  sheeting: {
    label: 'Sheeting + 2 vertical sets (slabby)',
    build: (rng, s) => [
      new JointSet({ pole: [0, 1, 0], kappa: 220, spacing: s * rng.range(0.35, 0.6), roughness: 0.55 }),
      new JointSet({ pole: [1, 0, 0], kappa: 30, spacing: s * rng.range(1.0, 1.6) }),
      new JointSet({ pole: [0, 0, 1], kappa: 30, spacing: s * rng.range(1.0, 1.6) }),
    ],
  },
  columnar: {
    label: 'Columnar (hexagonal prisms)',
    build: (rng, s) => {
      const sets = [];
      const phase = rng.next() * Math.PI;
      for (let i = 0; i < 3; i++) {
        const a = phase + (i * Math.PI) / 3;
        sets.push(new JointSet({ pole: [Math.cos(a), 0, Math.sin(a)], kappa: 400, spacing: s * 0.8, roughness: 0.5 }));
      }
      sets.push(new JointSet({ pole: [0, 1, 0], kappa: 150, spacing: s * 2.2, roughness: 1.4 }));
      return sets;
    },
  },
  polyhedral: {
    label: 'Random polyhedral (angular blocks)',
    build: (rng, s) => {
      const sets = [];
      const n = 4 + rng.int(3);
      for (let i = 0; i < n; i++) {
        sets.push(new JointSet({ pole: rng.unitVector(), kappa: 6, spacing: s * rng.range(0.7, 1.5) }));
      }
      return sets;
    },
  },
  conjugate: {
    label: 'Conjugate shear pair + bedding',
    build: (rng, s) => {
      const strike = rng.next() * Math.PI;
      const dip = (rng.range(55, 75) * Math.PI) / 180;
      const mk = (sign) => {
        const a = strike + sign * 0.5;
        return normalize([Math.cos(a) * Math.sin(dip), Math.cos(dip) * sign, Math.sin(a) * Math.sin(dip)]);
      };
      return [
        new JointSet({ pole: mk(1), kappa: 90, spacing: s * rng.range(0.7, 1.0) }),
        new JointSet({ pole: mk(-1), kappa: 90, spacing: s * rng.range(0.7, 1.0) }),
        new JointSet({ pole: [0, 1, 0], kappa: 200, spacing: s * rng.range(0.9, 1.4), roughness: 0.6 }),
      ];
    },
  },
};

/**
 * A single planar joint face bounding the block, with self-affine roughness.
 */
class JointFace {
  constructor(normal, offset, rough, hurst, lacunarity, seedOffset, grainScale) {
    this.n = normalize(normal.slice());
    this.d = offset;              // signed distance of the plane from origin
    this.rough = rough;           // metres, RMS of the roughness field
    this.hurst = hurst;
    this.lac = lacunarity;
    this.so = seedOffset;
    this.grain = grainScale;      // base wavelength of asperities
    // Basis in the plane, so roughness is parameterised on the joint surface
    let t = Math.abs(this.n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    this.u = normalize(cross(t, this.n));
    this.v = cross(this.n, this.u);
  }
  /**
   * Signed distance to the rough plane (negative inside the block).
   *
   * The fBm is band-limited with RMS `rough`, so it can never displace the
   * surface by more than a few sigma. Beyond that distance the roughness cannot
   * change the sign or the CSG max(), so evaluating five octaves of gradient
   * noise there is pure waste -- and since this is called once per grid cell per
   * face (12 faces x 262k cells at 64^3), it dominated block construction.
   * Skipping it outside the shell it can actually affect is exact, not an
   * approximation: the returned value only differs where it provably cannot
   * matter to the surface.
   */
  dist(x, y, z) {
    const p = this.n[0] * x + this.n[1] * y + this.n[2] * z - this.d;
    if (this.rough <= 0) return p;
    const reach = this.rough * 4;
    if (p > reach || p < -reach) return p;
    const su = (x * this.u[0] + y * this.u[1] + z * this.u[2]) / this.grain;
    const sv = (x * this.v[0] + y * this.v[1] + z * this.v[2]) / this.grain;
    const r = fbmSelfAffine(su, sv, this.so, this.hurst, this.lac, 5);
    return p - r * this.rough;
  }
}

/**
 * Builds the fresh (unweathered) block SDF from joint sets.
 *
 * @returns {{sdf:(x,y,z)=>number, faces:JointFace[], meta:object}}
 */
export function buildJointBlock({
  seed = 1,
  style = 'orthogonal',
  size = 1.0,             // characteristic block size, metres
  aspect = [1, 0.78, 0.92],
  jointRoughness = 1.0,   // multiplier on fracture-surface RMS
  hurst = 0.8,            // self-affine exponent of fracture surfaces
  grainSize = 0.0035,
  sheetingCurvature = 0.0,
} = {}) {
  const rng = new RNG(seed * 2654435761 + 17);
  const sets = (JOINT_STYLES[style] || JOINT_STYLES.orthogonal).build(rng, size);
  const faces = [];
  let soCounter = 0;

  for (const set of sets) {
    // Two bounding faces per set (block sits between consecutive joints).
    for (const sign of [1, -1]) {
      const pole = rng.fisher(set.pole, set.kappa);
      const n = [sign * pole[0], sign * pole[1], sign * pole[2]];
      // Spacing is log-normally distributed in nature; CV ~0.3-0.5 is typical.
      const spac = Math.max(0.15 * size, set.spacing * Math.exp(set.spacingCV * rng.normal()));
      const axisScale = Math.abs(n[0]) * aspect[0] + Math.abs(n[1]) * aspect[1] + Math.abs(n[2]) * aspect[2];
      const d = 0.5 * spac * axisScale * rng.range(0.82, 1.18);
      // Asperity amplitude: joint roughness in granite scales with grain size
      // (crack path deviates around/through crystals) and with joint length.
      const rmsBase = 0.35 * Math.sqrt(grainSize * spac);
      faces.push(new JointFace(
        n, d,
        rmsBase * set.roughness * jointRoughness,
        hurst, 2.07, soCounter++ * 131 + seed * 7919,
        Math.max(grainSize * 6, spac * 0.45),
      ));
    }
  }

  // Sheeting joints are gently *curved* (convex-up, parallel to the free
  // surface). Model as a large-radius sphere subtraction on the top face.
  const sheetR = sheetingCurvature > 0 ? size * (6 / Math.max(0.05, sheetingCurvature)) : 0;

  const sdf = (x, y, z) => {
    let d = -1e9;
    for (let i = 0; i < faces.length; i++) {
      const fd = faces[i].dist(x, y, z);
      if (fd > d) d = fd;                        // CSG intersection = max
    }
    if (sheetR > 0) {
      const cy = sheetR + size * 0.42;
      const sd = sheetR - Math.hypot(x, y - cy, z);   // inside sphere -> keep
      d = Math.max(d, -sd);
    }
    return d;
  };

  return { sdf, faces, meta: { sets, style, size } };
}

/**
 * Smooth-minimum blend used when we want the fresh block to already carry a
 * little arris rounding (e.g. a block that was quarried a while ago).
 */
export function smoothIntersect(a, b, k) {
  const h = Math.max(0, Math.min(1, 0.5 - (0.5 * (b - a)) / k));
  return b * (1 - h) + a * h + k * h * (1 - h);
}
