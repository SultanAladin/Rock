/**
 * End-to-end boulder generation.
 *
 *   1. STRUCTURE  - carve a joint-bounded block out of the rock mass
 *   2. PETROLOGY  - build the crystal aggregate (durability field)
 *   3. WEATHERING - curvature-driven level-set erosion on the SDF
 *   4. MESHING    - dual contouring (sharp arrises AND rounded shoulders)
 *   5. DETAIL     - grain-scale differential micro-relief
 *   6. ATTRIBUTES - bake retreat / shelter / curvature per vertex for shading
 *
 * The staging matters: weathering must run on the *volume*, before meshing,
 * because rounding a corner is a topological/geometric change to the solid, not
 * a displacement of a surface. Anything that fakes it as a surface displacement
 * cannot produce a concave tafone or a flared base, and it always shows.
 */

import { Field3, redistance, meanCurvatureField } from './grid.js';
import { GrainField, LITHOLOGIES } from './petrology.js';
import { buildJointBlock, JOINT_STYLES } from './joints.js';
import { weather, microRelief, DEFAULT_WEATHERING } from './weathering.js';
import { dualContour, largestComponent, recomputeNormals, taubinSmooth } from './mesher.js';
import { RNG } from './rng.js';

export const DEFAULT_PARAMS = {
  seed: 1,
  lithology: 'biotite-granite',
  jointStyle: 'orthogonal',
  resolution: 64,
  size: 1.0,                 // characteristic diameter, metres
  aspectVariation: 0.28,
  jointRoughness: 1.0,
  hurst: 0.8,
  sheetingCurvature: 0.0,
  sharpness: 0.85,           // dual-contour QEF weight: 1 = razor arrises
  smoothing: 1,
  microReliefAmount: 1.0,
  weathering: { ...DEFAULT_WEATHERING },
};

/**
 * @param {object} params
 * @param {(f:number,label:string)=>void} [onProgress]
 */
export function generateRock(params = {}, onProgress = () => {}) {
  const P = { ...DEFAULT_PARAMS, ...params, weathering: { ...DEFAULT_WEATHERING, ...(params.weathering || {}) } };
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const rng = new RNG(P.seed);
  const litho = LITHOLOGIES[P.lithology] || LITHOLOGIES['biotite-granite'];

  // --- 1. structure ------------------------------------------------------
  onProgress(0.02, 'jointing');
  const av = P.aspectVariation;
  const aspect = [
    1.0,
    Math.exp(rng.normal() * av * 0.8) * 0.85,
    Math.exp(rng.normal() * av * 0.8),
  ];
  const block = buildJointBlock({
    seed: P.seed,
    style: P.jointStyle,
    size: P.size,
    aspect,
    jointRoughness: P.jointRoughness,
    hurst: P.hurst,
    grainSize: litho.grain,
    sheetingCurvature: P.sheetingCurvature,
  });

  // Domain: enough headroom that the block never touches the grid boundary.
  const extent = P.size * 0.95;
  const field = new Field3(P.resolution, extent);
  field.fill(block.sdf);
  // Only the band needs to be metric; capping the sweep keeps this off the
  // critical path (it is called once per rock, but at 96^3 it is not free).
  redistance(field, 2, field.h * 8);
  onProgress(0.12, 'redistancing');

  // --- 2. petrology ------------------------------------------------------
  const grains = new GrainField(litho, P.seed * 7919 + 13);

  // --- 3. weathering -----------------------------------------------------
  const wres = weather(field, grains, P.weathering, (f) => onProgress(0.12 + 0.6 * f, 'weathering'));
  onProgress(0.74, 'meshing');

  // --- 4. meshing --------------------------------------------------------
  let mesh = dualContour(field, { sharpness: P.sharpness });
  mesh = largestComponent(mesh);
  if (P.smoothing > 0) taubinSmooth(mesh, P.smoothing);
  recomputeNormals(mesh);
  onProgress(0.86, 'micro-relief');

  // --- 5. grain-scale micro-relief ---------------------------------------
  if (P.microReliefAmount > 0) {
    mesh.positions = microRelief(mesh.positions, mesh.normals, grains, P.microReliefAmount);
    recomputeNormals(mesh);
  }

  // --- 6. per-vertex attributes for the shader ---------------------------
  onProgress(0.93, 'baking attributes');
  const nv = mesh.positions.length / 3;
  const aRetreat = new Float32Array(nv);
  const aShelter = new Float32Array(nv);
  const aCurv = new Float32Array(nv);

  const retreatField = new Field3(field.n, field.extent);
  retreatField.data.set(wres.retreat);
  const shelterField = new Field3(field.n, field.extent);
  shelterField.data.set(wres.shelter);
  const curvArr = meanCurvatureField(field);
  const curvField = new Field3(field.n, field.extent);
  curvField.data.set(curvArr);

  for (let v = 0; v < nv; v++) {
    const x = mesh.positions[v * 3], y = mesh.positions[v * 3 + 1], z = mesh.positions[v * 3 + 2];
    aRetreat[v] = retreatField.sample(x, y, z);
    aShelter[v] = shelterField.sample(x, y, z);
    aCurv[v] = curvField.sample(x, y, z) * P.size;
  }

  // --- stats -------------------------------------------------------------
  let minY = Infinity, maxY = -Infinity, minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let v = 0; v < nv; v++) {
    const x = mesh.positions[v * 3], y = mesh.positions[v * 3 + 1], z = mesh.positions[v * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  // volume + surface area from the triangle soup (divergence theorem)
  let vol = 0, area = 0;
  const I = mesh.indices, Pp = mesh.positions;
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const ax = Pp[a], ay = Pp[a + 1], az = Pp[a + 2];
    const bx = Pp[b], by = Pp[b + 1], bz = Pp[b + 2];
    const cx = Pp[c], cy = Pp[c + 1], cz = Pp[c + 2];
    vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    area += 0.5 * Math.hypot(e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x);
  }
  vol = Math.abs(vol);
  // Sphericity (Wadell): ratio of the surface area of a sphere of equal volume
  // to the actual surface area. 1.0 = a perfect ball. Field-measured granite
  // corestones run ~0.75-0.9; fresh joint blocks ~0.6-0.7. This is the number
  // to check if you want to know whether the weathering run is physical.
  const sphericity = area > 0 ? (Math.PI ** (1 / 3) * (6 * vol) ** (2 / 3)) / area : 0;

  const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  onProgress(1.0, 'done');

  return {
    mesh, aRetreat, aShelter, aCurv,
    litho, lithoKey: P.lithology, params: P, grains,
    stats: {
      vertices: nv,
      triangles: I.length / 3,
      volume: vol,
      area,
      sphericity,
      bbox: [maxX - minX, maxY - minY, maxZ - minZ],
      massKg: vol * 2680,            // granite bulk density ~2.65-2.70 t/m^3
      elapsedMs: t1 - t0,
      weatheringSteps: wres.elapsed / (wres.budget || 1),
    },
  };
}

/**
 * Batch: derive per-instance variation from a master seed. Sizes follow a
 * log-normal distribution, which is what block-size distributions from joint
 * spacing actually look like in the field.
 */
export function makeBatchParams(base, count, spread = {}) {
  const rng = new RNG((base.seed ?? 1) * 2654435761);
  const S = {
    sizeSigma: 0.35,
    weatherSigma: 0.45,
    styleMix: null,          // array of joint styles to draw from
    lithoMix: null,
    ...spread,
  };
  const out = [];
  const styles = S.styleMix && S.styleMix.length ? S.styleMix : [base.jointStyle || 'orthogonal'];
  const lithos = S.lithoMix && S.lithoMix.length ? S.lithoMix : [base.lithology || 'biotite-granite'];
  for (let i = 0; i < count; i++) {
    const w = { ...(base.weathering || {}) };
    w.years = Math.max(0.05, (w.years ?? 1) * Math.exp(S.weatherSigma * rng.normal()));
    w.cavernous = Math.max(0, (w.cavernous ?? 0.35) * Math.exp(0.5 * rng.normal()));
    w.buriedFraction = Math.max(0, Math.min(0.6, (w.buriedFraction ?? 0.25) + 0.12 * rng.normal()));
    out.push({
      ...base,
      seed: (base.seed ?? 1) * 1000 + i * 7919 + 1,
      size: Math.max(0.08, (base.size ?? 1) * Math.exp(S.sizeSigma * rng.normal())),
      jointStyle: styles[rng.int(styles.length)],
      lithology: lithos[rng.int(lithos.length)],
      aspectVariation: Math.max(0.05, (base.aspectVariation ?? 0.28) * Math.exp(0.3 * rng.normal())),
      jointRoughness: Math.max(0.2, (base.jointRoughness ?? 1) * Math.exp(0.25 * rng.normal())),
      weathering: w,
    });
  }
  return out;
}

export { LITHOLOGIES, JOINT_STYLES };
