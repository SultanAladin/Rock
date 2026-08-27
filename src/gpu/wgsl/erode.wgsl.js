/**
 * GPU erosion passes.
 *
 * WHY THIS EXISTS
 * ---------------
 * The CPU solver was correct but structurally incapable of what a level-set
 * erosion tool should do: show you the front retreating, iteration by
 * iteration. Two things blocked it, and both are architectural rather than
 * "slow code":
 *
 *   1. phi lived in a worker's heap, so nothing could draw it mid-solve without
 *      copying the whole grid out every step.
 *   2. every displayed frame needed a full CPU re-polygonisation.
 *
 * Here phi lives in a GPU storage buffer and is RAYMARCHED directly, so there
 * is no meshing in the loop at all. Every iteration is visible for free because
 * the thing being drawn IS the field being solved. Dual contouring is demoted
 * to an export-time operation.
 *
 * THE REDISTANCING PROBLEM
 * ------------------------
 * The one part of the CPU solver that genuinely does not port is fast sweeping:
 * it is a Gauss-Seidel sweep whose whole efficiency comes from each cell
 * reading its neighbour's already-updated value, which is inherently
 * sequential. On a GPU that degenerates.
 *
 * The replacement is JUMP FLOODING (Rong & Tan). Instead of propagating
 * distances, we propagate the *closest interface point* itself: each cell holds
 * a seed coordinate, and in log2(N) passes with halving stride every cell finds
 * the nearest seed. Distance is then just |p - seed|. This is O(log N) fully
 * parallel passes instead of O(N) sequential sweeps, and it gives Euclidean
 * distance directly rather than the Eikonal approximation, so the field is
 * arguably *more* metric than the sweep it replaces.
 *
 * The physics is unchanged from core/weathering.js -- same saturating
 * curvature-driven rate law, same velocity extension, same crystal-derived
 * weakness field. See that file for the geomorphology; this file is the
 * parallel implementation of it.
 */

import { PARAMS_WGSL, HASH_WGSL, NOISE_WGSL, GRAIN_WGSL, FIELD_WGSL, samplerFor } from './common.wgsl.js';

const PRELUDE = /* wgsl */`
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read>       phiIn   : array<f32>;
@group(0) @binding(2) var<storage, read_write> phiOut  : array<f32>;
@group(0) @binding(3) var<storage, read>       seedIn  : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> seedOut : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       aux     : array<f32>;   // phi0 / shelter
@group(0) @binding(6) var<storage, read_write> auxOut  : array<f32>;
@group(0) @binding(7) var<storage, read_write> counter : array<atomic<u32>>;
// phi0 is bound separately from aux because the erosion step needs BOTH the
// shelter field (aux) and the fresh block field at the same time: rindlet
// shells are pinned to the ORIGINAL surface, so phasing them off the current
// phi would make the spalling bands migrate inward with the front instead of
// staying where the oxidation shells actually formed.
@group(0) @binding(8) var<storage, read> phi0In : array<f32>;

${HASH_WGSL}
${NOISE_WGSL}
${GRAIN_WGSL}
${FIELD_WGSL}
${samplerFor('phiIn')}
${samplerFor('aux')}
`;

/**
 * Pass 1 - INIT: evaluate the analytic joint-block SDF onto the grid.
 * Runs on the GPU so a parameter change re-seeds the whole domain in under a
 * millisecond, which is what makes the tool feel live rather than "baked".
 */
export const INIT_WGSL = /* wgsl */`
${PRELUDE}

// Joint faces are uploaded as planes + roughness parameters.
struct Face {
  n     : vec4<f32>,   // xyz normal, w = plane offset d
  u     : vec4<f32>,   // xyz in-plane basis, w = roughness RMS (metres)
  v     : vec4<f32>,   // xyz in-plane basis, w = asperity wavelength
  extra : vec4<f32>,   // x = hurst, y = seedOffset, z = lacunarity, w = unused
};
@group(1) @binding(0) var<storage, read> faces : array<Face>;
@group(1) @binding(1) var<uniform> faceCount : vec4<u32>;

// 2D self-affine fBm on the joint plane. Granite mode-I fracture surfaces are
// self-affine with H ~ 0.75-0.85 across five decades, so this exponent is a
// measured statistic, not a look.
fn fbm2(x: f32, y: f32, seed: u32, H: f32, oct: i32) -> f32 {
  var sum = 0.0; var amp = 1.0; var norm = 0.0; var fq = 1.0;
  let lac = 2.07;
  for (var i = 0; i < 6; i = i + 1) {
    if (i >= oct) { break; }
    sum  = sum + amp * pnoise3(vec3<f32>(x * fq, y * fq, 0.5), seed + u32(i) * 1013u);
    norm = norm + amp * amp;
    amp  = amp * pow(lac, -H);
    fq   = fq * lac;
  }
  return sum / sqrt(max(norm, 1e-6));
}

fn blockSDF(p: vec3<f32>) -> f32 {
  var d = -1e9;
  let cnt = faceCount.x;
  for (var i = 0u; i < cnt; i = i + 1u) {
    let f = faces[i];
    var pd = dot(f.n.xyz, p) - f.n.w;
    let rough = f.u.w;
    let reach = rough * 4.0;
    // Band-limited roughness cannot move the surface further than a few sigma,
    // so skipping the fBm outside that shell is exact, not an approximation.
    if (rough > 0.0 && pd < reach && pd > -reach) {
      let su = dot(p, f.u.xyz) / f.v.w;
      let sv = dot(p, f.v.xyz) / f.v.w;
      pd = pd - fbm2(su, sv, u32(f.extra.y), f.extra.x, 5) * rough;
    }
    d = max(d, pd);   // CSG intersection of half-spaces
  }
  return d;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n || gid.z >= P.n) { return; }
  let id = idx3(gid.x, gid.y, gid.z);
  let p  = posOf(gid);
  let d  = blockSDF(p);
  phiOut[id] = d;
  auxOut[id] = d;          // phi0, for rindlet phasing
}
`;

/**
 * Pass 2 - SEED: mark cells adjacent to the interface with a sub-cell accurate
 * closest-point, as input to the jump flood. Everything else gets an
 * "unassigned" marker (w < 0).
 */
export const JFA_SEED_WGSL = /* wgsl */`
${PRELUDE}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n || gid.z >= P.n) { return; }
  let id = idx3(gid.x, gid.y, gid.z);
  let c  = phiIn[id];
  let p  = posOf(gid);

  var best = 1e30;
  var bestP = vec3<f32>(0.0);
  var found = false;

  // Check the 6 axis neighbours for a sign change and linearly interpolate the
  // crossing. This gives sub-cell placement of the zero level set, which is
  // what keeps the reconstructed distance smooth.
  for (var a = 0; a < 3; a = a + 1) {
    for (var s = -1; s <= 1; s = s + 2) {
      var o = vec3<i32>(0, 0, 0);
      if (a == 0) { o.x = s; } else if (a == 1) { o.y = s; } else { o.z = s; }
      let q = vec3<i32>(gid) + o;
      if (q.x < 0 || q.y < 0 || q.z < 0 ||
          q.x >= i32(P.n) || q.y >= i32(P.n) || q.z >= i32(P.n)) { continue; }
      let nv = phiIn[idx3(u32(q.x), u32(q.y), u32(q.z))];
      if ((c <= 0.0) != (nv <= 0.0)) {
        let t  = abs(c) / max(1e-9, abs(c - nv));
        let cp = p + vec3<f32>(o) * (P.h * t);
        let dd = length(cp - p);
        if (dd < best) { best = dd; bestP = cp; found = true; }
      }
    }
  }

  if (found) { seedOut[id] = vec4<f32>(bestP, 1.0); }
  else       { seedOut[id] = vec4<f32>(0.0, 0.0, 0.0, -1.0); }
}
`;

/**
 * Pass 3 - JUMP FLOOD: propagate closest-points with halving stride.
 * Dispatched ceil(log2(n)) times; P.jfaStep carries the current stride.
 */
export const JFA_STEP_WGSL = /* wgsl */`
${PRELUDE}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n || gid.z >= P.n) { return; }
  let id = idx3(gid.x, gid.y, gid.z);
  let p  = posOf(gid);

  var best = seedIn[id];
  var bestD = select(1e30, length(best.xyz - p), best.w > 0.0);
  let st = i32(P.jfaStep);

  for (var k = -1; k <= 1; k = k + 1) {
  for (var j = -1; j <= 1; j = j + 1) {
  for (var i = -1; i <= 1; i = i + 1) {
    let q = vec3<i32>(gid) + vec3<i32>(i, j, k) * st;
    if (q.x < 0 || q.y < 0 || q.z < 0 ||
        q.x >= i32(P.n) || q.y >= i32(P.n) || q.z >= i32(P.n)) { continue; }
    let cand = seedIn[idx3(u32(q.x), u32(q.y), u32(q.z))];
    if (cand.w <= 0.0) { continue; }
    let dd = length(cand.xyz - p);
    if (dd < bestD) { bestD = dd; best = cand; }
  }}}

  seedOut[id] = best;
}
`;

/**
 * Pass 4 - RESOLVE: turn closest-points into a signed distance field, keeping
 * the original inside/outside sign.
 */
export const JFA_RESOLVE_WGSL = /* wgsl */`
${PRELUDE}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n || gid.z >= P.n) { return; }
  let id = idx3(gid.x, gid.y, gid.z);
  let p  = posOf(gid);
  let s  = seedIn[id];
  let old = phiIn[id];
  var d: f32;
  if (s.w > 0.0) { d = length(s.xyz - p); } else { d = abs(old); }
  phiOut[id] = select(d, -d, old <= 0.0);
}
`;

/**
 * Pass 5 - SHELTER: short-range ambient occlusion of the solid, the proxy for
 * "does water sit here and does the sun miss it". Gates the cavernous
 * (tafoni) term, without which that positive feedback dissolves the whole rock.
 */
export const SHELTER_WGSL = /* wgsl */`
${PRELUDE}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n || gid.z >= P.n) { return; }
  let id = idx3(gid.x, gid.y, gid.z);
  let d0 = phiIn[id];
  if (abs(d0) > P.bandWidth * P.h * 2.0) { auxOut[id] = 0.0; return; }

  let p = posOf(gid);
  let SAMPLES = 12;
  let STEPS = 4;
  let ga = 3.14159265 * (3.0 - sqrt(5.0));
  var occ = 0.0;

  for (var i = 0; i < SAMPLES; i = i + 1) {
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(SAMPLES);
    let r = sqrt(max(0.0, 1.0 - z * z));
    let t = ga * f32(i);
    let dir = vec3<f32>(r * cos(t), r * sin(t), z);
    for (var s = 1; s <= STEPS; s = s + 1) {
      let tt = (f32(s) / f32(STEPS)) * P.shelterRadius;
      if (sample_phiIn(p + dir * tt) < 0.0) { occ = occ + 1.0; break; }
    }
  }
  auxOut[id] = occ / f32(SAMPLES);
}
`;

/**
 * Pass 6 - EROSION STEP. One explicit level-set update:
 *     phi += dt * F * |grad phi|_Godunov
 *
 * F is the same saturating, curvature-driven speed as the CPU solver:
 *     F = [A_sph*sat(k,p) + A_cav*sat(-k,q)*shelter^2 + A_uni] * weakness
 *
 * with velocity extension: the speed is a property of the SURFACE, so for an
 * off-interface cell it is evaluated at the closest point ON the interface
 * (p - phi*n). Skipping that makes phi stop being a distance function and the
 * Godunov term amplifies exactly the high-frequency modes it should suppress --
 * the symptom is surface area RISING as the rock erodes.
 */
export const STEP_WGSL = /* wgsl */`
${PRELUDE}

// Mean curvature of the level set at an interior cell, by direct indexing.
fn curvatureAt(id: u32, n: u32) -> f32 {
  let nn = n * n;
  let h2 = 2.0 * P.h; let hh = P.h * P.h; let h4 = 4.0 * P.h * P.h;
  let c = phiIn[id];
  let px = (phiIn[id + 1u] - phiIn[id - 1u]) / h2;
  let py = (phiIn[id + n]  - phiIn[id - n])  / h2;
  let pz = (phiIn[id + nn] - phiIn[id - nn]) / h2;
  let m2 = px*px + py*py + pz*pz;
  if (m2 < 1e-10) { return 0.0; }
  let pxx = (phiIn[id + 1u] - 2.0 * c + phiIn[id - 1u]) / hh;
  let pyy = (phiIn[id + n]  - 2.0 * c + phiIn[id - n])  / hh;
  let pzz = (phiIn[id + nn] - 2.0 * c + phiIn[id - nn]) / hh;
  let pxy = (phiIn[id+1u+n]  - phiIn[id+1u-n]  - phiIn[id-1u+n]  + phiIn[id-1u-n])  / h4;
  let pxz = (phiIn[id+1u+nn] - phiIn[id+1u-nn] - phiIn[id-1u+nn] + phiIn[id-1u-nn]) / h4;
  let pyz = (phiIn[id+n+nn]  - phiIn[id+n-nn]  - phiIn[id-n+nn]  + phiIn[id-n-nn])  / h4;
  let num = (pyy + pzz) * px * px + (pxx + pzz) * py * py + (pxx + pyy) * pz * pz
          - 2.0 * (px * py * pxy + px * pz * pxz + py * pz * pyz);
  return num / (m2 * sqrt(m2));
}

// Static weakness: mineralogy (upscaled) x mesoscale heterogeneity x moisture
// x rindlet shells. Clamped, because the factors are independent worst cases
// that never physically co-occur and an unclamped product both exaggerates
// contrast beyond anything measured and destroys the timestep.
fn weaknessAt(p: vec3<f32>, phi0v: f32) -> f32 {
  let dur = durabilityCell(p, P.h);
  var m = 1.0 + P.grussification * (1.0 / max(0.10, dur) - 1.0);
  m = m / (1.0 + P.grussification * 1.4);

  if (P.heterogeneity > 0.0) {
    let s = 1.0 / max(0.02, P.heteroScale * P.extent);
    m = m * (1.0 + P.heterogeneity * 0.55 * fbmH(p * s, 1777u, 0.85, 3));
  }

  let yMin = -P.extent; let yMax = P.extent;
  let buriedY = yMin + (yMax - yMin) * P.buriedFraction;
  let t = clamp((p.y - buriedY) / (0.55 * (yMax - buriedY) + 1e-6), 0.0, 1.0);
  m = m * (1.0 + P.moistureGradient * (1.0 - t * t * (3.0 - 2.0 * t)));

  if (P.rindlet > 0.0) {
    let jit = 0.35 * P.rindletSpacing * fbmH(p * 5.0, 991u, 0.9, 3);
    m = m * (1.0 + P.rindlet * 0.5 * (1.0 + sin((phi0v + jit) * 6.2831853 / max(1e-4, P.rindletSpacing))));
  }
  return clamp(m, P.weakMin, P.weakMax);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P.n;
  if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
  let id = idx3(gid.x, gid.y, gid.z);

  // Boundary cells: the clamped stencil is not a valid difference operator and
  // letting the front reach the wall produces a flat-sided rock.
  if (gid.x < 1u || gid.y < 1u || gid.z < 1u ||
      gid.x >= n - 1u || gid.y >= n - 1u || gid.z >= n - 1u) {
    phiOut[id] = phiIn[id];
    return;
  }

  let d0 = phiIn[id];
  let lim = P.bandWidth * P.h;
  if (d0 < -lim || d0 > lim * 0.8) { phiOut[id] = d0; return; }

  let nn = n * n;
  let p = posOf(gid);
  let inv2h = 1.0 / (2.0 * P.h);
  let gx = (phiIn[id + 1u] - phiIn[id - 1u]) * inv2h;
  let gy = (phiIn[id + n]  - phiIn[id - n])  * inv2h;
  let gz = (phiIn[id + nn] - phiIn[id - nn]) * inv2h;
  let glen = max(1e-6, sqrt(gx*gx + gy*gy + gz*gz));
  let nrm = vec3<f32>(gx, gy, gz) / glen;

  // velocity extension: evaluate the speed at the closest interface point
  let ip = p - d0 * nrm;

  let cs = P.extent * P.roundingRadius;
  let kh = curvatureAt(id, n) * cs;
  let conv = max(0.0, kh);
  let conc = max(0.0, -kh);

  // Saturating rate law. A raw power law is unbounded: at grid resolution an
  // arris has kappa ~ 1/h, the rate blows up and the timestep collapses.
  // Saturation is also the physical asymptote -- a corner is attacked from at
  // most three joint faces, so its rate is bounded however sharp it is.
  let cp = pow(conv, P.spheroidalPower);
  let sph = P.spheroidal * (cp / (1.0 + cp));
  let cq = pow(conc, P.cavernousPower);
  let shl = sample_aux(ip);          // aux holds shelter during stepping
  let shc = clamp(shl, 0.0, 1.0);
  let cav = P.cavernous * (cq / (1.0 + cq)) * shc * shc;

  var f = sph + cav + P.uniformRate;

  if (P.insolation > 0.0) {
    let cosA = dot(nrm, normalize(P.sunDir.xyz));
    f = f * (1.0 + P.insolation * (0.5 * cosA + 0.25 * max(0.0, -cosA)));
  }

  f = f * weaknessAt(ip, phi0In[id]);

  // Godunov upwind |grad phi| for motion in the +normal (erosive) direction.
  let c = d0;
  let dxm = (c - phiIn[id - 1u]) / P.h; let dxp = (phiIn[id + 1u] - c) / P.h;
  let dym = (c - phiIn[id - n])  / P.h; let dyp = (phiIn[id + n]  - c) / P.h;
  let dzm = (c - phiIn[id - nn]) / P.h; let dzp = (phiIn[id + nn] - c) / P.h;
  let ax = max(max(dxm, 0.0) * max(dxm, 0.0), min(dxp, 0.0) * min(dxp, 0.0));
  let ay = max(max(dym, 0.0) * max(dym, 0.0), min(dyp, 0.0) * min(dyp, 0.0));
  let az = max(max(dzm, 0.0) * max(dzm, 0.0), min(dzp, 0.0) * min(dzp, 0.0));
  let gnorm = sqrt(ax + ay + az);

  phiOut[id] = d0 + P.dt * f * gnorm;
}
`;

/** Pass 7 - count interior cells, for the survival guard and live volume readout. */
export const COUNT_WGSL = /* wgsl */`
${PRELUDE}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n || gid.z >= P.n) { return; }
  if (phiIn[idx3(gid.x, gid.y, gid.z)] < 0.0) {
    atomicAdd(&counter[0], 1u);
  }
}
`;

/**
 * Pass 8 - RETREAT: how far the surface has moved back from the fresh block,
 * in metres. Bind aux = phi0 and auxOut = retreat.
 *
 * Because phi and phi0 are both (near-)signed-distance, their difference at a
 * point is the normal displacement of the front, which is exactly the rind
 * thickness the shader needs to decide how far the oxidation front has
 * penetrated. Accumulating it per step would drift with the redistancing;
 * differencing against the original field does not.
 */
export const RETREAT_WGSL = /* wgsl */`
${PRELUDE}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n || gid.z >= P.n) { return; }
  let id = idx3(gid.x, gid.y, gid.z);
  auxOut[id] = max(0.0, phiIn[id] - aux[id]);
}
`;
