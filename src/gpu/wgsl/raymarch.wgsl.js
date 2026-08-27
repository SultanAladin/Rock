/**
 * SDF raymarcher + granite surface shading.
 *
 * This is what makes the erosion visible in realtime: there is no mesh in the
 * loop. Each frame we sphere-trace the live phi buffer, so whatever the compute
 * pass wrote this iteration is what you see this frame. Polygonisation happens
 * only when you press Export.
 *
 * The shading is a direct port of gpu/rockMaterial.js -- same crystal
 * aggregate, same cleavage-plane specular flash, same Fe staining / case
 * hardening / grussification / lichen model, same sub-pixel fade of the
 * micro-relief into roughness. See that file for the petrology reasoning.
 */

import { PARAMS_WGSL, HASH_WGSL, NOISE_WGSL, GRAIN_WGSL, FIELD_WGSL, samplerFor } from './common.wgsl.js';

export const RAYMARCH_WGSL = /* wgsl */`
${PARAMS_WGSL}

struct Camera {
  invViewProj : mat4x4<f32>,
  eye         : vec4<f32>,
  sun         : vec4<f32>,
  misc        : vec4<f32>,   // x = aspect, y = time, z = maxSteps, w = showBox
};

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> phiIn   : array<f32>;
@group(0) @binding(2) var<storage, read> aux     : array<f32>;   // shelter
@group(0) @binding(3) var<storage, read> retreat : array<f32>;
@group(0) @binding(4) var<uniform> Cam : Camera;

${HASH_WGSL}
${NOISE_WGSL}
${GRAIN_WGSL}
${FIELD_WGSL}
${samplerFor('phiIn')}
${samplerFor('aux')}
${samplerFor('retreat')}

const PI = 3.14159265359;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // fullscreen triangle
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.uv  = p[vi];
  return o;
}

fn sdfAt(p: vec3<f32>) -> f32 {
  // Outside the grid the field is meaningless, so fall back to the distance to
  // the domain box, which keeps sphere tracing conservative and convergent.
  let e = P.extent;
  let q = abs(p) - vec3<f32>(e);
  let outside = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
  if (outside > 0.0) { return outside + P.h; }
  return sample_phiIn(p);
}

fn sdfNormal(p: vec3<f32>) -> vec3<f32> {
  let e = P.h * 0.6;
  let dx = sdfAt(p + vec3<f32>(e,0,0)) - sdfAt(p - vec3<f32>(e,0,0));
  let dy = sdfAt(p + vec3<f32>(0,e,0)) - sdfAt(p - vec3<f32>(0,e,0));
  let dz = sdfAt(p + vec3<f32>(0,0,e)) - sdfAt(p - vec3<f32>(0,0,e));
  return normalize(vec3<f32>(dx, dy, dz));
}

// ray/box entry so we do not waste steps in empty space
fn boxEntry(ro: vec3<f32>, rd: vec3<f32>, e: f32) -> vec2<f32> {
  let inv = 1.0 / rd;
  let t0 = (vec3<f32>(-e) - ro) * inv;
  let t1 = (vec3<f32>( e) - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let a = max(max(tmin.x, tmin.y), tmin.z);
  let b = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(a, b);
}

// ---------------------------------------------------------------- GGX
fn D_GGX(NoH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = (NoH * a2 - NoH) * NoH + 1.0;
  return a2 / max(1e-7, PI * d * d);
}
fn V_Smith(NoV: f32, NoL: f32, a: f32) -> f32 {
  let a2 = a * a;
  let gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  let gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / max(1e-7, gv + gl);
}
fn D_GGXaniso(NoH: f32, ToH: f32, BoH: f32, ax: f32, ay: f32) -> f32 {
  let d = ToH * ToH / (ax * ax) + BoH * BoH / (ay * ay) + NoH * NoH;
  return 1.0 / max(1e-7, PI * ax * ay * d * d);
}
fn F_Schlick(f0: vec3<f32>, u: f32) -> vec3<f32> {
  let f = pow(1.0 - u, 5.0);
  return f0 + (vec3<f32>(1.0) - f0) * f;
}

// ------------------------------------------------- grain-scale micro-relief
// Continuous by construction: per-crystal relief tapers to zero at the grain
// boundary, so the field can be finite-differenced without producing a delta
// spike at every boundary (which is what turns a rock into per-pixel speckle).
fn microHeight(p: vec3<f32>) -> f32 {
  let g = sampleGrain(p);
  let hardness = P.minProps[g.id].y;
  let interior = smoothstep(0.0, 0.45, g.boundary);
  let proud = ((hardness - 5.6) / 4.5) * interior;
  let groove = -(1.0 - interior) * 0.45;
  let cleav = P.minProps[g.id].w;
  var intra: f32;
  if (cleav < 0.5) {
    intra = 0.35 * fbmH(p / max(g.size, 1e-4) * 3.1, 771u, 0.62, 3);
  } else if (cleav > 1.5) {
    intra = 0.22 * sin(dot(p, g.axis) * 6.2831853 / (g.size * 0.28));
  } else {
    intra = 0.30 * sin(dot(p, g.axis) * 6.2831853 / (g.size * 0.18));
  }
  return (proud + groove + intra * 0.6 * interior) * g.size;
}

struct Relief { n: vec3<f32>, subpixel: f32 };

fn perturbNormal(p: vec3<f32>, n: vec3<f32>, strength: f32, footprint: f32) -> Relief {
  var r: Relief;
  let g = sampleGrain(p);
  let grainPx = g.size / max(footprint, 1e-7);
  // Fade out once a crystal is smaller than a pixel: below that the correct
  // answer is the average of many crystals, not a random normal.
  r.subpixel = 1.0 - smoothstep(1.0, 3.0, 1.0 / max(grainPx, 1e-6));
  if (r.subpixel <= 0.001) { r.n = n; return r; }

  let e = max(footprint * 0.9, g.size * 0.06);
  let h0 = microHeight(p);
  var t1: vec3<f32>;
  if (abs(n.y) < 0.9) { t1 = normalize(cross(vec3<f32>(0.0,1.0,0.0), n)); }
  else                { t1 = normalize(cross(vec3<f32>(1.0,0.0,0.0), n)); }
  let t2 = cross(n, t1);
  let hx = microHeight(p + t1 * e);
  let hy = microHeight(p + t2 * e);
  var grad = ((hx - h0) * t1 + (hy - h0) * t2) / e;
  let gl = length(grad);
  if (gl > 2.0) { grad = grad * (2.0 / gl); }
  r.n = normalize(n - strength * r.subpixel * grad);
  return r;
}

// ------------------------------------------------------------- lichen
struct Lichen { cov: f32, edge: f32, species: f32 };
fn lichenField(p: vec3<f32>) -> Lichen {
  var o: Lichen;
  let cs = 0.055;
  let l = laguerre(p, cs, 0x9c1a3fu);
  o.species = u2f(hashU32(l.key ^ 0x3ab5u));
  let present = select(0.0, 1.0, u2f(hashU32(l.key ^ 0x77aa11u)) < P.lichen);
  let r = cs * (0.25 + 0.55 * u2f(hashU32(l.key ^ 0x1122u)));
  var dist = length(l.site - p);
  dist = dist + 0.35 * r * fbmH(p * 55.0, 313u, 0.75, 4);
  o.cov  = present * smoothstep(r, r * 0.55, dist);
  o.edge = present * smoothstep(r * 1.02, r * 0.80, dist)
         * (1.0 - smoothstep(r * 0.85, r * 0.55, dist));
  return o;
}

fn shade(p: vec3<f32>, nGeo: vec3<f32>, footprint: f32) -> vec3<f32> {
  let g = sampleGrain(p);
  let alRo = P.minAlbedoRough[g.id];
  let pr   = P.minProps[g.id];
  let ex   = P.minExtra[g.id];

  let shelter = clamp(sample_aux(p), 0.0, 1.0);
  let retreatRaw = sample_retreat(p);
  let ret = clamp(retreatRaw * P.retreatScale, 0.0, 1.0);

  // ---- debug channels -------------------------------------------------
  if (P.debugMode > 0.5) {
    var c: vec3<f32>;
    if (P.debugMode < 1.5) {
      var pal = array<vec3<f32>, 6>(
        vec3<f32>(0.85,0.85,0.90), vec3<f32>(0.95,0.35,0.30),
        vec3<f32>(0.35,0.75,0.95), vec3<f32>(0.15,0.12,0.10),
        vec3<f32>(0.95,0.85,0.45), vec3<f32>(0.20,0.55,0.30));
      c = pal[g.id] * (0.55 + 0.45 * g.boundary);
    } else if (P.debugMode < 2.5) {
      c = mix(vec3<f32>(0.06,0.10,0.25), vec3<f32>(1.0,0.85,0.25), ret);
    } else if (P.debugMode < 3.5) {
      c = mix(vec3<f32>(0.05), vec3<f32>(0.2,0.9,0.6), shelter);
    } else {
      let hh = P.h;
      let k = (sample_phiIn(p + nGeo*hh) - 2.0*sample_phiIn(p) + sample_phiIn(p - nGeo*hh)) / (hh*hh);
      c = select(mix(vec3<f32>(0.1), vec3<f32>(0.2,0.5,1.0), clamp(-k*0.05,0.0,1.0)),
                 mix(vec3<f32>(0.1), vec3<f32>(1.0,0.3,0.2), clamp( k*0.05,0.0,1.0)), k > 0.0);
    }
    let lam = 0.35 + 0.65 * clamp(dot(nGeo, normalize(Cam.sun.xyz)), 0.0, 1.0);
    return c * lam;
  }

  let rel = perturbNormal(p, nGeo, P.microRelief * 0.9, footprint);
  let N = rel.n;
  let lostDetail = (1.0 - rel.subpixel) * P.microRelief;

  var albedo = alRo.rgb * (1.0 + 0.16 * (g.jitter - 0.5));
  var rough  = clamp(alRo.a * (1.0 + 0.22 * (g.jitter - 0.5)), 0.045, 1.0);

  // sub-grain stochastic detail: sericite clouding, perthite lamellae
  if (g.id == 2) {
    let ser = 0.5 + 0.5 * fbmH(p * 900.0 + vec3<f32>(g.jitter * 17.0), 55u, 0.8, 3);
    albedo = mix(albedo, albedo * vec3<f32>(0.94,0.93,0.88), ser * 0.35 * P.weatherAge);
    rough = mix(rough, rough * 1.35, ser * 0.4);
  } else if (g.id == 1) {
    let lam2 = 0.5 + 0.5 * sin(dot(p, g.axis) * 2400.0 + g.jitter * 30.0);
    albedo = mix(albedo, albedo * vec3<f32>(1.05,0.98,0.95), lam2 * 0.22);
  }

  // ---- Fe(III) staining, with a mafic-source halo ----------------------
  let s = g.size * 1.4;
  var feHalo = P.minExtra[sampleGrain(p + vec3<f32>(s,0.0,0.0)).id].x;
  feHalo = feHalo + P.minExtra[sampleGrain(p - vec3<f32>(0.0,s,0.0)).id].x;
  feHalo = feHalo + P.minExtra[sampleGrain(p + vec3<f32>(0.0,0.0,s)).id].x;
  feHalo = feHalo / 3.0;
  let feSource = max(ex.x, feHalo * 0.85);
  let damp = mix(0.55, 1.0, shelter) * (0.45 + 0.55 * clamp(-nGeo.y * 0.5 + 0.6, 0.0, 1.0));
  var stainMask = clamp(P.stainStrength * P.weatherAge * (0.35 + 0.9 * feSource) * damp
                        * (0.4 + 0.9 * ret), 0.0, 1.0);
  stainMask = clamp(stainMask * (0.5 + 0.5 * fbmH(p * 130.0, 4211u, 0.85, 4) + 0.25), 0.0, 1.0);
  albedo = mix(albedo, albedo * 0.55 + P.stainColor.rgb * 0.62, stainMask);
  rough  = mix(rough, clamp(rough * 1.25 + 0.10, 0.0, 1.0), stainMask * 0.7);

  // ---- grussification --------------------------------------------------
  let grus = clamp(P.weatherAge * (1.0 - pr.z) * (0.3 + 0.9 * ret), 0.0, 1.0);
  albedo = mix(albedo, mix(albedo, vec3<f32>(0.52,0.49,0.45), 0.55), grus * 0.8);
  rough  = mix(rough, clamp(rough + 0.34, 0.0, 1.0), grus * 0.85);
  let bnd = 1.0 - g.boundary;
  albedo = albedo * (1.0 - 0.38 * bnd * (0.35 + 0.65 * P.weatherAge));
  rough  = clamp(rough + 0.22 * bnd * P.weatherAge, 0.0, 1.0);

  // ---- case hardening ---------------------------------------------------
  let caseH = clamp(P.caseHardening * P.weatherAge * (1.0 - shelter) * ret, 0.0, 1.0);
  albedo = mix(albedo, albedo * 0.72 + P.stainColor.rgb * 0.10, caseH * 0.7);
  rough  = mix(rough, clamp(rough * 0.55, 0.03, 1.0), caseH * 0.8);

  // ---- dust / soil contact ---------------------------------------------
  var dustMask = clamp(P.dust * clamp(-nGeo.y * 1.2 + 0.15, 0.0, 1.0) * (0.4 + 0.6 * shelter), 0.0, 1.0);
  dustMask = clamp(dustMask + clamp(P.dust * clamp(0.35 - p.y * 3.0, 0.0, 1.0), 0.0, 1.0) * 0.6, 0.0, 1.0);
  albedo = mix(albedo, vec3<f32>(0.30,0.26,0.21), dustMask * 0.55);
  rough  = mix(rough, 0.92, dustMask * 0.7);

  // ---- lichen ------------------------------------------------------------
  let li = lichenField(p);
  let lichGate = clamp((0.35 + 0.65 * shelter) * (0.45 + 0.75 * grus + 0.4 * ret)
                       * (0.5 + 0.5 * clamp(1.0 - nGeo.y, 0.0, 1.0)), 0.0, 1.0);
  let lich = li.cov * lichGate;
  var lichCol = mix(vec3<f32>(0.42,0.46,0.30), vec3<f32>(0.62,0.63,0.52), li.species);
  lichCol = lichCol * (0.75 + 0.5 * fbmH(p * 320.0, 88u, 0.7, 3));
  albedo = mix(albedo, lichCol, lich * 0.92);
  albedo = mix(albedo, lichCol * 1.25, li.edge * lichGate * 0.5);
  rough  = mix(rough, 0.96, lich * 0.9);

  // ---- wetness -----------------------------------------------------------
  let wet = clamp(P.wetness * (0.4 + 0.6 * shelter) * (0.5 + 0.5 * clamp(-nGeo.y + 0.8, 0.0, 1.0)), 0.0, 1.0);
  albedo = albedo * mix(1.0, 0.55, wet);
  rough  = mix(rough, rough * 0.25 + 0.02, wet);

  rough = clamp(rough + 0.22 * lostDetail, 0.0, 1.0);

  // ---- BRDF --------------------------------------------------------------
  let V = normalize(Cam.eye.xyz - p);
  let L = normalize(Cam.sun.xyz);
  let H = normalize(L + V);
  let NoV = clamp(dot(N, V), 0.0, 1.0) + 1e-5;
  let NoL = dot(N, L);
  let NoH = clamp(dot(N, H), 0.0, 1.0);
  let VoH = clamp(dot(V, H), 0.0, 1.0);

  let f0s = mix(pr.x, 0.62, wet);
  let f0 = vec3<f32>(0.04 * f0s / 0.5);

  // cleavage-plane specular flash
  let align = abs(dot(g.axis, N));
  var flash = 0.0;
  if (pr.w > 0.5) {
    let sharpAlign = pow(clamp((align - 0.86) / 0.14, 0.0, 1.0), 2.0);
    let w2 = select(1.0, 0.7, pr.w > 1.5);
    flash = sharpAlign * rel.subpixel * w2 * (1.0 - grus * 0.8) * (1.0 - lich);
  }
  let roughFlash = mix(rough, 0.035, flash);
  let a = max(0.02, roughFlash * roughFlash);

  var spec: f32;
  if (pr.w > 1.5 && flash > 0.01) {
    // mica: anisotropic, stretched along the sheet
    let T = normalize(cross(g.axis, N) + vec3<f32>(1e-6));
    let B = cross(N, T);
    spec = D_GGXaniso(NoH, dot(T, H), dot(B, H), max(0.02, roughFlash * 0.35), max(0.02, roughFlash * 1.6));
  } else {
    spec = D_GGX(NoH, a);
  }
  let Vis = V_Smith(NoV, max(NoL, 1e-4), a);
  let F = F_Schlick(f0, VoH);

  // soft shadow by marching toward the sun
  var sh = 1.0;
  {
    var t = P.h * 2.0;
    for (var i = 0; i < 24; i = i + 1) {
      let d = sdfAt(p + L * t);
      if (d < P.h * 0.25) { sh = 0.0; break; }
      sh = min(sh, 12.0 * d / t);
      t = t + max(d, P.h * 0.5);
      if (t > P.extent * 2.2) { break; }
    }
    sh = clamp(sh, 0.0, 1.0);
  }

  let sunCol = vec3<f32>(1.0, 0.96, 0.88) * 3.2;
  let direct = sunCol * clamp(NoL, 0.0, 1.0) * smoothstep(-0.15, 0.25, NoL) * sh;
  var diffuse = albedo / PI;
  // quartz translucency: light that entered the grain and came back out
  let wrap = ex.y * 0.5;
  diffuse = diffuse + albedo * ex.y * 0.22 * clamp((NoL + wrap) / (1.0 + wrap), 0.0, 1.0) / PI;

  var col = (diffuse * (vec3<f32>(1.0) - F) + F * spec * Vis) * direct;

  let ao = mix(1.0, 0.35, shelter * 0.9);
  let sky = vec3<f32>(0.28,0.38,0.55) * 0.85;
  let gnd = vec3<f32>(0.16,0.13,0.10) * 0.7;
  let hemi = 0.5 + 0.5 * N.y;
  col = col + albedo * mix(gnd, sky, hemi) * ao * (1.0 - 0.5 * F);
  let R = reflect(-V, N);
  col = col + mix(gnd, sky, 0.5 + 0.5 * R.y) * ao * F_Schlick(f0, NoV)
            * (1.0 - roughFlash) * (0.35 + 0.65 * flash);
  return col;
}

fn acesTonemap(x: vec3<f32>) -> vec3<f32> {
  let m1 = mat3x3<f32>(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777);
  let m2 = mat3x3<f32>(1.60475,-0.10208,-0.00327, -0.53108,1.10813,-0.07276, -0.07367,-0.00605,1.07602);
  let v = m1 * x;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(m2 * (a / b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let ndc = vec4<f32>(in.uv.x, in.uv.y, 1.0, 1.0);
  var wp = Cam.invViewProj * ndc;
  // NB: not "target" -- that is a WGSL reserved keyword.
  let farPoint = wp.xyz / wp.w;
  let ro = Cam.eye.xyz;
  let rd = normalize(farPoint - ro);

  let sky = mix(vec3<f32>(0.055,0.065,0.080), vec3<f32>(0.10,0.13,0.18),
                clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));
  var col = sky;

  let hit = boxEntry(ro, rd, P.extent);
  if (hit.y > max(hit.x, 0.0)) {
    var t = max(hit.x, 0.0) + 1e-4;
    let tEnd = hit.y;
    var found = false;
    var p = vec3<f32>(0.0);
    let maxSteps = i32(Cam.misc.z);

    for (var i = 0; i < 256; i = i + 1) {
      if (i >= maxSteps || t > tEnd) { break; }
      p = ro + rd * t;
      let d = sample_phiIn(p);
      if (d < P.h * 0.15) { found = true; break; }
      // Trilinear reconstruction slightly underestimates distance near the
      // surface; 0.85 keeps sphere tracing from stepping through thin features.
      t = t + max(d * 0.85, P.h * 0.12);
    }

    if (found) {
      // one bisection refinement for a cleaner silhouette
      var lo = t - P.h * 0.5;
      var hi = t;
      for (var i = 0; i < 4; i = i + 1) {
        let mid = 0.5 * (lo + hi);
        if (sample_phiIn(ro + rd * mid) < 0.0) { hi = mid; } else { lo = mid; }
      }
      t = hi;
      p = ro + rd * t;
      let n = sdfNormal(p);
      // pixel footprint in object space, for micro-relief filtering
      let footprint = max(1e-6, t * Cam.misc.x);
      col = shade(p, n, footprint);
      col = col * P.exposure;
      col = acesTonemap(col);
    } else {
      col = sky;
    }
  }

  return vec4<f32>(pow(col, vec3<f32>(1.0 / 2.2)), 1.0);
}
`;
