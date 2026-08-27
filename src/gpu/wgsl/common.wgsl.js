/**
 * Shared WGSL: bindings, hashing, noise and the Laguerre crystal aggregate.
 *
 * This is a port of core/rng.js + core/petrology.js (and the GLSL mirror in
 * glsl/common.glsl.js). All three implement the SAME integer hash, so the
 * crystal that the compute shader erodes is the crystal the raymarcher shades.
 * tools/checkhash.mjs pins the JS and GLSL versions bit-for-bit; the WGSL here
 * uses identical constants and identical u32 wrapping semantics.
 */

export const PARAMS_WGSL = /* wgsl */`
struct Params {
  // grid ---------------------------------------------------------------
  n           : u32,
  seed        : u32,
  jfaStep     : u32,
  flags       : u32,

  h           : f32,
  origin      : f32,
  extent      : f32,
  dt          : f32,

  // rate law -----------------------------------------------------------
  spheroidal      : f32,
  spheroidalPower : f32,
  cavernous       : f32,
  cavernousPower  : f32,

  uniformRate    : f32,
  roundingRadius : f32,
  weakMin        : f32,
  weakMax        : f32,

  insolation       : f32,
  moistureGradient : f32,
  buriedFraction   : f32,
  grussification   : f32,

  rindlet        : f32,
  rindletSpacing : f32,
  heterogeneity  : f32,
  heteroScale    : f32,

  shelterRadius : f32,
  bandWidth     : f32,
  minVolFrac    : f32,
  _pad0         : f32,

  sunDir : vec4<f32>,

  // crystal aggregate ---------------------------------------------------
  cellSize   : f32,
  cellSize2  : f32,
  grainSigma : f32,
  seriate    : f32,

  foliation : f32,
  phenFrac  : f32,
  phenSize  : f32,
  phenId    : f32,

  // shading -------------------------------------------------------------
  weatherAge     : f32,
  lichen         : f32,
  caseHardening  : f32,
  dust           : f32,

  wetness       : f32,
  microRelief   : f32,
  retreatScale  : f32,
  stainStrength : f32,

  stainColor : vec4<f32>,

  exposure  : f32,
  debugMode : f32,
  _pad1     : f32,
  _pad2     : f32,

  minAlbedoRough : array<vec4<f32>, 6>,
  minProps       : array<vec4<f32>, 6>,   // spec, hardness, durability, cleavage
  minExtra       : array<vec4<f32>, 6>,   // fe, translucency, _, _
  modeCDF        : array<vec4<f32>, 6>,   // .x used
};
`;

export const HASH_WGSL = /* wgsl */`
fn hashU32(xin: u32) -> u32 {
  var x = xin;
  x = x ^ (x >> 16u); x = x * 0x7feb352du;
  x = x ^ (x >> 15u); x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}
fn hash3i(p: vec3<i32>, seed: u32) -> u32 {
  let h = u32(p.x) * 0x8da6b343u ^ u32(p.y) * 0xd8163841u
        ^ u32(p.z) * 0xcb1ab31fu ^ seed * 0x165667b1u;
  return hashU32(h);
}
fn u2f(h: u32) -> f32 { return f32(h) * (1.0 / 4294967296.0); }
`;

export const NOISE_WGSL = /* wgsl */`
fn grad3(p: vec3<i32>, seed: u32) -> vec3<f32> {
  let h  = hash3i(p, seed);
  let h2 = hashU32(h ^ 0x9e3779b9u);
  let z = u2f(h) * 2.0 - 1.0;
  let t = u2f(h2) * 6.2831853;
  let r = sqrt(max(0.0, 1.0 - z * z));
  return vec3<f32>(r * cos(t), r * sin(t), z);
}
fn fade1(t: f32) -> f32 { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

fn pnoise3(P: vec3<f32>, seed: u32) -> f32 {
  let ip = vec3<i32>(floor(P));
  let f  = P - floor(P);
  let u  = vec3<f32>(fade1(f.x), fade1(f.y), fade1(f.z));
  let n000 = dot(grad3(ip + vec3<i32>(0,0,0), seed), f - vec3<f32>(0.,0.,0.));
  let n100 = dot(grad3(ip + vec3<i32>(1,0,0), seed), f - vec3<f32>(1.,0.,0.));
  let n010 = dot(grad3(ip + vec3<i32>(0,1,0), seed), f - vec3<f32>(0.,1.,0.));
  let n110 = dot(grad3(ip + vec3<i32>(1,1,0), seed), f - vec3<f32>(1.,1.,0.));
  let n001 = dot(grad3(ip + vec3<i32>(0,0,1), seed), f - vec3<f32>(0.,0.,1.));
  let n101 = dot(grad3(ip + vec3<i32>(1,0,1), seed), f - vec3<f32>(1.,0.,1.));
  let n011 = dot(grad3(ip + vec3<i32>(0,1,1), seed), f - vec3<f32>(0.,1.,1.));
  let n111 = dot(grad3(ip + vec3<i32>(1,1,1), seed), f - vec3<f32>(1.,1.,1.));
  let x00 = mix(n000, n100, u.x); let x10 = mix(n010, n110, u.x);
  let x01 = mix(n001, n101, u.x); let x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z) * 1.1547;
}

// Self-affine fBm with Hurst exponent H (see core/noise.js for why H matters).
fn fbmH(p: vec3<f32>, seed: u32, H: f32, oct: i32) -> f32 {
  var sum = 0.0; var amp = 1.0; var norm = 0.0; var fq = 1.0;
  let lac = 2.03;
  for (var i = 0; i < 8; i = i + 1) {
    if (i >= oct) { break; }
    sum  = sum + amp * pnoise3(p * fq, seed + u32(i) * 7919u);
    norm = norm + amp * amp;
    amp  = amp * pow(lac, -H);
    fq   = fq * lac;
  }
  return sum / sqrt(max(norm, 1e-6));
}
`;

export const GRAIN_WGSL = /* wgsl */`
struct Grain {
  id       : i32,
  boundary : f32,
  jitter   : f32,
  size     : f32,
  axis     : vec3<f32>,
};

fn foliate(p: vec3<f32>) -> vec3<f32> {
  let f = P.foliation;
  if (f <= 0.0) { return p; }
  return vec3<f32>(p.x / (1.0 + 0.55 * f), p.y * (1.0 + 1.35 * f), p.z / (1.0 + 0.15 * f));
}

struct Lag { key: u32, d1: f32, d2: f32, site: vec3<f32> };

// Laguerre (radical Voronoi) cell lookup: the geometric idealisation of an
// interlocking igneous crystal mosaic.
fn laguerre(p: vec3<f32>, cs: f32, salt: u32) -> Lag {
  let g = vec3<i32>(floor(p / cs));
  var o: Lag;
  o.d1 = 1e30; o.d2 = 1e30; o.key = 0u; o.site = vec3<f32>(0.0);
  for (var k = -1; k <= 1; k = k + 1) {
  for (var j = -1; j <= 1; j = j + 1) {
  for (var i = -1; i <= 1; i = i + 1) {
    let c  = g + vec3<i32>(i, j, k);
    let h0 = hash3i(c, P.seed ^ salt);
    let h1 = hashU32(h0 ^ 0x9e3779b9u);
    let h2 = hashU32(h1 ^ 0x85ebca6bu);
    let h3 = hashU32(h2 ^ 0xc2b2ae35u);
    let q  = (vec3<f32>(c) + vec3<f32>(u2f(h0), u2f(h1), u2f(h2))) * cs;
    let w  = exp(P.grainSigma * (u2f(h3) * 2.0 - 1.0)) * cs * 0.5;
    let dd = length(q - p) - w;
    if (dd < o.d1) { o.d2 = o.d1; o.d1 = dd; o.key = h0; o.site = q; }
    else if (dd < o.d2) { o.d2 = dd; }
  }}}
  return o;
}

fn pickMineral(u: f32) -> i32 {
  for (var i = 0; i < 6; i = i + 1) {
    if (u <= P.modeCDF[i].x) { return i; }
  }
  return 5;
}

fn packGrain(id: i32, key: u32, boundary: f32, size: f32) -> Grain {
  var g: Grain;
  g.id = id; g.boundary = boundary; g.size = size;
  let h1 = u2f(hashU32(key ^ 0x165667b1u));
  let h2 = u2f(hashU32(key ^ 0x9e3779b1u));
  g.jitter = u2f(hashU32(key ^ 0x27d4eb2du));
  let th = acos(clamp(2.0 * h1 - 1.0, -1.0, 1.0));
  let ph = h2 * 6.2831853;
  g.axis = vec3<f32>(sin(th) * cos(ph), sin(th) * sin(ph), cos(th));
  return g;
}

fn sampleGrain(pObj: vec3<f32>) -> Grain {
  let p = foliate(pObj);

  if (P.phenFrac > 0.0) {
    let pc = laguerre(p, P.phenSize * 1.3, 0x51ed27u);
    if (u2f(hashU32(pc.key ^ 0x1b873593u)) < P.phenFrac) {
      let r = P.phenSize * 0.5 * (0.7 + 0.6 * u2f(hashU32(pc.key ^ 0x27d4eb2fu)));
      let dist = length(pc.site - p);
      if (dist < r) {
        return packGrain(i32(P.phenId), pc.key, clamp((r - dist) / (0.25 * r), 0.0, 1.0), P.phenSize);
      }
    }
  }

  var c = laguerre(p, P.cellSize, 0u);
  var size = P.cellSize;
  if (P.seriate > 0.0 && u2f(hashU32(c.key ^ 0x7ed55d16u)) < P.seriate * 0.45) {
    c = laguerre(p, P.cellSize2, 0x2545f4u);
    size = P.cellSize2;
  }
  let id = pickMineral(u2f(hashU32(c.key ^ 0xa5a5a5a5u)));
  let boundary = clamp((c.d2 - c.d1) / (0.35 * size), 0.0, 1.0);
  return packGrain(id, c.key, boundary, size);
}

// Point durability: mineral resistance weakened at grain boundaries, where
// hydrolysis fronts actually run.
fn durabilityAt(p: vec3<f32>) -> f32 {
  let g = sampleGrain(p);
  let boundaryWeak = 1.0 - 0.45 * (1.0 - g.boundary);
  return max(0.02, P.minProps[g.id].z * boundaryWeak);
}

// Durability UPSCALED to a grid cell (Reuss / harmonic mean over the cell).
// The grid has ~30 mm cells and crystals are ~3 mm, so point sampling would be
// white noise at grid scale and would grow fuzz instead of rounding the rock.
// The front advances through the weakest connected path, hence harmonic.
fn durabilityCell(p: vec3<f32>, cell: f32) -> f32 {
  let o = cell * 0.25;
  var inv = 0.0;
  for (var k = -1; k <= 1; k = k + 2) {
  for (var j = -1; j <= 1; j = j + 2) {
  for (var i = -1; i <= 1; i = i + 2) {
    inv = inv + 1.0 / durabilityAt(p + vec3<f32>(f32(i), f32(j), f32(k)) * o);
  }}}
  return 8.0 / inv;
}
`;

/** Index helpers shared by every pass. */
export const FIELD_WGSL = /* wgsl */`
fn idx3(i: u32, j: u32, k: u32) -> u32 { return (k * P.n + j) * P.n + i; }
fn coordOf(i: u32) -> f32 { return P.origin + f32(i) * P.h; }
fn posOf(c: vec3<u32>) -> vec3<f32> {
  return vec3<f32>(P.origin) + vec3<f32>(c) * P.h;
}
`;

/**
 * Emit a trilinear sampler bound to a specific storage buffer.
 *
 * Two constraints force this shape:
 *
 *   1. r32float storage buffers are not filterable in WebGPU, and we need the
 *      field readable from both compute and fragment stages, so the eight taps
 *      are done by hand rather than depending on the optional
 *      float32-filterable feature.
 *
 *   2. We do NOT pass the buffer as a ptr<storage, ...> parameter. Pointer
 *      parameters in the storage address space are the "unrestricted pointer
 *      parameters" WGSL language extension, which is not guaranteed across
 *      implementations -- and a shader that fails to compile is a blank canvas,
 *      not an error message. Generating one specialised function per buffer is
 *      plain core WGSL and costs nothing at runtime.
 *
 * @param {string} name  the storage variable to sample
 * @returns {string} WGSL declaring sample_<name>(wp: vec3<f32>) -> f32
 */
export function samplerFor(name) {
  return /* wgsl */`
fn sample_${name}(wp: vec3<f32>) -> f32 {
  let nf = f32(P.n);
  var f = (wp - vec3<f32>(P.origin)) / P.h;
  f = clamp(f, vec3<f32>(0.0), vec3<f32>(nf - 1.0001));
  let b = vec3<u32>(floor(f));
  let t = f - floor(f);
  let i = b.x; let j = b.y; let k = b.z;
  let c000 = ${name}[idx3(i,    j,    k   )];
  let c100 = ${name}[idx3(i+1u, j,    k   )];
  let c010 = ${name}[idx3(i,    j+1u, k   )];
  let c110 = ${name}[idx3(i+1u, j+1u, k   )];
  let c001 = ${name}[idx3(i,    j,    k+1u)];
  let c101 = ${name}[idx3(i+1u, j,    k+1u)];
  let c011 = ${name}[idx3(i,    j+1u, k+1u)];
  let c111 = ${name}[idx3(i+1u, j+1u, k+1u)];
  let x00 = mix(c000, c100, t.x); let x10 = mix(c010, c110, t.x);
  let x01 = mix(c001, c101, t.x); let x11 = mix(c011, c111, t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
`;
}
