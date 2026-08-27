/**
 * Granite surface shader.
 *
 * This is a physically-motivated, procedurally-shaded material -- there is no
 * bitmap texture anywhere in it. Everything is evaluated per-pixel from the
 * crystal aggregate, which means it is resolution-independent: you can put the
 * camera 5 cm from the surface and still be looking at individual crystals with
 * correct cleavage-plane speculars, not a stretched 4K photo.
 *
 * What the shader models, and why:
 *
 *  - PER-CRYSTAL ALBEDO. Each Laguerre cell gets its mineral's albedo with a
 *    per-crystal jitter. Real granite reads as salt-and-pepper mottling at
 *    grain scale, and that mottle is *discontinuous* across grain boundaries.
 *    Noise-based rock shaders always smear across boundaries; this does not.
 *
 *  - CLEAVAGE SPECULAR LOBES. Feldspar has two good cleavages and mica one
 *    perfect one. When such a plane happens to lie parallel to the surface it
 *    throws a mirror flash ("schiller"). We detect alignment between the
 *    crystal's cleavage normal and the shading normal and sharpen roughness
 *    there. This glint field, which winks in and out as you orbit, is the
 *    single strongest cue that you are looking at crystalline rock.
 *
 *  - ANISOTROPIC BIOTITE. Mica books are layered; their specular is stretched
 *    along the sheet. Handled with an anisotropic GGX using the cleavage frame.
 *
 *  - MICRO-RELIEF NORMALS. Analytic derivative of the grain-boundary /
 *    hardness field, so quartz reads as standing proud and biotite as pitted,
 *    at any zoom, matching the geometric micro-relief applied to the mesh.
 *
 *  - WEATHERING RIND. `aRetreat` (from the solver) says how much rock was lost
 *    at this point; `aShelter` says how sheltered it is. Together they drive:
 *      * Fe(III) staining (goethite/limonite ochre) that intensifies on
 *        long-exposed, damp, sheltered surfaces -- biotite and hornblende are
 *        the Fe sources, so the stain is strongest where mafics were dense;
 *      * case hardening -- a silica/iron-cemented skin that is *smoother and
 *        darker* than fresh rock -- on exposed convex faces;
 *      * grussification -- opened grain boundaries, elevated roughness and a
 *        bleached, sugary look where feldspar has gone to clay.
 *
 *  - LICHEN. Crustose lichen is not decoration; on any granite older than a
 *    few decades in a temperate climate it covers a large fraction of the
 *    surface and is the dominant colour signal. It is placed by a thallus
 *    cellular field, gated by shelter, aspect (prefers non-baking faces),
 *    moisture and time, with a rough, matte, slightly raised response.
 *
 *  - SUBSURFACE SCATTERING IN QUARTZ. Quartz grains are translucent; light
 *    enters and scatters a fraction of a millimetre. Approximated with a
 *    wrapped-diffuse term weighted by mineral translucency, which is what
 *    keeps quartz from reading as grey plastic.
 *
 *  - DUST / SOIL CONTACT. Downward-facing and basal surfaces pick up fine
 *    sediment; this desaturates and roughens them.
 */

import * as THREE from 'three';
import { HASH_GLSL, NOISE_GLSL, GRAIN_GLSL } from './glsl/common.glsl.js';
import { MINERAL_LIST, buildModeCDF } from '../core/petrology.js';

const VERT = /* glsl */`
precision highp float;
precision highp int;

in float aRetreat;
in float aShelter;
in float aCurvature;

out vec3 vObj;
out vec3 vWorldPos;
out vec3 vNormalW;
out vec3 vNormalObj;
out float vRetreat;
out float vShelter;
out float vCurv;
// The object->world rotation, forwarded to the fragment stage. three.js only
// declares modelMatrix in its VERTEX prefix, not the fragment one, so a
// fragment-stage reference to it fails to compile. We need it there to rotate
// per-crystal cleavage normals into world space for the specular flash.
out mat3 vObjToWorld;

void main(){
  vObj = position;
  vNormalObj = normalize(normal);
  vRetreat = aRetreat;
  vShelter = aShelter;
  vCurv = aCurvature;
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vObjToWorld = mat3(modelMatrix);
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const FRAG = /* glsl */`
precision highp float;
precision highp int;

in vec3 vObj;
in vec3 vWorldPos;
in vec3 vNormalW;
in vec3 vNormalObj;
in float vRetreat;
in float vShelter;
in float vCurv;
in mat3 vObjToWorld;

out vec4 fragColor;

uniform vec3  uCameraPos;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform vec3  uSkyColor;
uniform vec3  uGroundColor;
uniform float uExposure;

uniform vec3  uStainColor;
uniform float uStainStrength;
uniform float uWeatherAge;      // 0 fresh .. 1 deeply weathered
uniform float uLichen;
uniform vec3  uLichenColorA;
uniform vec3  uLichenColorB;
uniform float uCaseHardening;
uniform float uDust;
uniform float uWetness;
uniform float uMicroRelief;
uniform float uRetreatScale;
uniform float uDebugMode;       // 0 shaded, 1 mineral, 2 retreat, 3 shelter, 4 curvature

${HASH_GLSL}
${NOISE_GLSL}
${GRAIN_GLSL}

const float PI = 3.14159265359;

float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3  saturate3(vec3 x){ return clamp(x, 0.0, 1.0); }

// ---------------------------------------------------------------- GGX
float D_GGX(float NoH, float a){
  float a2 = a*a;
  float d = (NoH*a2 - NoH)*NoH + 1.0;
  return a2 / max(1e-7, PI * d * d);
}
float V_SmithGGX(float NoV, float NoL, float a){
  float a2 = a*a;
  float gv = NoL * sqrt(NoV*NoV*(1.0-a2)+a2);
  float gl = NoV * sqrt(NoL*NoL*(1.0-a2)+a2);
  return 0.5 / max(1e-7, gv+gl);
}
float D_GGXaniso(float NoH, float ToH, float BoH, float ax, float ay){
  float d = ToH*ToH/(ax*ax) + BoH*BoH/(ay*ay) + NoH*NoH;
  return 1.0 / max(1e-7, PI*ax*ay*d*d);
}
vec3 F_Schlick(vec3 f0, float u){
  float f = pow(1.0 - u, 5.0);
  return f0 + (1.0 - f0) * f;
}

// -------------------------------------------- grain-scale height for normals
// Scalar micro-height: hard minerals proud, grain boundaries grooved, mica
// pits. Differentiated analytically below to perturb the normal.
// Micro-relief height field.
//
// CONTINUITY IS MANDATORY HERE. This field gets finite-differenced to perturb
// the shading normal, so any step discontinuity differentiates to a spike of
// height (delta_h / epsilon) -- with 4 mm grains and epsilon 0.35 mm a single
// grain-boundary crossing produced gradients of 10-20 against a perturbation
// strength of ~1, randomising the normal on every boundary pixel. That is what
// made the surface read as static/speckle rather than stone.
//
// Two things made it discontinuous and both are fixed:
//  1. proud is constant per crystal, so it JUMPED at every grain boundary.
//     Now it is faded out by the boundary proximity itself, so neighbouring
//     crystals both approach zero relief at their shared boundary and the
//     field is continuous across it. Physically this is also more correct: the
//     etched groove IS the boundary, and a crystal's proud face tapers into it
//     rather than meeting its neighbour at a cliff.
//  2. fract() sawtooths are C0-discontinuous once per period. Replaced with
//     sin(), which carries the same cleavage-step periodicity smoothly.
float microHeight(vec3 p){
  Grain g = sampleGrain(p);
  float hardness = uMinProps[g.id].y;
  // taper to zero at the grain boundary -> continuous across crystals
  float interior = smoothstep(0.0, 0.45, g.boundary);
  float proud = ((hardness - 5.6) / 4.5) * interior;
  // the groove itself: deepest exactly at the boundary
  float groove = -(1.0 - interior) * 0.45;
  // intragranular relief: conchoidal chipping on quartz, stepped cleavage on
  // feldspar, flaky steps on mica. Also faded at boundaries.
  float cleav = uMinProps[g.id].w;
  float intra;
  if(cleav < 0.5)      intra = 0.35 * fbmH(p / max(g.size,1e-4) * 3.1, 771u, 0.62, 3);
  else if(cleav > 1.5) intra = 0.22 * sin(dot(p, g.axis) * 6.2831853 / (g.size*0.28));
  else                 intra = 0.30 * sin(dot(p, g.axis) * 6.2831853 / (g.size*0.18));
  return (proud + groove + intra * 0.6 * interior) * g.size;
}

// Perturb the shading normal by the micro-relief gradient.
//
// The differencing step is tied to the PIXEL FOOTPRINT, not a fixed constant.
// A constant epsilon is only valid while a grain covers many pixels; as soon as
// the camera pulls back and a 4 mm crystal is smaller than a pixel, a fixed
// epsilon samples uncorrelated crystals and the normal becomes per-pixel white
// noise -- classic normal-map aliasing, and it does not go away with MSAA
// because the noise is in the shading, not the geometry.
//
// fwidth(p) gives the object-space size of one pixel, so e tracks it. We also
// fade the whole perturbation out once the footprint exceeds the grain size:
// below that scale the correct answer is not "random normals" but "the average
// of many crystals", i.e. the unperturbed normal with slightly raised
// roughness, which is what a real rough surface does at distance.
vec3 perturbNormal(vec3 p, vec3 n, float strength, out float subpixel){
  float fw = max(1e-6, length(fwidth(p)));
  Grain g = sampleGrain(p);
  // how many pixels across is one crystal
  float grainPx = g.size / fw;
  subpixel = 1.0 - smoothstep(1.0, 3.0, 1.0 / max(grainPx, 1e-6));
  if(subpixel <= 0.001) return n;

  float e = max(fw * 0.9, g.size * 0.06);
  float h0 = microHeight(p);
  vec3 t1 = normalize(abs(n.y) < 0.9 ? cross(vec3(0,1,0), n) : cross(vec3(1,0,0), n));
  vec3 t2 = cross(n, t1);
  float hx = microHeight(p + t1*e);
  float hy = microHeight(p + t2*e);
  vec3 grad = ((hx - h0) * t1 + (hy - h0) * t2) / e;
  // Clamp the slope. Even a continuous field can be locally steep, and a normal
  // tilted past grazing produces black speckle under the BRDF.
  float gl = length(grad);
  if(gl > 2.0) grad *= 2.0 / gl;
  return normalize(n - strength * subpixel * grad);
}

// ------------------------------------------------------------- lichen field
// Crustose thalli: roughly circular colonies that grow radially and merge.
float lichenField(vec3 p, out float edge, out float species){
  uint key; float d1, d2; vec3 site;
  float cs = 0.055;
  laguerre(p, cs, 0x9c1a3fu, key, d1, d2, site);
  species = u2f(hashU32(key ^ 0x3ab5u));
  float present = step(u2f(hashU32(key ^ 0x77aa11u)), uLichen);
  float r = cs * (0.25 + 0.55 * u2f(hashU32(key ^ 0x1122u)));
  float dist = length(site - p);
  // ragged margin
  dist += 0.35 * r * fbmH(p * 55.0, 313u, 0.75, 4);
  float cov = present * smoothstep(r, r * 0.55, dist);
  edge = present * smoothstep(r*1.02, r*0.80, dist) * (1.0 - smoothstep(r*0.85, r*0.55, dist));
  return cov;
}

void main(){
  vec3 N = normalize(vNormalW);
  vec3 Nobj = normalize(vNormalObj);
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 L = normalize(uSunDir);

  Grain g = sampleGrain(vObj);
  vec4 alRo = uMinAlbedoRough[g.id];
  vec4 pr   = uMinProps[g.id];
  vec2 ex   = uMinExtra[g.id];

  // ---------------- debug channels -------------------------------------
  if(uDebugMode > 0.5){
    vec3 c;
    if(uDebugMode < 1.5){
      vec3 pal[6];
      pal[0]=vec3(0.85,0.85,0.90); pal[1]=vec3(0.95,0.35,0.30);
      pal[2]=vec3(0.35,0.75,0.95); pal[3]=vec3(0.15,0.12,0.10);
      pal[4]=vec3(0.95,0.85,0.45); pal[5]=vec3(0.20,0.55,0.30);
      c = pal[g.id] * (0.55 + 0.45*g.boundary);
    } else if(uDebugMode < 2.5){
      float t = saturate(vRetreat * uRetreatScale);
      c = mix(vec3(0.06,0.10,0.25), vec3(1.0,0.85,0.25), t);
    } else if(uDebugMode < 3.5){
      c = mix(vec3(0.05), vec3(0.2,0.9,0.6), saturate(vShelter));
    } else {
      float k = vCurv;
      c = k > 0.0 ? mix(vec3(0.1), vec3(1.0,0.3,0.2), saturate(k*3.0))
                  : mix(vec3(0.1), vec3(0.2,0.5,1.0), saturate(-k*3.0));
    }
    float lam = 0.35 + 0.65 * saturate(dot(N, L));
    fragColor = vec4(pow(c * lam, vec3(1.0/2.2)), 1.0);
    return;
  }

  // ---------------- micro-relief normal ---------------------------------
  float subpixel;
  N = perturbNormal(vObj, N, uMicroRelief * 0.9, subpixel);
  // Detail lost to the sub-pixel fade is not discarded, it is folded into
  // roughness (LEAN/Toksvig-style): geometric micro-relief that can no longer
  // be resolved as normal variation still scatters light, so a distant rock
  // must get rougher rather than smoother. Without this the fade would make
  // far boulders look polished.
  float lostDetail = (1.0 - subpixel) * uMicroRelief;

  // ---------------- base mineral colour ---------------------------------
  float cj = (g.jitter - 0.5);
  vec3 albedo = alRo.rgb * (1.0 + 0.16 * cj);
  float rough = clamp(alRo.a * (1.0 + 0.22 * cj), 0.045, 1.0);

  // Sericite clouding on plagioclase, perthite lamellae on K-feldspar: these
  // really are stochastic sub-grain features, so noise is the right tool here
  // (and only here).
  if(g.id == 2){
    float ser = 0.5 + 0.5 * fbmH(vObj * 900.0 + float(g.jitter)*17.0, 55u, 0.8, 3);
    albedo = mix(albedo, albedo * vec3(0.94,0.93,0.88), ser * 0.35 * uWeatherAge);
    rough = mix(rough, rough*1.35, ser*0.4);
  } else if(g.id == 1){
    float lam = 0.5 + 0.5 * sin(dot(vObj, g.axis) * 2400.0 + g.jitter * 30.0);
    albedo = mix(albedo, albedo * vec3(1.05,0.98,0.95), lam * 0.22);
  }

  // ---------------- weathering rind -------------------------------------
  float retreat = saturate(vRetreat * uRetreatScale);
  float shelter = saturate(vShelter);
  // Fe(III) staining: needs an Fe source nearby, moisture, and time. Mafic-rich
  // patches bleed ochre into the surrounding felsic grains, so we sample the Fe
  // content of the neighbourhood rather than just this grain.
  float feLocal = ex.x;
  float feHalo = 0.0;
  {
    float s = g.size * 1.4;
    feHalo += uMinExtra[sampleGrain(vObj + vec3(s,0,0)).id].x;
    feHalo += uMinExtra[sampleGrain(vObj - vec3(0,s,0)).id].x;
    feHalo += uMinExtra[sampleGrain(vObj + vec3(0,0,s)).id].x;
    feHalo /= 3.0;
  }
  float feSource = max(feLocal, feHalo * 0.85);
  float damp = mix(0.55, 1.0, shelter) * (0.45 + 0.55 * saturate(-Nobj.y * 0.5 + 0.6));
  float stainMask = saturate(uStainStrength * uWeatherAge *
                             (0.35 + 0.9 * feSource) * damp *
                             (0.4 + 0.9 * retreat));
  // stain is a thin ferruginous film: multiplies rather than replaces
  stainMask *= 0.5 + 0.5 * fbmH(vObj * 130.0, 4211u, 0.85, 4) + 0.25;
  stainMask = saturate(stainMask);
  albedo = mix(albedo, albedo * 0.55 + uStainColor * 0.62, stainMask);
  rough = mix(rough, clamp(rough * 1.25 + 0.10, 0.0, 1.0), stainMask * 0.7);

  // Grussification: opened boundaries, clay-altered feldspar. Bleaches and
  // roughens; strongest where retreat is high and durability was low.
  float grus = saturate(uWeatherAge * (1.0 - pr.z) * (0.3 + 0.9 * retreat));
  albedo = mix(albedo, mix(albedo, vec3(0.52,0.49,0.45), 0.55), grus * 0.8);
  rough = mix(rough, clamp(rough + 0.34, 0.0, 1.0), grus * 0.85);
  // and the boundary grooves themselves darken with trapped dust
  float bnd = 1.0 - g.boundary;
  albedo *= 1.0 - 0.38 * bnd * (0.35 + 0.65 * uWeatherAge);
  rough = clamp(rough + 0.22 * bnd * uWeatherAge, 0.0, 1.0);

  // Case hardening: exposed convex faces develop a silica/Fe-cemented skin -
  // darker, smoother, and it survives while the surrounding rock spalls.
  float exposedConvex = saturate(vCurv * 2.0) * (1.0 - shelter);
  float caseH = saturate(uCaseHardening * uWeatherAge * exposedConvex);
  albedo = mix(albedo, albedo * 0.72 + uStainColor * 0.10, caseH * 0.7);
  rough = mix(rough, clamp(rough * 0.55, 0.03, 1.0), caseH * 0.8);

  // Dust / soil contact on downward and basal surfaces.
  float dustMask = saturate(uDust * saturate(-Nobj.y * 1.2 + 0.15) * (0.4 + 0.6*shelter));
  dustMask += saturate(uDust * saturate(0.35 - vObj.y * 3.0)) * 0.6;
  dustMask = saturate(dustMask);
  albedo = mix(albedo, vec3(0.30,0.26,0.21), dustMask * 0.55);
  rough = mix(rough, 0.92, dustMask * 0.7);

  // ---------------- lichen ----------------------------------------------
  float lichEdge, lichSp;
  float lich = lichenField(vObj, lichEdge, lichSp);
  // lichen prefers surfaces that stay damp and are not baking: gate on shelter,
  // aspect and roughness of the substrate (it grips grussified rock better)
  float lichGate = saturate((0.35 + 0.65*shelter) * (0.45 + 0.75*grus + 0.4*retreat)
                            * (0.5 + 0.5*saturate(1.0 - Nobj.y)));
  lich *= lichGate;
  vec3 lichCol = mix(uLichenColorA, uLichenColorB, lichSp);
  lichCol *= 0.75 + 0.5 * fbmH(vObj * 320.0, 88u, 0.7, 3);
  albedo = mix(albedo, lichCol, lich * 0.92);
  // apothecia / raised margin
  albedo = mix(albedo, lichCol * 1.25, lichEdge * 0.5);
  rough = mix(rough, 0.96, lich * 0.9);

  // ---------------- wetness ---------------------------------------------
  // Water fills micro-porosity: darkens albedo, drops roughness, raises F0.
  float wet = saturate(uWetness * (0.4 + 0.6*shelter) * (0.5 + 0.5*saturate(-Nobj.y+0.8)));
  albedo *= mix(1.0, 0.55, wet);
  rough = mix(rough, rough * 0.25 + 0.02, wet);

  // ---------------- BRDF -------------------------------------------------
  float NoV = saturate(dot(N, V)) + 1e-5;
  float NoL = dot(N, L);
  vec3 H = normalize(L + V);
  float NoH = saturate(dot(N, H));
  float VoH = saturate(dot(V, H));

  float f0s = mix(pr.x, 0.62, wet);
  vec3 f0 = vec3(0.04 * f0s / 0.5);

  // Cleavage-plane specular: when the crystal's cleavage normal lines up with
  // the shading normal, the facet is a mirror.
  vec3 axisW = normalize(vObjToWorld * g.axis);
  float align = abs(dot(axisW, N));
  float cleavage = pr.w;
  float flash = 0.0;
  if(cleavage > 0.5){
    float sharpAlign = pow(saturate((align - 0.86) / 0.14), 2.0);
    // Gate on subpixel for the same reason as the normal perturbation: a
    // near-mirror lobe on crystals smaller than a pixel cannot be integrated by
    // point sampling and turns into firefly speckle. Once grains go sub-pixel
    // the flash is folded into general roughness instead (above).
    flash = sharpAlign * subpixel * (cleavage > 1.5 ? 0.7 : 1.0) * (1.0 - grus*0.8) * (1.0 - lich);
  }
  // fold unresolvable micro-relief into roughness before the specular lobes
  rough = clamp(rough + 0.22 * lostDetail, 0.0, 1.0);
  float roughFlash = mix(rough, 0.035, flash);

  float specular = 0.0;
  if(cleavage > 1.5 && flash > 0.01){
    // mica: anisotropic, stretched along the sheet
    vec3 T = normalize(cross(axisW, N) + 1e-6);
    vec3 B = cross(N, T);
    float ax = max(0.02, roughFlash * 0.35);
    float ay = max(0.02, roughFlash * 1.6);
    specular = D_GGXaniso(NoH, dot(T,H), dot(B,H), ax, ay);
  } else {
    specular = D_GGX(NoH, max(0.02, roughFlash*roughFlash));
  }
  float Vis = V_SmithGGX(NoV, max(NoL, 1e-4), max(0.02, roughFlash*roughFlash));
  vec3 F = F_Schlick(f0, VoH);

  // Direct sun
  float shadowish = smoothstep(-0.15, 0.25, NoL);
  vec3 direct = uSunColor * saturate(NoL) * shadowish;
  vec3 diffuse = albedo / PI;
  // Quartz translucency: wrapped diffuse, the light that entered the grain and
  // came back out a fraction of a millimetre away.
  float wrap = ex.y * 0.5;
  float ndlWrap = saturate((dot(N, L) + wrap) / (1.0 + wrap));
  diffuse += albedo * ex.y * 0.22 * ndlWrap / PI;

  vec3 col = (diffuse * (1.0 - F) + F * specular * Vis) * direct;

  // Hemispheric ambient with an AO term from the solver's shelter field.
  float ao = mix(1.0, 0.35, shelter * 0.9);
  float hemi = 0.5 + 0.5 * N.y;
  vec3 ambient = mix(uGroundColor, uSkyColor, hemi) * ao;
  col += albedo * ambient * (1.0 - 0.5*F);
  // ambient specular approximation
  vec3 R = reflect(-V, N);
  float rh = 0.5 + 0.5 * R.y;
  vec3 ambSpec = mix(uGroundColor, uSkyColor, rh) * ao;
  col += ambSpec * F_Schlick(f0, NoV) * (1.0 - roughFlash) * (0.35 + 0.65*flash);

  col *= uExposure;
  // Filmic-ish tonemap (ACES fit) then sRGB
  const mat3 ACESin = mat3(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777);
  const mat3 ACESout = mat3(1.60475,-0.10208,-0.00327, -0.53108,1.10813,-0.07276, -0.07367,-0.00605,1.07602);
  vec3 v = ACESin * col;
  vec3 a2 = v * (v + 0.0245786) - 0.000090537;
  vec3 b2 = v * (0.983729*v + 0.4329510) + 0.238081;
  col = saturate3(ACESout * (a2/b2));
  fragColor = vec4(pow(col, vec3(1.0/2.2)), 1.0);
}
`;

/**
 * Build a THREE.ShaderMaterial configured for a given lithology.
 */
export function createRockMaterial(litho, seed, opts = {}) {
  const albedoRough = [];
  const props = [];
  const extra = [];
  for (const m of MINERAL_LIST) {
    albedoRough.push(new THREE.Vector4(m.albedo[0], m.albedo[1], m.albedo[2], m.rough));
    props.push(new THREE.Vector4(m.spec, m.hardness, m.durability, m.cleavage));
    extra.push(new THREE.Vector2(m.fe, m.translucency));
  }
  const cdf = buildModeCDF(litho.mode).map((c) => c[0]);
  const phen = litho.phenocryst || { frac: 0, size: 0.02, mineral: 'kfeldspar' };
  const phenId = MINERAL_LIST.findIndex((m) => m.name === phen.mineral);

  const mat = new THREE.ShaderMaterial({
    // GLSL ES 3.00 is mandatory here, not a preference: the crystal-aggregate
    // hash is built on `uint` arithmetic and bitwise ops (^, >>, *) which do
    // not exist in GLSL ES 1.00. Emulating it in floats would drift from the
    // CPU hash and the grains would stop lining up with the eroded relief.
    glslVersion: THREE.GLSL3,
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      uMinAlbedoRough: { value: albedoRough },
      uMinProps: { value: props },
      uMinExtra: { value: extra },
      uModeCDF: { value: cdf },
      uCellSize: { value: litho.grain * 1.15 },
      uCellSize2: { value: litho.grain * 1.15 * 0.42 },
      uGrainSigma: { value: litho.grainSigma },
      uSeriate: { value: litho.seriate },
      uFoliation: { value: litho.foliation || 0 },
      uPhenocryst: { value: new THREE.Vector3(phen.frac, phen.size, Math.max(0, phenId)) },
      uGrainSeed: { value: seed >>> 0 },

      uCameraPos: { value: new THREE.Vector3() },
      uSunDir: { value: new THREE.Vector3(0.45, 0.78, 0.44).normalize() },
      uSunColor: { value: new THREE.Color(1.0, 0.96, 0.88).multiplyScalar(3.2) },
      uSkyColor: { value: new THREE.Color(0.28, 0.38, 0.55).multiplyScalar(0.85) },
      uGroundColor: { value: new THREE.Color(0.16, 0.13, 0.10).multiplyScalar(0.7) },
      uExposure: { value: 1.0 },

      uStainColor: { value: new THREE.Color(...(litho.stain || [0.42, 0.24, 0.10])) },
      uStainStrength: { value: litho.stainStrength ?? 0.7 },
      uWeatherAge: { value: opts.weatherAge ?? 0.6 },
      uLichen: { value: opts.lichen ?? 0.25 },
      uLichenColorA: { value: new THREE.Color(0.42, 0.46, 0.30) },
      uLichenColorB: { value: new THREE.Color(0.62, 0.63, 0.52) },
      uCaseHardening: { value: opts.caseHardening ?? 0.4 },
      uDust: { value: opts.dust ?? 0.3 },
      uWetness: { value: opts.wetness ?? 0.0 },
      uMicroRelief: { value: opts.microRelief ?? 1.0 },
      uRetreatScale: { value: opts.retreatScale ?? 8.0 },
      uDebugMode: { value: 0 },
    },
  });
  return mat;
}
