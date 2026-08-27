/*
 * shaders.js — GLSL sources for the real-time SDF rock-erosion renderer.
 *
 * The renderer is a single-pass raymarcher: each screen pixel casts a ray that
 * sphere-traces a signed distance field. The rock is defined analytically so we
 * can carve it with procedural 3D noise to simulate REAL erosion processes
 * fully on the GPU, in real time, and scrub an "erosion time" slider.
 *
 * The language is GLSL ES 1.0 (WebGL 1) for maximum browser compatibility.
 */

/* ----------------------------------------------------------------------------
 * VERTEX SHADER — fullscreen triangle/quad.
 * ------------------------------------------------------------------------- */
const ROCK_VERT = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
    vUv = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

/* ----------------------------------------------------------------------------
 * FRAGMENT SHADER
 * ------------------------------------------------------------------------- */
const ROCK_FRAG = `
precision highp float;

varying vec2 vUv;

uniform vec2  uRes;          // drawing-buffer size
uniform float uTime;

// camera
uniform vec3  uCamPos;
uniform vec3  uCamTarget;
uniform float uFov;

// erosion + shape
uniform float uE;           // erosion amount 0..1 (the "time" slider)
uniform float uErodeType;   // 0..5 erosion process
uniform float uShape;       // 0..6 base rock shape
uniform float uShapeRough;  // macro shape rough-ness
uniform float uSeed;        // shape/material seed
uniform vec3  uRockPos;     // offset of rock so it rests on the ground

// material (built from the selected rock type on the CPU)
uniform vec3  uC1;          // base mineral colour
uniform vec3  uC2;          // primary variation colour
uniform vec3  uC3;          // accent colour (specks / veins / bands)
uniform float uSpeck;       // mineral speckle amount
uniform float uBand;        // sedimentary / foliation banding
uniform float uBandFreq;
uniform float uMoss;        // biological weathering (moss / lichen)
uniform float uWeather;     // weathering stain amount
uniform vec3  uWeatherColor;
uniform float uDetailAmp;   // micro-detail amplitude (abrasion)
uniform float uDetailFreq;
uniform float uCarveAmt;    // how much material the carve pattern removes
uniform float uCarveFreq;
uniform float uStriAmt;     // striation amount
uniform float uStriFreq;
uniform float uShininess;   // specular exponent (crystalline rocks glossier)
uniform float uSpecAmt;

// lighting
uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform vec3  uSkyTop;
uniform vec3  uSkyHorizon;
uniform vec3  uSunColor;

#define PI 3.14159265359

/* ----------------------------- noise ----------------------------------- */
float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i + vec3(0.0, 0.0, 0.0)), hash13(i + vec3(1.0, 0.0, 0.0)), f.x),
            mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x),
            mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x), f.y),
        f.z);
}

/* Cheap 2-octave FBM — used in the SDF where it's evaluated many times/px. */
float fbm(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 2; i++) {
        s += a * noise(p);
        p *= 2.02;
        a *= 0.5;
    }
    return s / 0.75;
}

/* 3-octave FBM — used once per surface pixel for colour detail. */
float fbm3(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 3; i++) {
        s += a * noise(p);
        p *= 2.02;
        a *= 0.5;
    }
    return s / 0.875;
}

/* F1 / F2 cellular (Voronoi) distances — drives pits + cracks. */
vec2 voronoiFC(vec3 x) {
    vec3 ip = floor(x);
    vec3 fp = fract(x);
    float f1 = 1e9;
    float f2 = 1e9;
    for (int i = -1; i <= 1; i++)
    for (int j = -1; j <= 1; j++)
    for (int k = -1; k <= 1; k++) {
        vec3 g = vec3(float(i), float(j), float(k));
        vec3 o = vec3(hash13(ip + g));
        vec3 r = g + o - fp;
        float d = dot(r, r);
        if (d < f1) { f2 = f1; f1 = d; }
        else if (d < f2) { f2 = d; }
    }
    return vec2(sqrt(f1), sqrt(f2));
}

/* Directional grooves (hydraulic / aeolian striations). */
float striation(vec3 p) {
    float n = fbm(p * 1.5 + 7.0);
    float s = sin((p.x * 1.1 + p.z * 0.7) * uStriFreq + n * 3.0);
    return smoothstep(-0.25, 0.55, s);
}

/* ------------------------- distance primitives ------------------------- */
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

/* A little cluster of boulders, positions/radii derived from the seed. */
float clusterSD(vec3 p) {
    float d = 1e9;
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        vec3 c = vec3(hash13(vec3(fi, uSeed, 1.0)) - 0.5,
                      hash13(vec3(fi, uSeed, 2.0)) - 0.35,
                      hash13(vec3(fi, uSeed, 3.0)) - 0.5);
        c *= 1.05;
        float r = 0.45 + 0.35 * hash13(vec3(fi, uSeed, 4.0));
        d = smin(d, length(p - c) - r, 0.55);
    }
    return d;
}

/* --------------------------- base rock shapes --------------------------- */
float baseShape(vec3 p) {
    if (uShape < 0.5)      return length(p) - 0.9;                          // round boulder
    else if (uShape < 1.5) return sdRoundBox(p, vec3(0.66, 0.66, 0.66), 0.16); // block
    else if (uShape < 2.5) return sdRoundBox(p, vec3(0.82, 0.30, 0.60), 0.14); // slab
    else if (uShape < 3.5) return sdRoundBox(p, vec3(0.40, 0.98, 0.42), 0.05); // crystal shard
    else if (uShape < 4.5) return length(p / vec3(1.05, 0.78, 0.92)) - 0.9; // pebble
    else if (uShape < 5.5) return sdRoundBox(p, vec3(0.92, 0.44, 0.52), 0.22); // shelf
    else                   return clusterSD(p);                             // boulder cluster
}

/* Space warp: turns a primitive into an irregular boulder. */
vec3 warp(vec3 p) {
    return p + uShapeRough * vec3(fbm(p * 1.3 + uSeed * 13.0) - 0.5,
                                  fbm(p * 1.3 + uSeed * 29.0 + 7.0) - 0.5,
                                  fbm(p * 1.3 + uSeed * 53.0 + 21.0) - 0.5);
}

/* Erosion-process weights -> which carve style dominates. */
void erodeWeights(float type, out float wPit, out float wCrack, out float wStri) {
    if      (type < 0.5) { wPit = 0.15; wCrack = 0.10; wStri = 0.00; } // abrasion
    else if (type < 1.5) { wPit = 0.20; wCrack = 0.15; wStri = 0.70; } // hydraulic
    else if (type < 2.5) { wPit = 0.78; wCrack = 0.14; wStri = 0.08; } // chemical
    else if (type < 3.5) { wPit = 0.10; wCrack = 0.15; wStri = 0.75; } // wind
    else if (type < 4.5) { wPit = 0.10; wCrack = 0.85; wStri = 0.05; } // frost
    else                 { wPit = 0.42; wCrack = 0.34; wStri = 0.24; } // combined
}

/* -------------------------- the rock's SDF ------------------------------ */
float rockSDF(vec3 p) {
    vec3 q = warp(p);
    float d = baseShape(q);

    // Micro-detail (rocky roughness). Abrasion smooths it off over time.
    float detail = fbm(q * uDetailFreq + uSeed * 3.0) - 0.5;
    d -= uDetailAmp * (1.0 - uE) * detail;

    // Erosion carve: pits (honeycomb/karst), cracks (jointing/frost),
    // striations (water/wind). All scale with the erosion slider.
    vec2 vc = voronoiFC(q * uCarveFreq);
    float pit    = 1.0 - smoothstep(0.0, 0.55, vc.x);          // pit at cell centres
    float crack  = 1.0 - smoothstep(0.0, 0.11, vc.y - vc.x);   // chasm at cell borders
    float stri   = striation(q);

    float wPit, wCrack, wStri;
    erodeWeights(uErodeType, wPit, wCrack, wStri);

    float carve = (pit * wPit + crack * wCrack + stri * wStri) * uCarveAmt;
    d += carve * uE;

    return d;
}

float map(vec3 p) {
    float dRock = rockSDF(p - uRockPos);
    float dGround = p.y + 1.05;           // ground plane at y = -1.05
    return min(dRock, dGround);
}

float map2(vec3 p) { return min(rockSDF(p - uRockPos), p.y + 1.05); }

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(1.0, -1.0) * 0.0009;
    return normalize(e.xyy * map2(p + e.xyy) + e.yyx * map2(p + e.yyx) +
                     e.yxy * map2(p + e.yxy) + e.xxx * map2(p + e.xxx));
}

float softshadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 6; i++) {
        float h = map(ro + rd * t);
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.3);
        if (res < 0.005 || t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 2; i++) {
        float h = 0.02 + 0.12 * float(i);
        float d = map2(p + n * h);
        occ += (h - d) * sca;
        sca *= 0.75;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

/* --------------------------- surface colour ----------------------------- */
vec3 rockColor(vec3 p) {
    vec3 alb = uC1;

    // Mineral variation.
    float g = fbm3(p * 2.5 + uSeed * 5.0);
    alb = mix(uC1, uC2, smoothstep(0.35, 0.70, g));

    // Speckle / veins (accent colour).
    float sp = noise(p * 16.0 + uSeed * 9.0);
    alb = mix(alb, uC3, uSpeck * smoothstep(0.80, 0.96, sp));
    float sp2 = noise(p * 14.0 + uSeed * 2.0);
    alb = mix(alb, uC2 * 0.55, uSpeck * smoothstep(0.05, 0.20, sp2));

    // Sedimentary / foliated banding (marbles, gneiss, sandstone, slate).
    if (uBand > 0.001) {
        float b = sin(p.y * uBandFreq + fbm3(p * 1.3 + uSeed * 4.0) * 1.6);
        b = smoothstep(-0.30, 0.42, b);
        alb = mix(alb, uC3, uBand * b * 0.85);
    }

    // Crevice darkening from the carve pattern (evaluated in the SAME warped
    // space as the geometry so colour matches the carved holes exactly).
    vec3 q = warp(p);
    vec2 vc = voronoiFC(q * uCarveFreq);
    float pit = 1.0 - smoothstep(0.0, 0.55, vc.x);
    float crack = 1.0 - smoothstep(0.0, 0.11, vc.y - vc.x);
    float cavity = clamp(pit * 0.6 + crack, 0.0, 1.0);
    alb *= 1.0 - 0.45 * cavity * uE;

    // Weathering stains (iron oxide / carbonate crust) accumulate in cavities.
    float w = fbm3(p * 1.1 + uSeed * 2.0);
    alb = mix(alb, uWeatherColor, uWeather * smoothstep(0.45, 0.85, w) * (0.35 + 0.65 * cavity));

    // Biological weathering: moss / lichen takes hold in damp crevices over time.
    float mossMask = smoothstep(0.50, 0.75, fbm3(p * 2.0 + uSeed * 11.0)) * cavity;
    alb = mix(alb, vec3(0.34, 0.44, 0.20), uMoss * (0.15 + 0.85 * uE) * mossMask * 1.4);

    // Fresh-broken edge glint on sharp cracks (crystal / glass rocks).
    float edge = 1.0 - smoothstep(0.0, 0.04, vc.y - vc.x);
    alb += uC3 * edge * uSpeck * 0.25 * uE;

    return alb;
}

vec3 groundColor(vec3 p) {
    vec3 base = vec3(0.35, 0.31, 0.27);
    float n = fbm(p * 2.2 + 3.0);
    base = mix(base, vec3(0.45, 0.40, 0.34), n);
    float sp = noise(p * 24.0 + 7.0);
    base = mix(base, vec3(0.30, 0.28, 0.25), smoothstep(0.7, 0.95, sp));
    return base;
}

/* ----------------------------- lighting --------------------------------- */
vec3 applyLight(vec3 p, vec3 n, vec3 rd, vec3 alb) {
    vec3 l = normalize(uLightDir);
    float diff = clamp(dot(n, l), 0.0, 1.0);

    float shadow = softshadow(p + n * 0.002, l, 0.02, 6.0, 6.0);
    float ao = calcAO(p, n);

    float sky = 0.5 + 0.5 * n.y;
    vec3 amb = mix(uLightColor * 0.30, uSkyTop, sky);
    vec3 sun = uLightColor * diff * shadow;

    vec3 col = alb * (sun / PI + amb * (0.5 + 0.5 * ao));

    // Specular (subdued — rocks are mostly diffuse).
    vec3 r = reflect(-rd, n);
    float spec = pow(clamp(dot(r, l), 0.0, 1.0), uShininess);
    col += uLightColor * spec * shadow * uSpecAmt;

    // Fresnel rim light.
    float fre = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0);
    col += uSkyTop * fre * 0.07;

    return col;
}

vec3 background(vec3 rd) {
    vec3 l = normalize(uLightDir);
    vec3 col = mix(uSkyHorizon, uSkyTop, smoothstep(-0.1, 0.6, rd.y));
    float sun = clamp(dot(rd, l), 0.0, 1.0);
    col += uSunColor * pow(sun, 512.0) * 4.0;
    col += uLightColor * pow(sun, 8.0) * 0.20;
    return col;
}

/* ------------------------------ main ------------------------------------ */
void main() {
    // Screen-space direction, aspect corrected.
    vec2 pla = (2.0 * gl_FragCoord.xy - uRes) / uRes.y;

    // Camera basis (up = +Y; no roll).
    vec3 ww = normalize(uCamTarget - uCamPos);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = cross(uu, ww);
    float f = 1.0 / tan(uFov * 0.5);
    vec3 ro = uCamPos;
    vec3 rd = normalize(pla.x * uu + pla.y * vv + f * ww);

    // --- ray-march the SDF ---------------------------------------------
    vec3 col = background(rd);
    float t = 0.02;
    float tmax = 30.0;
    float eps = 0.0006;
    bool hit = false;

    for (int i = 0; i < 40; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < eps * t) { hit = true; break; }
        if (t > tmax) break;
        t += d * 0.82;
    }

    if (hit) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        float dR = rockSDF(p - uRockPos);
        float dG = p.y + 1.05;

        vec3 alb = (dR < dG) ? rockColor(p - uRockPos) : groundColor(p);
        col = applyLight(p, n, rd, alb);

        // Simple distance fog toward the horizon.
        col = mix(col, uSkyHorizon, 1.0 - exp(-0.0016 * t * t));
    }

    // Film-like tone curve + gentle vignette + dither to hide banding.
    col = col / (1.0 + col);                          // Reinhard
    col = pow(col, vec3(1.0 / 2.2));

    vec2 qv = gl_FragCoord.xy / uRes;
    float vig = 1.0 - smoothstep(0.65, 1.45, length(qv - 0.5));
    col *= mix(0.86, 1.0, vig);

    col += (hash13(vec3(gl_FragCoord.xy, uTime)) - 0.5) / 255.0;

    gl_FragColor = vec4(col, 1.0);
}
`;

/* Expose on a global for the non-module script loading used here. */
window.RockShaders = { vert: ROCK_VERT, frag: ROCK_FRAG };
