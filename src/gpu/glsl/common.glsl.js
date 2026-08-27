/**
 * Shared GLSL: integer hashing identical to core/rng.js, gradient noise, and
 * the Laguerre crystal-aggregate evaluator identical to core/petrology.js.
 *
 * "Identical" is meant literally - the CPU solver decides where rock is removed
 * using the same crystal boundaries the fragment shader colours. Any drift here
 * shows up as quartz micro-relief that does not line up with quartz colour,
 * which is exactly the tell that makes procedural stone look painted.
 */

export const HASH_GLSL = /* glsl */`
uint hashU32(uint x){
  x ^= x >> 16u; x *= 0x7feb352du;
  x ^= x >> 15u; x *= 0x846ca68bu;
  x ^= x >> 16u; return x;
}
uint hash3i(ivec3 p, uint seed){
  uint h = uint(p.x) * 0x8da6b343u ^ uint(p.y) * 0xd8163841u
         ^ uint(p.z) * 0xcb1ab31fu ^ seed * 0x165667b1u;
  return hashU32(h);
}
float u2f(uint h){ return float(h) * (1.0 / 4294967296.0); }
`;

export const NOISE_GLSL = /* glsl */`
vec3 grad3(ivec3 p, uint seed){
  uint h = hash3i(p, seed);
  uint h2 = hashU32(h ^ 0x9e3779b9u);
  float z = u2f(h) * 2.0 - 1.0;
  float t = u2f(h2) * 6.2831853;
  float r = sqrt(max(0.0, 1.0 - z*z));
  return vec3(r*cos(t), r*sin(t), z);
}
float fade1(float t){ return t*t*t*(t*(t*6.0-15.0)+10.0); }
float pnoise3(vec3 P, uint seed){
  ivec3 ip = ivec3(floor(P));
  vec3 f = P - floor(P);
  vec3 u = vec3(fade1(f.x), fade1(f.y), fade1(f.z));
  float n000 = dot(grad3(ip+ivec3(0,0,0),seed), f-vec3(0,0,0));
  float n100 = dot(grad3(ip+ivec3(1,0,0),seed), f-vec3(1,0,0));
  float n010 = dot(grad3(ip+ivec3(0,1,0),seed), f-vec3(0,1,0));
  float n110 = dot(grad3(ip+ivec3(1,1,0),seed), f-vec3(1,1,0));
  float n001 = dot(grad3(ip+ivec3(0,0,1),seed), f-vec3(0,0,1));
  float n101 = dot(grad3(ip+ivec3(1,0,1),seed), f-vec3(1,0,1));
  float n011 = dot(grad3(ip+ivec3(0,1,1),seed), f-vec3(0,1,1));
  float n111 = dot(grad3(ip+ivec3(1,1,1),seed), f-vec3(1,1,1));
  float x00 = mix(n000,n100,u.x), x10 = mix(n010,n110,u.x);
  float x01 = mix(n001,n101,u.x), x11 = mix(n011,n111,u.x);
  return mix(mix(x00,x10,u.y), mix(x01,x11,u.y), u.z) * 1.1547;
}
// Self-affine fBm with Hurst exponent H (see core/noise.js for why H matters).
float fbmH(vec3 p, uint seed, float H, int oct){
  float sum = 0.0, amp = 1.0, norm = 0.0, f = 1.0;
  const float lac = 2.03;
  for(int i = 0; i < 8; i++){
    if(i >= oct) break;
    sum  += amp * pnoise3(p * f, seed + uint(i) * 7919u);
    norm += amp * amp;
    amp  *= pow(lac, -H);
    f    *= lac;
  }
  return sum / sqrt(max(norm, 1e-6));
}
`;

/**
 * Mineral table, uploaded as uniforms. Layout must match MINERAL_LIST order.
 * We pack: albedo.rgb + roughness, then spec/hardness/durability/cleavage,
 * then fe/translucency.
 */
export const GRAIN_GLSL = /* glsl */`
uniform vec4  uMinAlbedoRough[6];   // rgb = linear albedo, a = base roughness
uniform vec4  uMinProps[6];         // x=spec y=hardness z=durability w=cleavage
uniform vec2  uMinExtra[6];         // x=fe, y=translucency
uniform float uModeCDF[6];
uniform float uCellSize;
uniform float uCellSize2;
uniform float uGrainSigma;
uniform float uSeriate;
uniform float uFoliation;
uniform vec3  uPhenocryst;          // x=frac y=size z=mineralIndex
uniform uint  uGrainSeed;

struct Grain {
  int   id;
  float boundary;   // 0 at a grain boundary, 1 deep inside a crystal
  float jitter;     // per-crystal random in [0,1)
  vec3  axis;       // cleavage-plane normal, object space
  float size;
  float pheno;
};

vec3 foliate(vec3 p){
  float f = uFoliation;
  if(f <= 0.0) return p;
  return vec3(p.x / (1.0 + 0.55*f), p.y * (1.0 + 1.35*f), p.z / (1.0 + 0.15*f));
}

// Laguerre (radical Voronoi) cell lookup. Returns key in .x (as uint bits via
// out param), d1/d2 and the site position.
void laguerre(vec3 p, float cs, uint salt, out uint key, out float d1, out float d2, out vec3 site){
  ivec3 g = ivec3(floor(p / cs));
  d1 = 1e30; d2 = 1e30; key = 0u; site = vec3(0.0);
  for(int k = -1; k <= 1; k++)
  for(int j = -1; j <= 1; j++)
  for(int i = -1; i <= 1; i++){
    ivec3 c = g + ivec3(i,j,k);
    uint h0 = hash3i(c, uGrainSeed ^ salt);
    uint h1 = hashU32(h0 ^ 0x9e3779b9u);
    uint h2 = hashU32(h1 ^ 0x85ebca6bu);
    uint h3 = hashU32(h2 ^ 0xc2b2ae35u);
    vec3 q = (vec3(c) + vec3(u2f(h0), u2f(h1), u2f(h2))) * cs;
    float w = exp(uGrainSigma * (u2f(h3) * 2.0 - 1.0)) * cs * 0.5;
    float dd = length(q - p) - w;
    if(dd < d1){ d2 = d1; d1 = dd; key = h0; site = q; }
    else if(dd < d2){ d2 = dd; }
  }
}

int pickMineral(float u){
  for(int i = 0; i < 6; i++) if(u <= uModeCDF[i]) return i;
  return 5;
}

Grain packGrain(int id, uint key, float boundary, float size, float pheno){
  Grain g;
  g.id = id; g.boundary = boundary; g.size = size; g.pheno = pheno;
  float h1 = u2f(hashU32(key ^ 0x165667b1u));
  float h2 = u2f(hashU32(key ^ 0x9e3779b1u));
  g.jitter = u2f(hashU32(key ^ 0x27d4eb2du));
  float th = acos(clamp(2.0*h1 - 1.0, -1.0, 1.0));
  float ph = h2 * 6.2831853;
  g.axis = vec3(sin(th)*cos(ph), sin(th)*sin(ph), cos(th));
  return g;
}

Grain sampleGrain(vec3 pObj){
  vec3 p = foliate(pObj);
  uint key; float d1, d2; vec3 site;

  if(uPhenocryst.x > 0.0){
    laguerre(p, uPhenocryst.y * 1.3, 0x51ed27u, key, d1, d2, site);
    if(u2f(hashU32(key ^ 0x1b873593u)) < uPhenocryst.x){
      float r = uPhenocryst.y * 0.5 * (0.7 + 0.6 * u2f(hashU32(key ^ 0x27d4eb2fu)));
      float dist = length(site - p);
      if(dist < r){
        return packGrain(int(uPhenocryst.z), key,
                         clamp((r - dist) / (0.25 * r), 0.0, 1.0), uPhenocryst.y, 1.0);
      }
    }
  }

  laguerre(p, uCellSize, 0u, key, d1, d2, site);
  float size = uCellSize;
  if(uSeriate > 0.0 && u2f(hashU32(key ^ 0x7ed55d16u)) < uSeriate * 0.45){
    laguerre(p, uCellSize2, 0x2545f4u, key, d1, d2, site);
    size = uCellSize2;
  }
  int id = pickMineral(u2f(hashU32(key ^ 0xa5a5a5a5u)));
  float boundary = clamp((d2 - d1) / (0.35 * size), 0.0, 1.0);
  return packGrain(id, key, boundary, size, 0.0);
}
`;
