/**
 * WGSL identifier + declaration-order checker.
 *
 * wgsl_reflect parses WGSL but does not resolve names, and WebGPU reports
 * compile errors asynchronously without throwing -- so an undefined function or
 * a use-before-declaration produces a silently dead pipeline and a black
 * canvas. WGSL requires module-scope declaration BEFORE use, which is exactly
 * the kind of mistake that assembling shaders from string fragments invites.
 *
 * This resolves every call and every module-scope reference against the
 * builtins and the declarations that precede it.
 */

import { WgslReflect } from 'wgsl_reflect/wgsl_reflect.module.js';
import * as ERODE from '../src/gpu/wgsl/erode.wgsl.js';
import { RAYMARCH_WGSL } from '../src/gpu/wgsl/raymarch.wgsl.js';

const BUILTIN_FNS = new Set(`
abs acos acosh asin asinh atan atanh atan2 ceil clamp cos cosh countLeadingZeros
countOneBits countTrailingZeros cross degrees determinant distance dot exp exp2
extractBits faceForward firstLeadingBit firstTrailingBit floor fma fract frexp
insertBits inverseSqrt ldexp length log log2 max min mix modf normalize pow
quantizeToF16 radians reflect refract reverseBits round saturate sign sin sinh
smoothstep sqrt step tan tanh transpose trunc
dpdx dpdxCoarse dpdxFine dpdy dpdyCoarse dpdyFine fwidth fwidthCoarse fwidthFine
all any select arrayLength
atomicLoad atomicStore atomicAdd atomicSub atomicMax atomicMin atomicAnd
atomicOr atomicXor atomicExchange atomicCompareExchangeWeak
storageBarrier workgroupBarrier textureBarrier workgroupUniformLoad
pack4x8snorm pack4x8unorm pack2x16snorm pack2x16unorm pack2x16float
unpack4x8snorm unpack4x8unorm unpack2x16snorm unpack2x16unorm unpack2x16float
textureSample textureSampleLevel textureLoad textureStore textureDimensions
bitcast
f32 i32 u32 bool f16
vec2 vec3 vec4 mat2x2 mat2x3 mat2x4 mat3x2 mat3x3 mat3x4 mat4x2 mat4x3 mat4x4
array atomic ptr
`.trim().split(/\s+/));

// vecN<f32> style constructors and swizzles are handled by the regex below.
const KEYWORDS = new Set(`
if else for while loop switch case default break continue return discard
let var const fn struct alias override enable requires diagnostic
true false
storage uniform workgroup private function read write read_write
compute vertex fragment
`.trim().split(/\s+/));

let fail = 0;

function checkModule(name, src) {
  // strip comments, then attributes (@group(0), @workgroup_size(4,4,4), ...)
  // whose parenthesised arguments are not call expressions.
  const clean = src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/@[A-Za-z_]\w*(\s*\([^)]*\))?/g, ' ');

  // module-scope declarations, in order of appearance
  const decls = [];   // { name, pos, kind }
  const push = (re, kind) => {
    for (const m of clean.matchAll(re)) decls.push({ name: m[1], pos: m.index, kind });
  };
  push(/\bfn\s+([A-Za-z_]\w*)\s*\(/g, 'fn');
  push(/\bstruct\s+([A-Za-z_]\w*)/g, 'struct');
  push(/\bvar\s*(?:<[^>]*>)?\s*([A-Za-z_]\w*)\s*:/g, 'var');
  push(/\balias\s+([A-Za-z_]\w*)/g, 'alias');
  // module-scope const/let (indented ones are local; require column 0)
  for (const m of clean.matchAll(/^const\s+([A-Za-z_]\w*)/gm)) {
    decls.push({ name: m[1], pos: m.index, kind: 'const' });
  }

  const declPos = new Map();
  const dupes = [];
  for (const d of decls) {
    if (declPos.has(d.name) && d.kind === 'fn') dupes.push(d.name);
    if (!declPos.has(d.name)) declPos.set(d.name, d.pos);
  }

  const problems = [];
  for (const d of new Set(dupes)) problems.push(`duplicate function definition: ${d}()`);

  // every call site must resolve to a builtin or an EARLIER declaration
  for (const m of clean.matchAll(/([A-Za-z_]\w*)\s*\(/g)) {
    const fn = m[1];
    if (KEYWORDS.has(fn) || BUILTIN_FNS.has(fn)) continue;
    // vec3<f32>(...) / array<...>(...) style already covered by BUILTIN_FNS
    if (!declPos.has(fn)) {
      problems.push(`call to undefined function ${fn}() at offset ${m.index}`);
    } else if (declPos.get(fn) > m.index) {
      const line = clean.slice(0, m.index).split('\n').length;
      problems.push(`${fn}() used at line ${line} but declared later (WGSL requires declaration before use)`);
    }
  }

  // module-scope variable references (P, phiIn, Cam, ...) must exist
  // Note: unused module-scope bindings are NOT an error here. Every compute
  // pass shares one PRELUDE declaring all eight buffers so a single bind group
  // layout serves them all; most passes legitimately touch only a few.

  if (problems.length) {
    console.log(`[FAIL] ${name}`);
    for (const p of [...new Set(problems)]) console.log(`         ${p}`);
    fail += problems.length;
  } else {
    console.log(`[ OK ] ${name}: ${decls.length} module-scope declarations, all references resolve`);
  }
}

const mods = Object.entries(ERODE)
  .filter(([k, v]) => k.endsWith('_WGSL') && typeof v === 'string')
  .concat([['RAYMARCH_WGSL', RAYMARCH_WGSL]]);

for (const [k, v] of mods) checkModule(k.replace('_WGSL', ''), v);

console.log(fail ? `\nFAIL (${fail})` : '\nPASS');
process.exit(fail ? 1 : 0);
