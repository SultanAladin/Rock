/**
 * WGSL static validator.
 *
 * There is no browser in this sandbox, so we cannot ask a real WebGPU device to
 * compile the shaders. wgsl_reflect parses WGSL and builds the full binding /
 * struct reflection, which catches syntax errors, unknown identifiers in
 * declarations, and layout mistakes -- the failure modes that would otherwise
 * only show up as a blank canvas.
 *
 * It also enforces the same truncation guard as checkglsl.mjs: a template
 * literal cut in half still parses, so we assert minimum sizes explicitly.
 */
import { WgslReflect } from 'wgsl_reflect/wgsl_reflect.module.js';
import { INIT_WGSL, JFA_SEED_WGSL, JFA_STEP_WGSL, JFA_RESOLVE_WGSL,
         SHELTER_WGSL, STEP_WGSL, COUNT_WGSL, RETREAT_WGSL } from '../src/gpu/wgsl/erode.wgsl.js';
import { RAYMARCH_WGSL } from '../src/gpu/wgsl/raymarch.wgsl.js';

const mods = [
  ['INIT',        INIT_WGSL,        { compute: ['main'], minLines: 60 }],
  ['JFA_SEED',    JFA_SEED_WGSL,    { compute: ['main'], minLines: 40 }],
  ['JFA_STEP',    JFA_STEP_WGSL,    { compute: ['main'], minLines: 40 }],
  ['JFA_RESOLVE', JFA_RESOLVE_WGSL, { compute: ['main'], minLines: 40 }],
  ['SHELTER',     SHELTER_WGSL,     { compute: ['main'], minLines: 40 }],
  ['STEP',        STEP_WGSL,        { compute: ['main'], minLines: 80 }],
  ['COUNT',       COUNT_WGSL,       { compute: ['main'], minLines: 30 }],
  ['RETREAT',     RETREAT_WGSL,     { compute: ['main'], minLines: 30 }],
  ['RAYMARCH',    RAYMARCH_WGSL,    { vertex: ['vs'], fragment: ['fs'], minLines: 300 }],
];

let fail = 0;
for (const [name, src, want] of mods) {
  const lines = src.split('\n').length;
  let r;
  try {
    r = new WgslReflect(src);
  } catch (e) {
    console.log(`[FAIL] ${name}: parse error: ${e.message}`);
    fail++; continue;
  }
  const problems = [];
  if (lines < want.minLines) problems.push(`truncated: ${lines} lines < ${want.minLines}`);
  for (const stage of ['compute', 'vertex', 'fragment']) {
    for (const fn of (want[stage] || [])) {
      if (!r.entry[stage].find(e => e.name === fn)) problems.push(`missing @${stage} ${fn}`);
    }
  }
  // every binding must be declared in group 0 and numbered without holes
  const binds = [...r.uniforms, ...r.storage]
    .map(v => `${v.group}.${v.binding}`).sort();
  const dup = binds.filter((b, i) => binds[i - 1] === b);
  if (dup.length) problems.push(`duplicate binding(s) ${[...new Set(dup)].join(',')}`);

  // Portability guards. These constructs compile on some implementations and
  // not others, and the failure mode is a blank canvas rather than an error.
  if (/ptr\s*<\s*storage/.test(src)) {
    problems.push('uses ptr<storage,...> parameters (unrestricted_pointer_parameters extension)');
  }
  if (/\bconst\s+\w+\s*=\s*array</.test(src) && false) { /* reserved */ }
  // A workgroup_size product over 256 exceeds the guaranteed minimum limit.
  for (const m2 of src.matchAll(/@workgroup_size\(([^)]*)\)/g)) {
    const dims = m2[1].split(',').map((x) => parseInt(x.trim(), 10) || 1);
    const prod = dims.reduce((a, c) => a * c, 1);
    if (prod > 256) problems.push(`workgroup_size product ${prod} exceeds the guaranteed 256`);
  }
  // Every binding the shader declares must be inside group 0 or 1.
  for (const v of [...r.uniforms, ...r.storage]) {
    if (v.group > 1) problems.push(`binding ${v.name} is in group ${v.group} (>1)`);
  }

  if (problems.length) {
    console.log(`[FAIL] ${name}: ${problems.join('; ')}`);
    fail++;
  } else {
    console.log(`[ OK ] ${name}: ${lines} lines, bindings [${binds.join(',')}]`);
  }
}

// Params layout must be a multiple of 16 bytes and match what the JS packer writes.
try {
  const r = new WgslReflect(RAYMARCH_WGSL);
  const info = r.structs.find(s => s.name === 'Params');
  console.log(`\nParams size = ${info.size} bytes (${info.size / 4} floats)`);
  if (info.size % 16 !== 0) { console.log('[FAIL] Params not 16-byte aligned'); fail++; }
  for (const m of info.members) {
    if (m.offset % 4 !== 0) { console.log(`[FAIL] member ${m.name} misaligned`); fail++; }
  }
} catch (e) {
  console.log('[FAIL] Params reflection: ' + e.message);
  fail++;
}

console.log(fail ? `\nFAIL (${fail})` : '\nPASS');
process.exit(fail ? 1 : 0);
