import fs from 'fs';
import { HASH_GLSL, NOISE_GLSL, GRAIN_GLSL } from '../src/gpu/glsl/common.glsl.js';
const src=fs.readFileSync('src/gpu/rockMaterial.js','utf8');
const glsl=(src.match(/\/\* glsl \*\/`([\s\S]*?)`;/g)||[]).join('\n')+HASH_GLSL+NOISE_GLSL+GRAIN_GLSL;
const declared=new Set([...glsl.matchAll(/^\s*uniform\s+\w+\s+(\w+)/gm)].map(m=>m[1]));
// uniforms provided in the JS object
const ub=src.slice(src.indexOf('uniforms: {'));
const provided=new Set([...ub.matchAll(/^\s{6}(u[A-Z]\w*):\s*\{/gm)].map(m=>m[1]));
const builtin=new Set(['modelMatrix','modelViewMatrix','projectionMatrix','viewMatrix','normalMatrix','cameraPosition','isOrthographic']);
const missing=[...declared].filter(d=>!provided.has(d)&&!builtin.has(d));
const unused=[...provided].filter(p=>!declared.has(p));
console.log('declared',declared.size,'provided',provided.size);
console.log('DECLARED BUT NOT PROVIDED:',missing);
console.log('PROVIDED BUT NOT DECLARED:',unused);
// attributes
const attrs=[...glsl.matchAll(/^\s*in\s+\w+\s+(a\w+);/gm)].map(m=>m[1]);
console.log('vertex attributes used:',attrs);
const geo=fs.readFileSync('src/app/main.js','utf8');
console.log('attributes set:',[...geo.matchAll(/setAttribute\('(\w+)'/g)].map(m=>m[1]));

// ---------------------------------------------------------------------------
// WGSL uniform contract.
//
// The Params struct is written by hand in JS (erosionEngine.js's O table) and
// read by name in WGSL. A rename on one side and not the other produces
// garbage physics with no error, so pin the two together.
// ---------------------------------------------------------------------------
{
  const { WgslReflect } = await import('wgsl_reflect/wgsl_reflect.module.js');
  const { RAYMARCH_WGSL } = await import('../src/gpu/wgsl/raymarch.wgsl.js');
  const src = await import('node:fs').then((fs) =>
    fs.readFileSync(new URL('../src/gpu/erosionEngine.js', import.meta.url), 'utf8'));

  const P = new WgslReflect(RAYMARCH_WGSL).structs.find((s) => s.name === 'Params');
  const shaderOffsets = new Map(P.members.map((m) => [m.name, m.offset / 4]));

  // parse the O table out of the driver
  const tbl = src.slice(src.indexOf('const O = {'), src.indexOf('};', src.indexOf('const O = {')));
  const jsOffsets = new Map();
  for (const m of tbl.matchAll(/(\w+):\s*(\d+)/g)) jsOffsets.set(m[1], +m[2]);

  let bad = 0;
  for (const [name, off] of jsOffsets) {
    if (!shaderOffsets.has(name)) {
      console.log(`[FAIL] JS packs "${name}" but the WGSL Params struct has no such member`);
      bad++;
    } else if (shaderOffsets.get(name) !== off) {
      console.log(`[FAIL] "${name}" offset mismatch: JS ${off}, WGSL ${shaderOffsets.get(name)}`);
      bad++;
    }
  }
  for (const [name] of shaderOffsets) {
    if (!jsOffsets.has(name) && !name.startsWith('_pad')) {
      console.log(`[FAIL] WGSL declares "${name}" but the JS packer never writes it`);
      bad++;
    }
  }
  if (bad) {
    console.log(`\nWGSL UNIFORM CONTRACT: FAIL (${bad})`);
    process.exitCode = 1;
  } else {
    console.log(`[ OK ] WGSL uniform contract: ${jsOffsets.size} members match the shader struct exactly`);
  }
}
