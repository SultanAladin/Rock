/** Parse the assembled GLSL with a real ES3 grammar to catch syntax errors. */
import { parser } from '@shaderfrog/glsl-parser';
import { HASH_GLSL, NOISE_GLSL, GRAIN_GLSL } from '../src/gpu/glsl/common.glsl.js';
import fs from 'fs';

const src = fs.readFileSync('src/gpu/rockMaterial.js','utf8');
const grab = (name) => {
  const i = src.indexOf(`const ${name} = /* glsl */\``);
  const s = src.indexOf('`', i)+1;
  let e = s; let depth=0;
  // find closing backtick not preceded by ${
  for (let p=s;p<src.length;p++){ if(src[p]==='`' && src[p-1] !== '\\'){ e=p; break; } }
  return src.slice(s,e);
};
let frag = grab('FRAG')
  .replace('${HASH_GLSL}',HASH_GLSL).replace('${NOISE_GLSL}',NOISE_GLSL).replace('${GRAIN_GLSL}',GRAIN_GLSL);
let vert = grab('VERT');
for (const [n,code] of [['VERT',vert],['FRAG',frag]]) {
  try { parser.parse(code); console.log(n,'OK', code.split('\n').length,'lines'); }
  catch(e){ console.log(n,'FAIL', e.message.split('\n').slice(0,6).join('\n')); }
}

// Regression guard: a stray backtick inside a /* glsl */`...` template silently
// TRUNCATES the shader (the literal closes early) and the rest of the file is
// parsed as JS. The GLSL still "parses" because a prefix of a shader is often
// valid, so this must be checked by length, not by parse success.
const MIN = { VERT: 25, FRAG: 400 };
let bad = 0;
for (const [n, code] of [['VERT', vert], ['FRAG', frag]]) {
  const lines = code.split('\n').length;
  if (lines < MIN[n]) { console.log(`${n} TRUNCATED: ${lines} lines, expected >= ${MIN[n]}`); bad = 1; }
  if (!/void main\(\)/.test(code)) { console.log(`${n} missing main()`); bad = 1; }
}
// fragment must write its output
if (!/fragColor\s*=/.test(frag)) { console.log('FRAG never writes fragColor'); bad = 1; }
console.log(bad ? 'FAIL' : 'PASS');
process.exit(bad);
