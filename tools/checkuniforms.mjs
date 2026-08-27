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
