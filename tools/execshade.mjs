/**
 * Type-check and execute the raymarcher's shading code.
 *
 * WGSL compile errors are asynchronous and non-fatal in WebGPU: the pipeline is
 * created, the draw silently produces nothing, and you get a black canvas with
 * an empty console. wgsl_reflect parses but does not type-check, so the only
 * way to prove the fragment path is sound without a GPU is to RUN it.
 *
 * The interpreter only executes compute entry points, so we append a compute
 * wrapper that calls the same helpers the fragment shader uses. Any type error
 * on an executed path throws here.
 */
import { WgslExec, WgslParser } from 'wgsl_reflect/wgsl_reflect.module.js';
import { RAYMARCH_WGSL } from '../src/gpu/wgsl/raymarch.wgsl.js';
import { ErosionEngine } from '../src/gpu/erosionEngine.js';

globalThis.GPUBufferUsage = { MAP_READ:1, COPY_SRC:4, COPY_DST:8, UNIFORM:64, STORAGE:128 };
globalThis.GPUShaderStage = { VERTEX:1, FRAGMENT:2, COMPUTE:4 };
let bufId = 0; const writes = new Map();
const dev = {
  limits: { maxStorageBufferBindingSize: 1<<30, maxBufferSize: 1<<30 },
  createBuffer: (d) => ({ size: d.size, usage: d.usage, destroy(){}, _id: ++bufId }),
  createBindGroupLayout: (d) => ({ entries: d.entries }),
  createPipelineLayout: () => ({}), createShaderModule: () => ({}),
  createComputePipeline: () => ({}), createRenderPipeline: () => ({}),
  createBindGroup: () => ({}),
  queue: { writeBuffer(b, off, data, dOff=0, sz) {
      const view = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset + dOff, sz ?? (data.byteLength - dOff))
        : new Uint8Array(data, dOff, sz ?? (data.byteLength - dOff));
      let dst = writes.get(b._id);
      if (!dst) { dst = new Uint8Array(b.size); writes.set(b._id, dst); }
      dst.set(view, off);
    }, submit(){} },
  createCommandEncoder: () => ({
    beginComputePass: () => ({ setPipeline(){}, setBindGroup(){}, dispatchWorkgroups(){}, end(){} }),
    beginRenderPass: () => ({ setPipeline(){}, setBindGroup(){}, draw(){}, end(){} }),
    copyBufferToBuffer(){}, clearBuffer(){}, finish: () => ({}),
  }),
  addEventListener(){},
};

const N = 8;
const engine = new ErosionEngine(dev, null, 'bgra8unorm');
engine.reset({ seed: 3, lithology: 'biotite-granite', jointStyle: 'orthogonal',
  resolution: N, size: 1.0, weathering: { years: 0.7 } });
const params = writes.get(engine.buf.params._id);

// A sphere SDF so the marcher actually hits something.
const N3 = N*N*N;
const phi = new Float32Array(N3);
const extent = 0.95, h = (2*extent)/(N-1);
for (let k=0;k<N;k++) for (let j=0;j<N;j++) for (let i=0;i<N;i++) {
  const x=-extent+i*h, y=-extent+j*h, z=-extent+k*h;
  phi[(k*N+j)*N+i] = Math.hypot(x,y,z) - 0.55;
}

// camera: eye on +Z looking at the origin, identity-ish invViewProj
const cam = new Float32Array(28);
cam.set([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], 0);
cam[16]=0; cam[17]=0; cam[18]=2.5; cam[19]=1;      // eye
cam[20]=0.45; cam[21]=0.78; cam[22]=0.44; cam[23]=0; // sun
cam[24]=0.002; cam[25]=0; cam[26]=64; cam[27]=0;     // misc

const WRAPPER = `
@group(0) @binding(5) var<storage, read_write> probe : array<f32>;
@compute @workgroup_size(1)
fn probeMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = vec3<f32>(0.0, 0.0, 0.55);
  let n = sdfNormal(p);
  let g = sampleGrain(p);
  let mh = microHeight(p);
  let rel = perturbNormal(p, n, 0.9, 0.001);
  let li = lichenField(p);
  let col = shade(p, n, 0.001);
  let tm = acesTonemap(col);
  let hit = boxEntry(vec3<f32>(0.0,0.0,2.5), vec3<f32>(0.0,0.0,-1.0), P.extent);
  probe[0] = f32(g.id);
  probe[1] = g.boundary;
  probe[2] = mh;
  probe[3] = rel.subpixel;
  probe[4] = li.cov;
  probe[5] = col.r; probe[6] = col.g; probe[7] = col.b;
  probe[8] = tm.r;  probe[9] = tm.g;  probe[10] = tm.b;
  probe[11] = hit.x; probe[12] = hit.y;
  probe[13] = n.x; probe[14] = n.y; probe[15] = n.z;
  probe[16] = sdfAt(p);
}`;

const src = RAYMARCH_WGSL + WRAPPER;
let ast;
try {
  ast = new WgslParser().parse(src);
} catch (e) {
  console.log('PARSE ERROR: ' + e.message);
  process.exit(1);
}

const probe = new Float32Array(32);
const bg = {
  0: {
    0: new Uint8Array(params),
    1: phi,
    2: new Float32Array(N3).fill(0.3),   // shelter
    3: new Float32Array(N3).fill(0.02),  // retreat
    4: cam,
    5: probe,
  },
};

console.log('executing the fragment shader\'s helper chain ...');
try {
  new WgslExec(ast).dispatchWorkgroups('probeMain', [1,1,1], bg);
} catch (e) {
  console.log('\nRUNTIME/TYPE ERROR: ' + e.message);
  console.log(e.stack.split('\n').slice(0, 6).join('\n'));
  process.exit(1);
}

const L = ['grain.id','grain.boundary','microHeight','subpixel','lichen.cov',
           'col.r','col.g','col.b','aces.r','aces.g','aces.b',
           'box.tmin','box.tmax','n.x','n.y','n.z','sdfAt'];
L.forEach((n,i) => console.log(`  ${n.padEnd(15)} ${probe[i]}`));

const bad = L.filter((_,i) => !Number.isFinite(probe[i]));
const black = probe[8] === 0 && probe[9] === 0 && probe[10] === 0;
console.log(bad.length ? `\nNON-FINITE: ${bad}` : '\nall outputs finite');
console.log(black ? 'TONEMAPPED COLOUR IS BLACK <-- shading bug' : 'shading produces colour');
process.exit(bad.length || black ? 1 : 0);
