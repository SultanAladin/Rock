/**
 * Execute the INIT compute shader on the CPU with wgsl_reflect's interpreter,
 * using the exact same uniform bytes the driver uploads, and inspect the field
 * it writes. This is the closest thing to a real GPU available here.
 */
import { WgslExec, WgslParser } from 'wgsl_reflect/wgsl_reflect.module.js';
import { INIT_WGSL } from '../src/gpu/wgsl/erode.wgsl.js';
import { ErosionEngine } from '../src/gpu/erosionEngine.js';

// --- reuse the driver's packer to get byte-identical uniforms --------------
globalThis.GPUBufferUsage = { MAP_READ:1, COPY_SRC:4, COPY_DST:8, UNIFORM:64, STORAGE:128 };
globalThis.GPUShaderStage = { VERTEX:1, FRAGMENT:2, COMPUTE:4 };
let paramsBytes = null, faceBytes = null, faceCount = 0, bufId = 0;
const writes = new Map();   // buffer id -> latest bytes
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

const N = 16;   // small grid: the interpreter is slow
const engine = new ErosionEngine(dev, null, 'bgra8unorm');
engine.reset({
  seed: 3, lithology: 'biotite-granite', jointStyle: 'orthogonal',
  resolution: N, size: 1.0, aspectVariation: 0.28, jointRoughness: 1.0,
  hurst: 0.8, sheetingCurvature: 0.0, weathering: { years: 0.7 },
});

// identify the buffers by what the engine recorded
paramsBytes = writes.get(engine.buf.params._id);
faceBytes   = writes.get(engine.buf.seedA._id);   // faces now ride in the seed buffer
faceCount   = engine.faceCount;
console.log(`params ${paramsBytes.length} B, faces ${faceCount}, faceBytes ${faceBytes.length}`);
const pf = new Float32Array(paramsBytes.slice().buffer);
const pu = new Uint32Array(paramsBytes.slice().buffer);
console.log(`  n=${pu[0]} seed=${pu[1]} h=${pf[4].toFixed(5)} origin=${pf[5].toFixed(4)} extent=${pf[6].toFixed(4)}`);

// --- run the shader --------------------------------------------------------
const N3 = N*N*N;
const phiOut = new Float32Array(N3);
const auxOut = new Float32Array(N3);
const bindGroups = {
  0: {
    0: new Uint8Array(paramsBytes),          // Params uniform
    1: new Float32Array(N3),                 // phiIn
    2: phiOut,                               // phiOut
    3: new Float32Array(new Uint8Array(faceBytes).buffer),   // seedIn = faces
    4: new Float32Array(N3*4),               // seedOut
    5: new Float32Array(N3),                 // aux
    6: auxOut,                               // auxOut
    7: new Uint32Array(4),                   // counter
    8: new Float32Array(N3),                 // phi0In
  },
};

const exec = new WgslExec(new WgslParser().parse(INIT_WGSL));
const wg = N/4;
console.log(`dispatching ${wg}^3 workgroups over ${N}^3 = ${N3} cells ...`);
const t0 = Date.now();
exec.dispatchWorkgroups('main', [wg, wg, wg], bindGroups);
console.log(`  took ${((Date.now()-t0)/1000).toFixed(1)}s`);

const phi = phiOut;
let neg = 0, pos = 0, nan = 0, mn = Infinity, mx = -Infinity;
for (const v of phi) {
  if (!Number.isFinite(v)) { nan++; continue; }
  if (v < 0) neg++; else pos++;
  if (v < mn) mn = v; if (v > mx) mx = v;
}
console.log(`\nphi: ${neg} inside, ${pos} outside, ${nan} non-finite`);
console.log(`     range [${mn.toFixed(4)}, ${mx.toFixed(4)}]`);
console.log(neg > 0 && nan === 0 ? '\nINIT PRODUCES A SOLID' : '\nINIT PRODUCED NOTHING <-- this is the bug');
