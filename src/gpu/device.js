/**
 * WebGPU device acquisition + a small amount of capability reporting.
 *
 * Kept separate so the app can degrade honestly: if WebGPU is missing we say
 * so and fall back to the CPU/WebGL path rather than showing a blank canvas.
 */

export async function requestRockDevice() {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU not available. Use Chrome/Edge 113+, Safari 18+, or Firefox 141+.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter. GPU may be blocklisted; try enabling hardware acceleration.');

  // We ask for a bigger storage buffer only if the adapter offers it: a 128^3
  // f32 grid is 8 MB and the seed buffer is 4x that, which exceeds the default
  // 128 MB *total* binding limit on some adapters at high resolution.
  const lim = adapter.limits;
  const requiredLimits = {};
  const want = {
    maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 512 * 1024 * 1024),
    maxBufferSize: Math.min(lim.maxBufferSize, 512 * 1024 * 1024),
    maxComputeInvocationsPerWorkgroup: Math.min(lim.maxComputeInvocationsPerWorkgroup, 256),
    maxComputeWorkgroupSizeX: Math.min(lim.maxComputeWorkgroupSizeX, 256),
  };
  for (const [k, v] of Object.entries(want)) if (v) requiredLimits[k] = v;

  const device = await adapter.requestDevice({ requiredLimits });

  // Surface shader compile / validation errors instead of letting them vanish.
  device.addEventListener?.('uncapturederror', (e) => {
    console.error('[WebGPU] uncaptured error:', e.error?.message || e.error);
  });

  const info = {
    vendor: adapter.info?.vendor || 'unknown',
    architecture: adapter.info?.architecture || '',
    description: adapter.info?.description || '',
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxBufferSize: device.limits.maxBufferSize,
  };
  return { adapter, device, info };
}

/** Largest cubic grid that fits the device's storage-buffer limits. */
export function maxResolutionFor(device) {
  // The JFA seed buffer (vec4<f32> per cell) is the biggest single binding.
  const cap = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const cells = Math.floor(cap / 16);
  return Math.floor(Math.cbrt(cells));
}

/**
 * Prove the device can actually run a compute shader and a render pipeline
 * before we blame our own solver.
 *
 * WGSL compile errors are asynchronous and non-fatal: createShaderModule
 * resolves, createComputePipeline resolves, the dispatch quietly writes
 * nothing. A black canvas is therefore ambiguous between "the GPU path is
 * broken" and "my shader has a type error". This distinguishes them.
 *
 * @returns {Promise<{compute:boolean, render:boolean, errors:string[]}>}
 */
export async function selfTest(device, format = 'rgba8unorm') {
  const errors = [];
  let compute = false, render = false;

  // ---- compute: write a known pattern and read it back --------------------
  try {
    device.pushErrorScope('validation');
    const code = `
@group(0) @binding(0) var<storage, read_write> outb : array<f32>;
@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  outb[gid.x] = f32(gid.x) * 2.0 + 1.0;
}`;
    const module = device.createShaderModule({ code, label: 'selftest-compute' });
    const info = await module.getCompilationInfo?.();
    for (const m of info?.messages || []) {
      if (m.type === 'error') errors.push(`selftest compute: ${m.message}`);
    }
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const out = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: out } }],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(2); pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, 32);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    compute = got[0] === 1 && got[3] === 7;
    if (!compute) errors.push(`selftest compute wrote [${Array.from(got.slice(0, 4))}], expected [1,3,5,7]`);
    out.destroy(); read.destroy();
    const e = await device.popErrorScope();
    if (e) errors.push(`selftest compute validation: ${e.message}`);
  } catch (e) {
    errors.push(`selftest compute threw: ${e.message}`);
  }

  // ---- render: a pipeline with a storage buffer bound to the FRAGMENT stage
  // (the configuration the raymarcher depends on)
  try {
    device.pushErrorScope('validation');
    const code = `
@group(0) @binding(0) var<storage, read> data : array<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  return vec4<f32>(p[i], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(data[0], 0.0, 0.0, 1.0);
}`;
    const module = device.createShaderModule({ code, label: 'selftest-render' });
    const info = await module.getCompilationInfo?.();
    for (const m of info?.messages || []) {
      if (m.type === 'error') errors.push(`selftest render: ${m.message}`);
    }
    device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    });
    const e = await device.popErrorScope();
    if (e) errors.push(`selftest render validation: ${e.message}`);
    else render = true;
  } catch (e) {
    errors.push(`selftest render threw: ${e.message}`);
  }

  return { compute, render, errors };
}
