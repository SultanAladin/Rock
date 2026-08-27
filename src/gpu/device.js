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
