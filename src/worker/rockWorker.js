/**
 * Generation worker. The level-set solve is heavy (O(steps * N^3)) and must
 * never touch the render thread. Results are transferred, not copied.
 */
import { generateRock } from '../core/generator.js';

self.onmessage = (e) => {
  const { id, params } = e.data;
  try {
    const r = generateRock(params, (f, label) => {
      self.postMessage({ id, type: 'progress', f, label });
    });
    const payload = {
      id, type: 'done',
      positions: r.mesh.positions,
      normals: r.mesh.normals,
      indices: r.mesh.indices,
      aRetreat: r.aRetreat,
      aShelter: r.aShelter,
      aCurv: r.aCurv,
      lithoKey: r.lithoKey,
      stats: r.stats,
      params: { seed: r.params.seed, size: r.params.size, jointStyle: r.params.jointStyle, lithology: r.params.lithology },
    };
    self.postMessage(payload, [
      payload.positions.buffer, payload.normals.buffer, payload.indices.buffer,
      payload.aRetreat.buffer, payload.aShelter.buffer, payload.aCurv.buffer,
    ]);
  } catch (err) {
    self.postMessage({ id, type: 'error', message: String(err && err.stack || err) });
  }
};
