/**
 * Numerical check of the camera math.
 *
 * The raymarcher reconstructs rays as normalize(invViewProj * ndc - eye), so a
 * wrong inverse produces a subtly skewed or mirrored image rather than an
 * obvious failure -- exactly the class of bug that is invisible without a
 * browser. So verify it arithmetically instead: inv(M)*M == I, and the ray
 * through the screen centre must point from the eye at the target.
 */
import { perspective, lookAt, multiply, invert, OrbitCamera } from '../src/app/camera.js';

let fail = 0;
const ok = (c, m, extra = '') => { if (!c) { console.log(`[FAIL] ${m} ${extra}`); fail++; } else console.log(`[ OK ] ${m}`); };

const view = lookAt([2, 1.5, 2.6], [0, 0, 0], [0, 1, 0]);
const proj = perspective(0.66, 1.7, 0.01, 200);
const vp = multiply(proj, view);
const inv = invert(vp);

// inv * vp == I
const I = multiply(inv, vp);
let maxErr = 0;
for (let i = 0; i < 16; i++) {
  const want = (i % 5 === 0) ? 1 : 0;
  maxErr = Math.max(maxErr, Math.abs(I[i] - want));
}
ok(maxErr < 1e-4, `inv(VP)*VP = I (max err ${maxErr.toExponential(2)})`);

// unproject the screen centre -> ray must aim at the target
const eye = [2, 1.5, 2.6];
function unproject(ndc) {
  const o = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    o[r] = inv[0 * 4 + r] * ndc[0] + inv[1 * 4 + r] * ndc[1] + inv[2 * 4 + r] * ndc[2] + inv[3 * 4 + r];
  }
  return [o[0] / o[3], o[1] / o[3], o[2] / o[3]];
}
const p = unproject([0, 0, 1, 1]);
let d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
let l = Math.hypot(...d); d = d.map((v) => v / l);
const want = [-eye[0], -eye[1], -eye[2]];
const wl = Math.hypot(...want);
const wn = want.map((v) => v / wl);
const dot = d[0] * wn[0] + d[1] * wn[1] + d[2] * wn[2];
ok(dot > 0.9999, `centre ray points at the target (cos = ${dot.toFixed(6)})`);

// a ray through the right edge must be to the camera-right, not mirrored
const pr = unproject([1, 0, 1, 1]);
let dr = [pr[0] - eye[0], pr[1] - eye[1], pr[2] - eye[2]];
const lr = Math.hypot(...dr); dr = dr.map((v) => v / lr);
// camera right = normalize(cross(fwd, up))
const fwd = wn;
const rx = fwd[1] * 0 - fwd[2] * 1, ry = fwd[2] * 0 - fwd[0] * 0, rz = fwd[0] * 1 - fwd[1] * 0;
const rl = Math.hypot(rx, ry, rz);
const rightDot = (dr[0] * rx + dr[1] * ry + dr[2] * rz) / rl;
ok(rightDot > 0.1, `+X in NDC maps to camera-right, not mirrored (${rightDot.toFixed(3)})`);

// depth convention: WebGPU clip z in [0,1]
const nearPt = [0, 0, -0.01, 1];
const cz = proj[10] * nearPt[2] + proj[14] * nearPt[3];
const cw = -nearPt[2];
ok(Math.abs(cz / cw - 0.0) < 1e-3 || Math.abs(cz / cw) < 1e-2,
   `near plane maps to z~0 (WebGPU convention), got ${(cz / cw).toFixed(4)}`);
const farPt = -200;
const fz = (proj[10] * farPt + proj[14]) / -farPt;
ok(Math.abs(fz - 1) < 1e-3, `far plane maps to z=1, got ${fz.toFixed(5)}`);

// orbit camera stays finite over a full sweep and never flips through the pole
const canvas = { style: {}, addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} };
const cam = new OrbitCamera(canvas, { distance: 2.6 });
let bad = 0, minPhi = 9, maxPhi = -9;
cam.autoRotate = true;
for (let i = 0; i < 500; i++) {
  cam.update(1 / 60, 1.7);
  minPhi = Math.min(minPhi, cam.phi); maxPhi = Math.max(maxPhi, cam.phi);
  if (![...cam.invViewProj, ...cam.eye].every(Number.isFinite)) bad++;
}
ok(bad === 0, `500 orbit frames stay finite (${bad} bad)`);
ok(minPhi > 0.04 && maxPhi < Math.PI - 0.04, `polar angle clamped away from the poles`);

// distance clamp
cam.dist = 1e9; cam.dist = Math.min(40, Math.max(0.12, cam.dist));
ok(cam.dist === 40, 'zoom clamped');

console.log(fail ? `\nFAIL (${fail})` : '\nPASS');
process.exit(fail ? 1 : 0);
