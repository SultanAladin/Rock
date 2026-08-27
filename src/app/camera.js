/**
 * Minimal orbit camera + mat4 math.
 *
 * The raymarcher only needs the inverse view-projection and the eye position,
 * so pulling in a scene graph for this would be dead weight. Column-major,
 * same convention as GLSL/WGSL.
 */

export function mat4Identity(o = new Float32Array(16)) {
  o.fill(0); o[0] = o[5] = o[10] = o[15] = 1; return o;
}

export function perspective(fovy, aspect, near, far, o = new Float32Array(16)) {
  const f = 1 / Math.tan(fovy / 2);
  o.fill(0);
  o[0] = f / aspect; o[5] = f; o[11] = -1;
  // WebGPU clip space is z in [0,1], not [-1,1]. Getting this wrong does not
  // break the raymarcher (it only uses the ray direction) but it would break
  // any future depth testing, so use the correct convention up front.
  o[10] = far / (near - far);
  o[14] = (far * near) / (near - far);
  return o;
}

export function lookAt(eye, center, up, o = new Float32Array(16)) {
  let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
  let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  o[0] = xx; o[1] = yx; o[2] = zx; o[3] = 0;
  o[4] = xy; o[5] = yy; o[6] = zy; o[7] = 0;
  o[8] = xz; o[9] = yz; o[10] = zz; o[11] = 0;
  o[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  o[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  o[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  o[15] = 1;
  return o;
}

export function multiply(a, b, o = new Float32Array(16)) {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

export function invert(m, o = new Float32Array(16)) {
  const a = m;
  const b00 = a[0] * a[5] - a[1] * a[4], b01 = a[0] * a[6] - a[2] * a[4];
  const b02 = a[0] * a[7] - a[3] * a[4], b03 = a[1] * a[6] - a[2] * a[5];
  const b04 = a[1] * a[7] - a[3] * a[5], b05 = a[2] * a[7] - a[3] * a[6];
  const b06 = a[8] * a[13] - a[9] * a[12], b07 = a[8] * a[14] - a[10] * a[12];
  const b08 = a[8] * a[15] - a[11] * a[12], b09 = a[9] * a[14] - a[10] * a[13];
  const b10 = a[9] * a[15] - a[11] * a[13], b11 = a[10] * a[15] - a[11] * a[14];
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return mat4Identity(o);
  det = 1 / det;
  o[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * det;
  o[1] = (a[2] * b10 - a[1] * b11 - a[3] * b09) * det;
  o[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * det;
  o[3] = (a[10] * b04 - a[9] * b05 - a[11] * b03) * det;
  o[4] = (a[6] * b08 - a[4] * b11 - a[7] * b07) * det;
  o[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * det;
  o[6] = (a[14] * b02 - a[12] * b05 - a[15] * b01) * det;
  o[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * det;
  o[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * det;
  o[9] = (a[1] * b08 - a[0] * b10 - a[3] * b06) * det;
  o[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * det;
  o[11] = (a[9] * b02 - a[8] * b04 - a[11] * b00) * det;
  o[12] = (a[5] * b07 - a[4] * b09 - a[6] * b06) * det;
  o[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * det;
  o[14] = (a[13] * b01 - a[12] * b03 - a[14] * b00) * det;
  o[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * det;
  return o;
}

/** Spherical orbit controls with damping. */
export class OrbitCamera {
  constructor(canvas, { distance = 2.6, target = [0, 0, 0] } = {}) {
    this.canvas = canvas;
    this.target = target.slice();
    this.dist = distance;
    this.theta = 0.9;      // azimuth
    this.phi = 1.15;       // polar
    this._vTheta = 0; this._vPhi = 0;
    this.fov = (38 * Math.PI) / 180;
    this.autoRotate = false;
    this.dirty = true;

    this.eye = [0, 0, 0];
    this.view = new Float32Array(16);
    this.proj = new Float32Array(16);
    this.viewProj = new Float32Array(16);
    this.invViewProj = new Float32Array(16);

    let dragging = 0, lx = 0, ly = 0;
    canvas.style.touchAction = 'none';
    canvas.addEventListener('pointerdown', (e) => {
      dragging = e.button === 2 || e.shiftKey ? 2 : 1;
      lx = e.clientX; ly = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', (e) => {
      dragging = 0;
      canvas.releasePointerCapture?.(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (dragging === 1) {
        this._vTheta -= dx * 0.005;
        this._vPhi -= dy * 0.005;
      } else {
        // pan in the camera plane
        const s = this.dist * 0.0016;
        const r = this._basis();
        for (let i = 0; i < 3; i++) this.target[i] += -r.right[i] * dx * s + r.up[i] * dy * s;
        this.dirty = true;
      }
    });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.dist *= Math.exp(e.deltaY * 0.0012);
      this.dist = Math.min(40, Math.max(0.12, this.dist));
      this.dirty = true;
    }, { passive: false });
  }

  _basis() {
    const sp = Math.sin(this.phi), cp = Math.cos(this.phi);
    const st = Math.sin(this.theta), ct = Math.cos(this.theta);
    const fwd = [-sp * st, -cp, -sp * ct];
    const right = [ct, 0, -st];
    const up = [
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0],
    ];
    return { fwd, right, up };
  }

  /** @returns {boolean} whether anything moved this frame */
  update(dt, aspect) {
    let moved = this.dirty;
    if (this.autoRotate) { this.theta += dt * 0.25; moved = true; }
    if (Math.abs(this._vTheta) > 1e-5 || Math.abs(this._vPhi) > 1e-5) {
      this.theta += this._vTheta;
      this.phi = Math.min(Math.PI - 0.05, Math.max(0.05, this.phi + this._vPhi));
      this._vTheta *= 0.82; this._vPhi *= 0.82;
      moved = true;
    }
    if (!moved && this._lastAspect === aspect) return false;
    this._lastAspect = aspect;
    this.dirty = false;

    const sp = Math.sin(this.phi), cp = Math.cos(this.phi);
    this.eye = [
      this.target[0] + this.dist * sp * Math.sin(this.theta),
      this.target[1] + this.dist * cp,
      this.target[2] + this.dist * sp * Math.cos(this.theta),
    ];
    lookAt(this.eye, this.target, [0, 1, 0], this.view);
    perspective(this.fov, aspect, 0.01, 200, this.proj);
    multiply(this.proj, this.view, this.viewProj);
    invert(this.viewProj, this.invViewProj);
    return true;
  }
}
