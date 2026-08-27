/**
 * Naive Surface Nets with QEF-based vertex placement (dual contouring).
 *
 * Marching Cubes was rejected deliberately: it cannot represent the sharp
 * arrises of a freshly jointed block, and it produces the sliver triangles and
 * staircase normals that make procedural rocks read as "CG". Dual contouring
 * places one vertex per active cell at the minimiser of the Hermite quadratic
 * error function built from the edge crossings and their gradients, which
 * reproduces sharp joint edges *and* smoothly rounded weathered surfaces from
 * the same field, with quad-dominant topology.
 *
 * The QEF is solved by regularised normal equations (A^T A + eps I) x = A^T b,
 * clamped to the cell, which is stable without needing an SVD.
 */

const EDGES = [
  [0, 1], [1, 3], [2, 3], [0, 2],
  [4, 5], [5, 7], [6, 7], [4, 6],
  [0, 4], [1, 5], [2, 6], [3, 7],
];
const CORNERS = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];

function solve3x3(A, b) {
  // Gaussian elimination with partial pivoting on a symmetric 3x3.
  const m = [
    [A[0], A[1], A[2], b[0]],
    [A[1], A[3], A[4], b[1]],
    [A[2], A[4], A[5], b[2]],
  ];
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) piv = r;
    if (Math.abs(m[piv][c]) < 1e-12) return null;
    const t = m[c]; m[c] = m[piv]; m[piv] = t;
    for (let r = 0; r < 3; r++) {
      if (r === c) continue;
      const f = m[r][c] / m[c][c];
      for (let k2 = c; k2 < 4; k2++) m[r][k2] -= f * m[c][k2];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

/**
 * @param {Field3} field
 * @param {object} opts { sharpness } 0 = fully smooth (mass-point), 1 = full QEF
 * @returns {{positions:Float32Array, normals:Float32Array, indices:Uint32Array}}
 */
export function dualContour(field, { sharpness = 1.0, clampBias = 0.06 } = {}) {
  const n = field.n, h = field.h, d = field.data;
  const idx = (i, j, k) => (k * n + j) * n + i;
  const vertexIndex = new Int32Array((n - 1) * (n - 1) * (n - 1)).fill(-1);
  const cidx = (i, j, k) => (k * (n - 1) + j) * (n - 1) + i;

  const positions = [];
  const normals = [];
  const grad = [0, 0, 0];

  // --- 1. one vertex per sign-changing cell ------------------------------
  for (let k = 0; k < n - 1; k++) {
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const s = new Array(8);
        let neg = 0;
        for (let c = 0; c < 8; c++) {
          const [dx, dy, dz] = CORNERS[c];
          s[c] = d[idx(i + dx, j + dy, k + dz)];
          if (s[c] < 0) neg++;
        }
        if (neg === 0 || neg === 8) continue;

        // Hermite data on crossing edges
        const pts = [], nrms = [];
        for (const [a, b] of EDGES) {
          const sa = s[a], sb = s[b];
          if ((sa < 0) === (sb < 0)) continue;
          const t = sa / (sa - sb);
          const ca = CORNERS[a], cb = CORNERS[b];
          const px = (ca[0] + (cb[0] - ca[0]) * t);
          const py = (ca[1] + (cb[1] - ca[1]) * t);
          const pz = (ca[2] + (cb[2] - ca[2]) * t);
          const wx = field.coord(i) + px * h;
          const wy = field.coord(j) + py * h;
          const wz = field.coord(k) + pz * h;
          field.gradient(wx, wy, wz, h * 0.6, grad);
          const gl = Math.hypot(grad[0], grad[1], grad[2]) || 1;
          pts.push([px, py, pz]);
          nrms.push([grad[0] / gl, grad[1] / gl, grad[2] / gl]);
        }
        if (!pts.length) continue;

        // mass point (centroid of crossings) - the smooth fallback
        let mx = 0, my = 0, mz = 0;
        for (const p of pts) { mx += p[0]; my += p[1]; mz += p[2]; }
        mx /= pts.length; my /= pts.length; mz /= pts.length;

        let vx = mx, vy = my, vz = mz;
        if (sharpness > 0) {
          // Build A^T A and A^T b relative to the mass point (better conditioned)
          let a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
          let b0 = 0, b1 = 0, b2 = 0;
          for (let e = 0; e < pts.length; e++) {
            const nn = nrms[e], p = pts[e];
            const dpn = nn[0] * (p[0] - mx) + nn[1] * (p[1] - my) + nn[2] * (p[2] - mz);
            a00 += nn[0] * nn[0]; a01 += nn[0] * nn[1]; a02 += nn[0] * nn[2];
            a11 += nn[1] * nn[1]; a12 += nn[1] * nn[2]; a22 += nn[2] * nn[2];
            b0 += nn[0] * dpn; b1 += nn[1] * dpn; b2 += nn[2] * dpn;
          }
          const eps = 0.06 / Math.max(0.02, sharpness);
          const sol = solve3x3([a00 + eps, a01, a02, a11 + eps, a12, a22 + eps], [b0, b1, b2]);
          if (sol) {
            vx = mx + sol[0]; vy = my + sol[1]; vz = mz + sol[2];
            // clamp inside the cell (with a small bias) to stop spikes
            const lo = -clampBias, hi = 1 + clampBias;
            vx = Math.min(hi, Math.max(lo, vx));
            vy = Math.min(hi, Math.max(lo, vy));
            vz = Math.min(hi, Math.max(lo, vz));
          }
        }

        const wx = field.coord(i) + vx * h;
        const wy = field.coord(j) + vy * h;
        const wz = field.coord(k) + vz * h;
        field.gradient(wx, wy, wz, h * 0.6, grad);
        const gl = Math.hypot(grad[0], grad[1], grad[2]) || 1;

        vertexIndex[cidx(i, j, k)] = positions.length / 3;
        positions.push(wx, wy, wz);
        normals.push(grad[0] / gl, grad[1] / gl, grad[2] / gl);
      }
    }
  }

  // --- 2. quads on each sign-changing grid edge --------------------------
  const indices = [];
  /**
   * Emit the two triangles of a dual quad.
   *
   * WINDING. The convention must be counter-clockwise seen from OUTSIDE the
   * solid (phi > 0 side), which is what WebGL/OpenGL, glTF, OBJ and every DCC
   * treat as front-facing. Getting this backwards is not a cosmetic issue: the
   * renderer culls or shades back faces, lighting inverts, and the surface
   * reads as a sparkling shell instead of a rock -- and the exported OBJ/PLY is
   * inside-out for anyone downstream.
   *
   * The `flip` argument is the sign of phi at the edge's first endpoint. When
   * that endpoint is INSIDE (phi < 0) the grid edge runs inside->outside, so the
   * natural cell-ring order is already CCW from outside; when it is outside the
   * ring must be reversed. Verified against an analytic sphere: signed volume
   * must come out POSITIVE and 100% of face normals must agree with the
   * outward radial direction (tools/checkwinding.mjs).
   */
  const emitQuad = (a, b, c, e, flip) => {
    if (a < 0 || b < 0 || c < 0 || e < 0) return;
    if (flip) { indices.push(a, b, c, a, c, e); }
    else { indices.push(a, c, b, a, e, c); }
  };
  for (let k = 0; k < n - 1; k++) for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const s0 = d[idx(i, j, k)];
    // +X edge -> quad in YZ from cells (i,j-1,k-1)...(i,j,k)
    if (i < n - 1 && j > 0 && k > 0) {
      const s1 = d[idx(i + 1, j, k)];
      if ((s0 < 0) !== (s1 < 0)) {
        emitQuad(vertexIndex[cidx(i, j - 1, k - 1)], vertexIndex[cidx(i, j, k - 1)],
                 vertexIndex[cidx(i, j, k)], vertexIndex[cidx(i, j - 1, k)], s0 < 0);
      }
    }
    if (j < n - 1 && i > 0 && k > 0) {
      const s1 = d[idx(i, j + 1, k)];
      if ((s0 < 0) !== (s1 < 0)) {
        emitQuad(vertexIndex[cidx(i - 1, j, k - 1)], vertexIndex[cidx(i - 1, j, k)],
                 vertexIndex[cidx(i, j, k)], vertexIndex[cidx(i, j, k - 1)], s0 < 0);
      }
    }
    if (k < n - 1 && i > 0 && j > 0) {
      const s1 = d[idx(i, j, k + 1)];
      if ((s0 < 0) !== (s1 < 0)) {
        emitQuad(vertexIndex[cidx(i - 1, j - 1, k)], vertexIndex[cidx(i, j - 1, k)],
                 vertexIndex[cidx(i, j, k)], vertexIndex[cidx(i - 1, j, k)], s0 < 0);
      }
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
  };
}

/** Keep only the largest connected component (drops spall fragments/noise). */
export function largestComponent(mesh) {
  const nv = mesh.positions.length / 3;
  const parent = new Int32Array(nv);
  for (let i = 0; i < nv; i++) parent[i] = i;
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const uni = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
  const I = mesh.indices;
  for (let t = 0; t < I.length; t += 3) { uni(I[t], I[t + 1]); uni(I[t + 1], I[t + 2]); }
  const count = new Map();
  for (let i = 0; i < nv; i++) { const r = find(i); count.set(r, (count.get(r) || 0) + 1); }
  let bestRoot = -1, best = -1;
  for (const [r, c] of count) if (c > best) { best = c; bestRoot = r; }
  if (best === nv) return mesh;
  const remap = new Int32Array(nv).fill(-1);
  const pos = [], nrm = [];
  for (let i = 0; i < nv; i++) {
    if (find(i) !== bestRoot) continue;
    remap[i] = pos.length / 3;
    pos.push(mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]);
    nrm.push(mesh.normals[i * 3], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]);
  }
  const ind = [];
  for (let t = 0; t < I.length; t += 3) {
    const a = remap[I[t]], b = remap[I[t + 1]], c = remap[I[t + 2]];
    if (a >= 0 && b >= 0 && c >= 0) ind.push(a, b, c);
  }
  return { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint32Array(ind) };
}

/** Area-weighted vertex normals recomputed from the final geometry. */
export function recomputeNormals(mesh) {
  const P = mesh.positions, I = mesh.indices;
  const N = new Float32Array(P.length);
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
    const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    N[a] += nx; N[a + 1] += ny; N[a + 2] += nz;
    N[b] += nx; N[b + 1] += ny; N[b + 2] += nz;
    N[c] += nx; N[c + 1] += ny; N[c + 2] += nz;
  }
  for (let v = 0; v < N.length; v += 3) {
    const l = Math.hypot(N[v], N[v + 1], N[v + 2]) || 1;
    N[v] /= l; N[v + 1] /= l; N[v + 2] /= l;
  }
  mesh.normals = N;
  return mesh;
}

/**
 * Taubin lambda/mu smoothing: shrinkage-free. Used only lightly, to remove the
 * dual-grid quantisation without eating the joint arrises.
 */
export function taubinSmooth(mesh, iterations = 2, lambda = 0.5, mu = -0.53) {
  const P = mesh.positions, I = mesh.indices;
  const nv = P.length / 3;
  const adjStart = new Uint32Array(nv + 1);
  const deg = new Uint32Array(nv);
  const pairs = [];
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t], b = I[t + 1], c = I[t + 2];
    pairs.push([a, b], [b, a], [b, c], [c, b], [c, a], [a, c]);
  }
  for (const [a] of pairs) deg[a]++;
  let acc = 0;
  for (let i = 0; i < nv; i++) { adjStart[i] = acc; acc += deg[i]; }
  adjStart[nv] = acc;
  const cursor = adjStart.slice();
  const adj = new Uint32Array(acc);
  for (const [a, b] of pairs) adj[cursor[a]++] = b;

  const step = (factor, src, dst) => {
    for (let v = 0; v < nv; v++) {
      const s = adjStart[v], e = adjStart[v + 1];
      if (e === s) { dst[v * 3] = src[v * 3]; dst[v * 3 + 1] = src[v * 3 + 1]; dst[v * 3 + 2] = src[v * 3 + 2]; continue; }
      let ax = 0, ay = 0, az = 0;
      for (let q = s; q < e; q++) { const w = adj[q] * 3; ax += src[w]; ay += src[w + 1]; az += src[w + 2]; }
      const c = e - s;
      dst[v * 3] = src[v * 3] + factor * (ax / c - src[v * 3]);
      dst[v * 3 + 1] = src[v * 3 + 1] + factor * (ay / c - src[v * 3 + 1]);
      dst[v * 3 + 2] = src[v * 3 + 2] + factor * (az / c - src[v * 3 + 2]);
    }
  };
  let a = P, b = new Float32Array(P.length);
  for (let it = 0; it < iterations; it++) {
    step(lambda, a, b); const t1 = a; a = b; b = t1;
    step(mu, a, b); const t2 = a; a = b; b = t2;
  }
  if (a !== mesh.positions) mesh.positions.set(a);
  return mesh;
}
