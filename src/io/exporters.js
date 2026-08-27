/**
 * Mesh export. OBJ for interchange, PLY (binary) when you want the baked
 * weathering attributes to survive the trip into Houdini/Blender - retreat,
 * shelter and curvature ride along as vertex colours so the material can be
 * rebuilt downstream without re-running the solver.
 */

export function toOBJ(rec, name = 'rock') {
  const { positions, normals, indices } = rec.mesh || rec;
  const out = [];
  out.push(`# ${name} - procedural granite boulder`);
  out.push(`# vertices ${positions.length / 3}  triangles ${indices.length / 3}`);
  if (rec.stats) {
    out.push(`# volume ${rec.stats.volume.toFixed(6)} m^3  mass ${rec.stats.massKg.toFixed(1)} kg  sphericity ${rec.stats.sphericity.toFixed(3)}`);
  }
  out.push(`o ${name}`);
  for (let i = 0; i < positions.length; i += 3) {
    out.push(`v ${positions[i].toFixed(6)} ${positions[i + 1].toFixed(6)} ${positions[i + 2].toFixed(6)}`);
  }
  for (let i = 0; i < normals.length; i += 3) {
    out.push(`vn ${normals[i].toFixed(5)} ${normals[i + 1].toFixed(5)} ${normals[i + 2].toFixed(5)}`);
  }
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] + 1, b = indices[i + 1] + 1, c = indices[i + 2] + 1;
    out.push(`f ${a}//${a} ${b}//${b} ${c}//${c}`);
  }
  return out.join('\n');
}

/** Binary little-endian PLY with normals + attribute-encoded colours. */
export function toPLY(rec) {
  const { positions, normals, indices } = rec.mesh || rec;
  const nv = positions.length / 3, nf = indices.length / 3;
  const retreat = rec.aRetreat, shelter = rec.aShelter, curv = rec.aCurv;
  const maxR = retreat ? Math.max(1e-6, ...retreat) : 1;

  const header =
`ply
format binary_little_endian 1.0
comment procedural granite boulder - vertex colour encodes weathering
comment red = normalised surface retreat, green = shelter (AO), blue = curvature
element vertex ${nv}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face ${nf}
property list uchar uint vertex_indices
end_header
`;
  const headerBytes = new TextEncoder().encode(header);
  const vertBytes = nv * (6 * 4 + 3);
  const faceBytes = nf * (1 + 3 * 4);
  const buf = new ArrayBuffer(headerBytes.length + vertBytes + faceBytes);
  const u8 = new Uint8Array(buf);
  u8.set(headerBytes, 0);
  const dv = new DataView(buf);
  let o = headerBytes.length;
  for (let v = 0; v < nv; v++) {
    dv.setFloat32(o, positions[v * 3], true); o += 4;
    dv.setFloat32(o, positions[v * 3 + 1], true); o += 4;
    dv.setFloat32(o, positions[v * 3 + 2], true); o += 4;
    dv.setFloat32(o, normals[v * 3], true); o += 4;
    dv.setFloat32(o, normals[v * 3 + 1], true); o += 4;
    dv.setFloat32(o, normals[v * 3 + 2], true); o += 4;
    const r = retreat ? Math.min(255, Math.max(0, (retreat[v] / maxR) * 255)) : 128;
    const g = shelter ? Math.min(255, Math.max(0, shelter[v] * 255)) : 128;
    const b = curv ? Math.min(255, Math.max(0, (curv[v] * 0.5 + 0.5) * 255)) : 128;
    dv.setUint8(o++, r | 0); dv.setUint8(o++, g | 0); dv.setUint8(o++, b | 0);
  }
  for (let f = 0; f < nf; f++) {
    dv.setUint8(o++, 3);
    dv.setUint32(o, indices[f * 3], true); o += 4;
    dv.setUint32(o, indices[f * 3 + 1], true); o += 4;
    dv.setUint32(o, indices[f * 3 + 2], true); o += 4;
  }
  return buf;
}

export function download(data, filename, mime = 'application/octet-stream') {
  const blob = data instanceof ArrayBuffer ? new Blob([data], { type: mime }) : new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 500);
}
