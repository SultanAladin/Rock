import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { createRockMaterial } from '../gpu/rockMaterial.js';
import { LITHOLOGIES, JOINT_STYLES, makeBatchParams, DEFAULT_PARAMS } from '../core/generator.js';
import { DEFAULT_WEATHERING } from '../core/weathering.js';
import { toOBJ, toPLY, download } from '../io/exporters.js';
import { buildUI } from './ui.js';

// ---------------------------------------------------------------- renderer
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1014);
scene.fog = new THREE.Fog(0x0d1014, 12, 42);

const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200);
camera.position.set(2.2, 1.5, 2.8);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.3, 0);
controls.maxPolarAngle = Math.PI * 0.495;

// ------------------------------------------------------------------ ground
const groundGeo = new THREE.PlaneGeometry(80, 80, 1, 1);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2620, roughness: 0.97, metalness: 0 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const sun = new THREE.DirectionalLight(0xfff2dd, 2.4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -6; sun.shadow.camera.right = 6;
sun.shadow.camera.top = 6; sun.shadow.camera.bottom = -6;
sun.shadow.camera.far = 40;
sun.shadow.bias = -0.0008;
scene.add(sun);
scene.add(new THREE.HemisphereLight(0x5c7899, 0x3a3026, 0.7));

// --------------------------------------------------------------- app state
const state = {
  params: structuredClone(DEFAULT_PARAMS),
  // Batch size. Overridable per-URL (?count=N) so a stale cached bundle can be
  // ruled out from the address bar without touching the code.
  batch: Math.max(1, parseInt(new URLSearchParams(location.search).get('count') || '1', 10) || 1),
  progressive: true,
  spread: { sizeSigma: 0.35, weatherSigma: 0.45, styleMix: null, lithoMix: null },
  shading: {
    weatherAge: 0.6, lichen: 0.25, caseHardening: 0.4, dust: 0.3,
    wetness: 0.0, microRelief: 1.0, retreatScale: 8.0,
  },
  env: { sunAzimuth: 42, sunElevation: 38, exposure: 1.0 },
  debugMode: 0,
  wireframe: false,
  autoRotate: false,
  rocks: [],
  selected: 0,
};

const group = new THREE.Group();
scene.add(group);

// ------------------------------------------------------------ worker pool
const HW = Math.max(2, Math.min(8, navigator.hardwareConcurrency || 4));
class Pool {
  constructor(n) {
    this.workers = [];
    this.free = [];
    for (let i = 0; i < n; i++) {
      const w = new Worker(new URL('../worker/rockWorker.js', import.meta.url), { type: 'module' });
      this.workers.push(w); this.free.push(w);
    }
    this.queue = [];
    this.jobs = new Map();
    this.nextId = 1;
    for (const w of this.workers) {
      w.onmessage = (e) => {
        const { id, type } = e.data;
        const job = this.jobs.get(id);
        if (!job) return;
        if (type === 'progress') { job.onProgress && job.onProgress(e.data.f, e.data.label); return; }
        this.jobs.delete(id);
        this.free.push(w);
        if (type === 'error') job.reject(new Error(e.data.message));
        else job.resolve(e.data);
        this._pump();
      };
    }
  }
  run(params, onProgress) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.queue.push({ id, params, resolve, reject, onProgress });
      this._pump();
    });
  }
  _pump() {
    while (this.queue.length && this.free.length) {
      const w = this.free.pop();
      const job = this.queue.shift();
      this.jobs.set(job.id, job);
      w.postMessage({ id: job.id, params: job.params });
    }
  }
}
const pool = new Pool(HW);

// ---------------------------------------------------------------- build
function buildMesh(data) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(data.normals, 3));
  geo.setAttribute('aRetreat', new THREE.BufferAttribute(data.aRetreat, 1));
  geo.setAttribute('aShelter', new THREE.BufferAttribute(data.aShelter, 1));
  geo.setAttribute('aCurvature', new THREE.BufferAttribute(data.aCurv, 1));
  geo.setIndex(new THREE.BufferAttribute(data.indices, 1));
  geo.computeBoundingSphere();
  geo.computeBoundingBox();

  const litho = LITHOLOGIES[data.lithoKey];
  const mat = createRockMaterial(litho, data.params.seed, state.shading);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData = data;
  return mesh;
}

function layout() {
  // Pack the batch on a loose grid, seated so each boulder rests on the ground
  // at roughly its buried fraction - a boulder floating in space reads as a
  // prop; one bedded in the soil reads as a boulder.
  const n = state.rocks.length;
  if (!n) return;
  const cols = Math.ceil(Math.sqrt(n));
  let cell = 0;
  for (const m of state.rocks) {
    const bb = m.geometry.boundingBox;
    cell = Math.max(cell, Math.max(bb.max.x - bb.min.x, bb.max.z - bb.min.z));
  }
  cell *= 1.45;
  let i = 0;
  for (const m of state.rocks) {
    const r = Math.floor(i / cols), c = i % cols;
    const bb = m.geometry.boundingBox;
    const bury = (m.userData.params.weathering?.buriedFraction ?? 0.2);
    m.position.set(
      (c - (cols - 1) / 2) * cell,
      -bb.min.y - (bb.max.y - bb.min.y) * bury * 0.5,
      (r - (Math.ceil(n / cols) - 1) / 2) * cell,
    );
    i++;
  }
  // frame the batch
  const box = new THREE.Box3();
  for (const m of state.rocks) box.expandByObject(m);
  const c = box.getCenter(new THREE.Vector3());
  const s = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(s * 0.55, s * 0.42, s * 0.68));
  const shadowExtent = Math.max(3, s * 0.75);
  sun.shadow.camera.left = -shadowExtent; sun.shadow.camera.right = shadowExtent;
  sun.shadow.camera.top = shadowExtent; sun.shadow.camera.bottom = -shadowExtent;
  sun.shadow.camera.updateProjectionMatrix();
  sun.target.position.copy(c);
  sun.target.updateMatrixWorld();
  scene.add(sun.target);
}

let generating = false;
let generation = 0;

/**
 * Two-pass progressive generation.
 *
 * The level-set solve is genuinely expensive (seconds per boulder), so the
 * question is not "how do we make it instant" but "how soon does the user see
 * something real". Two changes do that:
 *
 *  1. DRAFT PASS. Every boulder is solved first at a coarse grid, which is
 *     ~8x cheaper (cost scales roughly as N^4: N^3 cells x O(N) steps for the
 *     same physical retreat). The draft is the same physics, not a placeholder,
 *     so the silhouette and weathering are already correct -- it is then
 *     replaced in place by the full-resolution solve.
 *  2. STREAMING. Each result is mounted the moment it arrives instead of
 *     awaiting the whole batch. Previously a single Promise.all meant nothing
 *     appeared until the slowest worker finished, which reads as a hang.
 */
async function generateBatch() {
  if (generating) return;
  generating = true;
  const myGen = ++generation;
  ui.setBusy(true);
  const list = makeBatchParams(state.params, state.batch, state.spread);
  const t0 = performance.now();

  for (const m of state.rocks) { group.remove(m); m.geometry.dispose(); m.material.dispose(); }
  state.rocks = [];
  const slots = new Array(list.length).fill(null);

  const fullRes = state.params.resolution;
  const draftRes = Math.max(28, Math.round(fullRes * 0.55 / 4) * 4);
  const useDraft = state.progressive && fullRes > 40;

  const done = new Array(list.length).fill(0);
  const bump = (i, f) => {
    done[i] = f;
    ui.setProgress(done.reduce((a, b) => a + b, 0) / list.length);
  };

  const mount = (i, r) => {
    if (myGen !== generation || !r) return;
    const m = buildMesh(r);
    if (slots[i]) { group.remove(slots[i]); slots[i].geometry.dispose(); slots[i].material.dispose(); }
    slots[i] = m;
    group.add(m);
    state.rocks = slots.filter(Boolean);
    layout();
    applyShading();
    ui.setStats(state.rocks, performance.now() - t0);
  };

  const jobs = list.map(async (p, i) => {
    try {
      if (useDraft) {
        const d = await pool.run({ ...p, resolution: draftRes, smoothing: 1 },
                                 (f) => bump(i, f * 0.35));
        mount(i, d);
      }
      const full = await pool.run({ ...p, resolution: fullRes },
                                  (f) => bump(i, (useDraft ? 0.35 : 0) + f * (useDraft ? 0.65 : 1)));
      mount(i, full);
    } catch (e) { console.error(e); }
  });

  await Promise.all(jobs);
  if (myGen !== generation) return;
  ui.setBusy(false);
  ui.setProgress(0);
  generating = false;
}

function applyShading() {
  for (const m of state.rocks) {
    const u = m.material.uniforms;
    u.uWeatherAge.value = state.shading.weatherAge;
    u.uLichen.value = state.shading.lichen;
    u.uCaseHardening.value = state.shading.caseHardening;
    u.uDust.value = state.shading.dust;
    u.uWetness.value = state.shading.wetness;
    u.uMicroRelief.value = state.shading.microRelief;
    u.uRetreatScale.value = state.shading.retreatScale;
    u.uDebugMode.value = state.debugMode;
    u.uExposure.value = state.env.exposure;
    m.material.wireframe = state.wireframe;
  }
  updateSun();
}

function updateSun() {
  const az = (state.env.sunAzimuth * Math.PI) / 180;
  const el = (state.env.sunElevation * Math.PI) / 180;
  const d = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
  sun.position.copy(d).multiplyScalar(14);
  const warm = Math.max(0, Math.min(1, 1 - Math.sin(el)));
  sun.color.setRGB(1, 0.96 - 0.14 * warm, 0.88 - 0.30 * warm);
  const skyBase = new THREE.Color().setRGB(0.28, 0.38, 0.55).multiplyScalar(0.55 + 0.55 * Math.sin(el));
  for (const m of state.rocks) {
    m.material.uniforms.uSunDir.value.copy(d);
    m.material.uniforms.uSunColor.value.copy(sun.color).multiplyScalar(3.2 * (0.35 + 0.85 * Math.sin(el)));
    m.material.uniforms.uSkyColor.value.copy(skyBase);
  }
}

// ------------------------------------------------------------------ export
function exportSelected(fmt) {
  const m = state.rocks[Math.min(state.selected, state.rocks.length - 1)];
  if (!m) return;
  const d = m.userData;
  const rec = { mesh: { positions: d.positions, normals: d.normals, indices: d.indices },
                aRetreat: d.aRetreat, aShelter: d.aShelter, aCurv: d.aCurv, stats: d.stats };
  const name = `boulder_${d.params.lithology}_s${d.params.seed}`;
  if (fmt === 'obj') download(toOBJ(rec, name), `${name}.obj`, 'text/plain');
  else download(toPLY(rec), `${name}.ply`);
}

function exportAll(fmt) {
  state.rocks.forEach((m, i) => {
    setTimeout(() => { state.selected = i; exportSelected(fmt); }, i * 250);
  });
}

// ---------------------------------------------------------------------- UI
const ui = buildUI(document.getElementById('panel'), state, {
  LITHOLOGIES, JOINT_STYLES, DEFAULT_WEATHERING,
  BUILD: __BUILD__,
  onGenerate: generateBatch,
  onShading: applyShading,
  onEnv: () => { updateSun(); applyShading(); },
  onExport: exportSelected,
  onExportAll: exportAll,
});

// ---------------------------------------------------------------- run loop
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * renderer.getPixelRatio() || canvas.height !== h * renderer.getPixelRatio()) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}
function tick() {
  requestAnimationFrame(tick);
  resize();
  if (state.autoRotate) group.rotation.y += 0.0025;
  controls.update();
  for (const m of state.rocks) m.material.uniforms.uCameraPos.value.copy(camera.position);
  renderer.render(scene, camera);
}
tick();
updateSun();
generateBatch();
