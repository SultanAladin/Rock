/**
 * WebGPU application shell.
 *
 * The loop is deliberately simple, and that simplicity is the whole feature:
 *
 *   every animation frame:
 *     engine.advance(stepsPerFrame)     // compute; phi stays on the GPU
 *     engine.render(view)               // raymarch phi directly
 *
 * There is no bake phase, no mesh rebuild, and nothing blocking. Because the
 * renderer draws the same buffer the solver writes, every erosion iteration is
 * on screen the frame it happens -- you watch the corners round. Polygonisation
 * only runs when you press Export.
 */

import { requestRockDevice, maxResolutionFor } from '../gpu/device.js';
import { ErosionEngine } from '../gpu/erosionEngine.js';
import { OrbitCamera } from './camera.js';
import { LITHOLOGIES, DEFAULT_PARAMS, makeBatchParams } from '../core/generator.js';
import { JOINT_STYLES } from '../core/joints.js';
import { DEFAULT_WEATHERING } from '../core/weathering.js';
import { buildUI } from './ui.js';
import { Field3, meanCurvatureField } from '../core/grid.js';
import { dualContour, largestComponent, recomputeNormals, taubinSmooth } from '../core/mesher.js';

import { toOBJ, toPLY, download } from '../io/exporters.js';

const canvas = document.getElementById('view');

// --------------------------------------------------------------- app state
const state = {
  params: structuredClone(DEFAULT_PARAMS),
  // Batch. The requirement is a batch generator, but a batch of simultaneous
  // slow bakes is exactly what made this unusable before. Here a batch is a
  // list of parameter variants; because a full solve is a few milliseconds of
  // GPU time, switching between them re-solves from scratch instantly, and
  // Export All walks the list. Default 1, per the "one boulder at a time"
  // instruction; ?count=N still overrides from the address bar.
  batch: Math.max(1, parseInt(new URLSearchParams(location.search).get('count') || '1', 10) || 1),
  spread: { sizeSigma: 0.35, weatherSigma: 0.45, styleMix: null, lithoMix: null },
  shading: {
    weatherAge: 0.6, lichen: 0.25, caseHardening: 0.4, dust: 0.3,
    wetness: 0.0, microRelief: 1.0, retreatScale: 8.0, exposure: 1.0, debugMode: 0,
  },
  env: { sunAzimuth: 42, sunElevation: 38, exposure: 1.0 },
  debugMode: 0,
  autoRotate: false,
  // live-solve controls
  playing: true,
  stepsPerFrame: 4,
  quality: 1.0,
  selected: 0,
};
state.params.resolution = 64;

function fatal(msg, detail) {
  const box = document.createElement('div');
  box.className = 'fatal';
  box.innerHTML = `<h2>${msg}</h2><p>${detail || ''}</p>`;
  document.getElementById('stage').appendChild(box);
  console.error(msg, detail);
}

// ------------------------------------------------------------------- boot
let device, context, engine, ui, cam;
let format = 'bgra8unorm';

async function boot() {
  let d;
  try {
    d = await requestRockDevice();
  } catch (e) {
    fatal('WebGPU unavailable', `${e.message}<br><br>The erosion solver and the
      renderer both run as GPU compute; there is no software fallback that would
      be honest about being realtime.`);
    return;
  }
  device = d.device;

  context = canvas.getContext('webgpu');
  if (!context) { fatal('Could not create a WebGPU canvas context'); return; }
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  engine = new ErosionEngine(device, context, format);
  cam = new OrbitCamera(canvas, { distance: 2.6, target: [0, 0, 0] });

  const maxRes = Math.min(160, maxResolutionFor(device));

  ui = buildUI(document.getElementById('panel'), state, {
    LITHOLOGIES, JOINT_STYLES, DEFAULT_WEATHERING,
    BUILD: __BUILD__,
    gpu: `${d.info.vendor} ${d.info.architecture}`.trim() || 'GPU',
    maxResolution: maxRes,
    live: true,
    onGenerate: () => { state.params.seed = (state.params.seed | 0); restart(); },
    // Structure/petrology/weathering changes rebuild the block. Dragging a
    // slider fires per pixel, and a reset reallocates buffers when the
    // resolution changes, so coalesce to one restart per frame.
    onParams: scheduleRestart,
    onShading: applyShading,
    onEnv: applyShading,
    onPlayPause: (v) => { state.playing = v; },
    onScrub: (v) => scrubTo(v),
    onSelect: (i) => {
      state.selected = Math.max(0, Math.min(i, batchList().length - 1));
      restart();
    },
    onStepOnce: () => { engine.advance(1); ui.setSolve(solveInfo()); },
    onExport: exportMesh,
    onExportAll: exportAll,
  });

  restart();
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ solve
let solveStart = 0;
let stepsThisSecond = 0, lastRate = 0, rateClock = 0;

/** The parameter variants in the current batch. */
function batchList() {
  if (state.batch <= 1) return [state.params];
  return makeBatchParams(state.params, state.batch, state.spread);
}

function currentParams() {
  const list = batchList();
  const base = list[Math.min(state.selected, list.length - 1)] || state.params;
  return {
    ...base,
    // Resolution and mesh settings are global, not per-variant.
    resolution: state.params.resolution,
    shading: { ...state.shading, exposure: state.env.exposure, debugMode: state.debugMode },
  };
}

let restartQueued = false;
function scheduleRestart() {
  if (restartQueued) return;
  restartQueued = true;
  requestAnimationFrame(() => { restartQueued = false; restart(); });
}

function restart() {
  if (!engine) return;
  try {
    const info = engine.reset(currentParams());
    solveStart = performance.now();
    state.playing = true;
    ui.setSolve(solveInfo(info));
  } catch (e) {
    fatal('Solver init failed', e.message);
  }
}

/**
 * Scrub to an arbitrary fraction of the solve. Going backwards means
 * re-running from the fresh block, which sounds expensive and is not: a full
 * 90-step solve at 64^3 is a few milliseconds of GPU time, so a scrub is
 * effectively instant.
 */
function scrubTo(frac) {
  const target = Math.round(frac * engine.totalSteps);
  if (target < engine.step) {
    engine.reset(currentParams());
  }
  state.playing = false;
  const todo = target - engine.step;
  if (todo > 0) engine.advance(todo);
  ui.setSolve(solveInfo());
}

function solveInfo(extra) {
  return {
    step: engine.step,
    totalSteps: engine.totalSteps,
    done: engine.done,
    stoppedEarly: engine.stoppedEarly,
    volumeFraction: engine.volumeFraction,
    stepsPerSecond: lastRate,
    elapsedMs: performance.now() - solveStart,
    resolution: engine.n,
    ...extra,
  };
}

function applyShading() {
  if (!engine || !engine.buf) return;
  const S = state.shading;
  engine.setUniform('weatherAge', S.weatherAge);
  engine.setUniform('lichen', S.lichen);
  engine.setUniform('caseHardening', S.caseHardening);
  engine.setUniform('dust', S.dust);
  engine.setUniform('wetness', S.wetness);
  engine.setUniform('microRelief', S.microRelief);
  engine.setUniform('retreatScale', S.retreatScale);
  engine.setUniform('exposure', state.env.exposure);
  engine.setUniform('debugMode', state.debugMode);
}

function sunDirection() {
  const az = (state.env.sunAzimuth * Math.PI) / 180;
  const el = (state.env.sunElevation * Math.PI) / 180;
  return [Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)];
}

// ------------------------------------------------------------------ export
/**
 * Dual contouring, on demand only. This is the one genuinely CPU-bound thing
 * left, and it is now off the interactive path entirely: you pay for it when
 * you ask for a mesh file, not while you are looking at the rock.
 */
async function exportMesh(fmt = 'obj', variant) {
  if (!engine || !engine.buf) return;
  ui.setBusy(true, 'Reading back field\u2026');
  try {
    // Pull phi plus the two solver fields the material needs downstream.
    // Sequential, NOT Promise.all: readField reuses one staging buffer, so
    // concurrent calls would race on the same mapping. Three 1 MB readbacks at
    // 64^3, and the contouring below dominates them by far anyway.
    const data = await engine.readField('phi');
    const retreatF = await engine.readField('retreat');
    const shelterF = await engine.readField('shelter');
    ui.setBusy(true, 'Contouring\u2026');
    await new Promise((r) => setTimeout(r, 0));   // let the UI paint

    const field = new Field3(engine.n, engine.meta.extent);
    field.data.set(data);
    let mesh = dualContour(field, { sharpness: state.params.sharpness });
    mesh = largestComponent(mesh);
    if (state.params.smoothing > 0) taubinSmooth(mesh, state.params.smoothing);
    recomputeNormals(mesh);

    // Bake the solver fields onto the vertices, same as the old CPU pipeline:
    // R = retreat, G = shelter, B = curvature ride out in the PLY so the
    // material can be rebuilt without re-running the solve.
    const retreatField = new Field3(engine.n, engine.meta.extent);
    retreatField.data.set(retreatF);
    const shelterField = new Field3(engine.n, engine.meta.extent);
    shelterField.data.set(shelterF);
    const curvField = new Field3(engine.n, engine.meta.extent);
    meanCurvatureField(field, curvField.data);

    const nv = mesh.positions.length / 3;
    const aRetreat = new Float32Array(nv);
    const aShelter = new Float32Array(nv);
    const aCurv = new Float32Array(nv);
    for (let v = 0; v < nv; v++) {
      const x = mesh.positions[v * 3], y = mesh.positions[v * 3 + 1], z = mesh.positions[v * 3 + 2];
      aRetreat[v] = retreatField.sample(x, y, z);
      aShelter[v] = shelterField.sample(x, y, z);
      aCurv[v] = curvField.sample(x, y, z) * engine.meta.size;
    }

    const rec = { mesh, aRetreat, aShelter, aCurv, stats: meshStats(mesh) };
    const v = variant || currentParams();
    const name = `boulder_${v.lithology}_s${v.seed}`;
    if (fmt === 'obj') download(toOBJ(rec, name), `${name}.obj`, 'text/plain');
    else download(toPLY(rec), `${name}.ply`);
    ui.setExportStats(rec.stats);
  } catch (e) {
    console.error(e);
    fatal('Export failed', e.message);
  } finally {
    ui.setBusy(false);
  }
}

/**
 * Volume, area and Wadell sphericity from the triangle soup (divergence
 * theorem). These are the numbers that make the output checkable against field
 * measurements, so they belong in the exported file, not just the UI.
 */
function meshStats(mesh) {
  const I = mesh.indices, P = mesh.positions;
  let vol = 0, area = 0;
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const ax = P[a], ay = P[a + 1], az = P[a + 2];
    const bx = P[b], by = P[b + 1], bz = P[b + 2];
    const cx = P[c], cy = P[c + 1], cz = P[c + 2];
    vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const wx = cx - ax, wy = cy - ay, wz = cz - az;
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    area += 0.5 * Math.hypot(nx, ny, nz);
  }
  vol = Math.abs(vol);
  // Wadell: sphericity = (surface area of a sphere of equal volume) / area
  const sphericity = area > 0
    ? (Math.PI ** (1 / 3) * (6 * vol) ** (2 / 3)) / area : 0;
  return {
    triangles: I.length / 3,
    vertices: P.length / 3,
    volume: vol,
    area,
    sphericity,
    massKg: vol * 2680,      // granite bulk density
  };
}

/**
 * Export every variant in the batch. Each one is re-solved on the GPU (fast)
 * and contoured on the CPU (not fast), so this is sequential and explicitly
 * progress-reported rather than pretending to be instant.
 */
async function exportAll(fmt = 'obj') {
  const list = batchList();
  const wasSelected = state.selected;
  for (let i = 0; i < list.length; i++) {
    state.selected = i;
    restart();
    engine.advance(engine.totalSteps);        // solve to completion immediately
    ui.setBusy(true, `Exporting ${i + 1}/${list.length}\u2026`);
    await exportMesh(fmt, list[i]);
  }
  state.selected = wasSelected;
  restart();
}

// ---------------------------------------------------------------- run loop
let lastT = performance.now();

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2) * state.quality;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    cam.dirty = true;
  }
}

function tick(now) {
  requestAnimationFrame(tick);
  if (!engine || !engine.buf) return;
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;

  resize();
  cam.autoRotate = state.autoRotate;
  cam.update(dt, canvas.width / canvas.height);

  // --- solve ---------------------------------------------------------------
  if (state.playing && !engine.done) {
    const taken = engine.advance(state.stepsPerFrame);
    stepsThisSecond += taken;
  }
  rateClock += dt;
  if (rateClock >= 0.5) {
    lastRate = Math.round(stepsThisSecond / rateClock);
    stepsThisSecond = 0; rateClock = 0;
    ui.setSolve(solveInfo());
  }

  // --- draw ----------------------------------------------------------------
  engine.updateCamera({
    invViewProj: cam.invViewProj,
    eye: cam.eye,
    sun: sunDirection(),
    // Object-space size of one pixel at unit distance: drives micro-relief
    // filtering, so the grains fade into roughness instead of aliasing.
    aspect: (2 * Math.tan(cam.fov / 2)) / canvas.height,
    time: now / 1000,
    maxSteps: state.quality < 0.9 ? 96 : 160,
  });
  engine.render(context.getCurrentTexture().createView());
}

boot();
