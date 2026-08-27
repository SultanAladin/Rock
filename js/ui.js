/*
 * ui.js — control panel wiring. It owns the DOM state and produces a
 * "params" object the renderer reads each frame to set shader uniforms.
 */

const UI = (function () {
  const M = window.RockMaterials;

  // ------ state -------------------------------------------------------------
  const state = {
    rockIndex: 0,
    shapeIndex: 0,
    erosionIndex: 2,   // chemical dissolution / tafoni (honeycomb)
    seed: 1.0,
    erosion: 0.0,          // the "time" slider 0..1
    shapeRough: null,      // from shape preset unless overridden
    detailAmp: null,       // per-rock defaults unless overridden
    detailFreq: null,
    carving: null,
    striation: null,
    cameraAuto: true,
    cameraOrbit: 0.0,
    cameraPitch: 0.35,
    cameraDist: 5.2,
    selectedErosion: {}    // erosion-specific sliders kept per process
  };

  // ------ slider helpers ----------------------------------------------------
  function makeSlider(container, label, id, min, max, step, value) {
    const row = document.createElement('div');
    row.className = 'ctl-ctl';
    row.innerHTML =
      '<label for="' + id + '">' + label + '</label>' +
      '<div class="ctl-track"><input type="range" id="' + id + '" min="' + min +
      '" max="' + max + '" step="' + step + '" value="' + value + '">' +
      '</div><span class="ctl-val" data-for="' + id + '">' + value + '</span>';
    container.appendChild(row);
    const input = row.querySelector('input');
    const val = row.querySelector('.ctl-val');
    function fmt(v) { return max <= 1 ? (+v).toFixed(2) : (+v).toFixed(max >= 100 ? 1 : 3); }
    input.addEventListener('input', function () {
      val.textContent = fmt(input.value);
      onSliderChange(id, +input.value);
    });
    return input;
  }

  // one shared callback the renderer subscribes to
  let changeCallback = null;
  function onSliderChange(id, v) {
    if (id === 'erosion') state.erosion = v;
    else if (id === 'seed') { state.seed = v; }
    else if (id === 'detail-amp') state.detailAmp = v;
    else if (id === 'detail-freq') state.detailFreq = v;
    else if (id === 'carving') state.carving = v;
    else if (id === 'striation') state.striation = v;
    else if (id === 'cam-dist') state.cameraDist = v;
    else if (id === 'cam-pitch') state.cameraPitch = v;
    if (changeCallback) changeCallback();
  }

  // ------ build the panel ---------------------------------------------------
  function build() {
    const panel = document.getElementById('controls');

    // --- Rock type ---
    const rockSel = document.createElement('div');
    rockSel.className = 'group';
    rockSel.innerHTML = '<div class="group-title">Rock Type</div>';
    const rockRow = document.createElement('div');
    rockRow.className = 'picker';
    M.ROCK_TYPES.forEach((r, i) => {
      const btn = document.createElement('button');
      btn.className = 'chip';
      btn.textContent = r.name;
      btn.dataset.i = i;
      btn.addEventListener('click', () => pickRock(i));
      rockRow.appendChild(btn);
    });
    rockSel.appendChild(rockRow);
    panel.appendChild(rockSel);

    // --- Shape generator ---
    const shapeSel = document.createElement('div');
    shapeSel.className = 'group';
    shapeSel.innerHTML = '<div class="group-title">Shape Generator</div>';
    const shapeRow = document.createElement('div');
    shapeRow.className = 'picker';
    M.SHAPE_PRESETS.forEach((s, i) => {
      const btn = document.createElement('button');
      btn.className = 'chip';
      btn.textContent = s.name;
      btn.dataset.i = i;
      btn.addEventListener('click', () => pickShape(i));
      shapeRow.appendChild(btn);
    });
    shapeSel.appendChild(shapeRow);
    panel.appendChild(shapeSel);

    const seedRow = makeSlider(panel, 'Seed / variation', 'seed', 0, 5, 0.01, state.seed);
    seedRow.closest('.ctl-ctl').classList.add('seed-row');

    // --- Erosion process ---
    const eroSel = document.createElement('div');
    eroSel.className = 'group';
    eroSel.innerHTML = '<div class="group-title">Erosion Process</div>';
    const eroRow = document.createElement('div');
    eroRow.className = 'picker picker-wrap';
    M.EROSION_TYPES.forEach((e, i) => {
      const btn = document.createElement('button');
      btn.className = 'chip';
      btn.textContent = e.name;
      btn.dataset.i = i;
      btn.addEventListener('click', () => {
        state.erosionIndex = i;
        setActive($('#eroRow'), i);
        if (window.__rockUpdateReadouts) window.__rockUpdateReadouts();
      });
      eroRow.appendChild(btn);
    });
    eroRow.id = 'eroRow';
    eroSel.appendChild(eroRow);
    panel.appendChild(eroSel);
    setActive(eroRow, state.erosionIndex);

    // The erosion "time" scrubber lives at the bottom of the screen; add a note.
    const eroNote = document.createElement('div');
    eroNote.className = 'ctl-ctl';
    eroNote.style.gridTemplateColumns = '1fr';
    eroNote.innerHTML =
      '<label style="grid-column:1/-1">Erosion amount is scrubbed with the <b>timeline slider</b> at the bottom of the screen.</label>';
    panel.appendChild(eroNote);

    // --- Fine detail ---
    for (const [id, label, max] of [['detail-amp', 'Micro roughness', 1], ['detail-freq', 'Detail frequency', 10], ['carving', 'Carve amount', 1], ['striation', 'Striation', 1]]) {
      makeSlider(panel, label, id, 0, max, id === 'detail-freq' ? 0.1 : 0.01, 0.5);
    }

    // --- Camera ---
    const camSel = document.createElement('div');
    camSel.className = 'group';
    camSel.innerHTML = '<div class="group-title">Camera</div>';
    const camRow = document.createElement('div');
    camRow.className = 'picker';
    const autoBtn = document.createElement('button');
    autoBtn.className = 'chip';
    autoBtn.textContent = 'Orbit auto';
    autoBtn.style.marginRight = '6px';
    autoBtn.addEventListener('click', () => { state.cameraAuto = !state.cameraAuto; autoBtn.classList.toggle('active', state.cameraAuto); });
    autoBtn.classList.add('active');
    camRow.appendChild(autoBtn);
    camSel.appendChild(camRow);
    panel.appendChild(camSel);
    makeSlider(panel, 'Camera distance', 'cam-dist', 2.3, 12, 0.05, state.cameraDist);
    makeSlider(panel, 'Camera pitch', 'cam-pitch', 0.05, 1.4, 0.02, state.cameraPitch);

    // apply defaults
    applyRock(state.rockIndex);
    applyShape(state.shapeIndex);
  }

  function setActive(container, idx) {
    const chips = container.querySelectorAll('.chip');
    chips.forEach((c) => c.classList.toggle('active', (+c.dataset.i) === idx));
  }

  function pickRock(i) {
    if (i === state.rockIndex) { applyRock(i); return; }
    state.rockIndex = i;
    applyRock(i);
    if (window.__rockUpdateReadouts) window.__rockUpdateReadouts();
  }

  // push a rock's defaults into the detail sliders
  function applyRock(i) {
    setActive($('.picker')[0], i);      // first .picker = rock chips
    const r = M.ROCK_TYPES[i];
    setSlider('detail-amp', r.detailAmp);
    setSlider('detail-freq', r.detailFreq);
    setSlider('carving', r.carveAmt);
    setSlider('striation', r.striAmt);
    state.detailAmp = r.detailAmp;
    state.detailFreq = r.detailFreq;
    state.carving = r.carveAmt;
    state.striation = r.striAmt;
    if (changeCallback) changeCallback();
  }

  function pickShape(i) {
    state.shapeIndex = i;
    applyShape(i);
    if (window.__rockUpdateReadouts) window.__rockUpdateReadouts();
  }

  function applyShape(i) {
    setActive($('.picker')[1], i);
    const s = M.SHAPE_PRESETS[i];
    state.shapeRough = s.rough;
    if (changeCallback) changeCallback();
  }

  function setSlider(id, v) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = v;
    const val = document.querySelector('.ctl-val[data-for="' + id + '"]');
    if (val) val.textContent = (+v).toFixed(maxFor(id) ? 2 : 3);
  }
  function maxFor(id) { const el = document.getElementById(id); return el ? +el.max <= 1 : false; }

  function $(sel) { return document.querySelector(sel); }

  // ------ produce renderer-ready params ------------------------------------
  function params() {
    const r = M.ROCK_TYPES[state.rockIndex];
    const s = M.SHAPE_PRESETS[state.shapeIndex];
    const e = M.EROSION_TYPES[state.erosionIndex];
    return {
      c1: M.hexToRgb(r.c1),
      c2: M.hexToRgb(r.c2),
      c3: M.hexToRgb(r.c3),
      speck: r.speck,
      band: r.band,
      bandFreq: r.bandFreq,
      moss: r.moss,
      weather: r.weather,
      weatherColor: M.hexToRgb(r.weatherColor),
      detailAmp: state.detailAmp,
      detailFreq: state.detailFreq,
      carveAmt: state.carving,
      carveFreq: r.carveFreq,
      striAmt: state.striation,
      striFreq: r.striFreq,
      shine: r.shine,
      specAmt: r.specAmt,
      shape: s.shape,
      shapeRough: state.shapeRough,
      rockY: s.yOff,
      seed: state.seed,
      erosErodeType: e.idx,
      erosion: state.erosion,
      rockName: r.name,
      rockType: r.type,
      erosionName: e.name,
      cameraAuto: state.cameraAuto,
      cameraOrbit: state.cameraOrbit,
      cameraPitch: state.cameraPitch,
      cameraDist: state.cameraDist
    };
  }

  function onChange(cb) { changeCallback = cb; }

  return {
    build,
    params,
    onChange,
    get state() { return state; }
  };
})();

window.UI = UI;
