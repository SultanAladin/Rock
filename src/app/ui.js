/** Control panel. Plain DOM, no framework - fewer moving parts, faster. */

function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt !== undefined) e.textContent = txt;
  return e;
}

function section(parent, title, collapsed = false) {
  const s = el('div', 'section');
  const h = el('div', 'section-h');
  h.innerHTML = `<span class="chev">${collapsed ? '\u25b8' : '\u25be'}</span><span>${title}</span>`;
  const body = el('div', 'section-b');
  if (collapsed) body.style.display = 'none';
  h.onclick = () => {
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : '';
    h.querySelector('.chev').textContent = open ? '\u25b8' : '\u25be';
  };
  s.appendChild(h); s.appendChild(body); parent.appendChild(s);
  return body;
}

function slider(parent, label, obj, key, min, max, step, onChange, fmt) {
  const row = el('div', 'row');
  const lab = el('label', null, label);
  const val = el('span', 'val');
  const inp = el('input');
  inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step;
  inp.value = obj[key];
  const render = () => { val.textContent = fmt ? fmt(obj[key]) : (+obj[key]).toFixed(step < 0.01 ? 3 : step < 1 ? 2 : 0); };
  render();
  inp.oninput = () => { obj[key] = parseFloat(inp.value); render(); onChange && onChange(); };
  const head = el('div', 'row-head');
  head.appendChild(lab); head.appendChild(val);
  row.appendChild(head); row.appendChild(inp);
  parent.appendChild(row);
  return { input: inp, refresh: () => { inp.value = obj[key]; render(); } };
}

function select(parent, label, obj, key, options, onChange) {
  const row = el('div', 'row');
  row.appendChild(el('label', null, label));
  const sel = el('select');
  for (const [v, t] of options) {
    const o = el('option', null, t); o.value = v; sel.appendChild(o);
  }
  sel.value = obj[key];
  sel.onchange = () => { obj[key] = sel.value; onChange && onChange(); };
  row.appendChild(sel);
  parent.appendChild(row);
  return sel;
}

function toggle(parent, label, obj, key, onChange) {
  const row = el('div', 'row row-inline');
  const inp = el('input'); inp.type = 'checkbox'; inp.checked = !!obj[key];
  inp.onchange = () => { obj[key] = inp.checked; onChange && onChange(); };
  const lab = el('label', null, label);
  lab.prepend(inp);
  row.appendChild(lab);
  parent.appendChild(row);
  return inp;
}

function multi(parent, label, options, initial, onChange) {
  const row = el('div', 'row');
  row.appendChild(el('label', null, label));
  const box = el('div', 'chips');
  const chosen = new Set(initial || []);
  for (const [v, t] of options) {
    const c = el('div', 'chip' + (chosen.has(v) ? ' on' : ''), t);
    c.onclick = () => {
      if (chosen.has(v)) chosen.delete(v); else chosen.add(v);
      c.classList.toggle('on');
      onChange([...chosen]);
    };
    box.appendChild(c);
  }
  row.appendChild(box);
  parent.appendChild(row);
}

export function buildUI(root, state, ctx) {
  root.innerHTML = '';

  // ---- header -----------------------------------------------------------
  const head = el('div', 'panel-head');
  head.innerHTML = `<h1>Granite Boulder Forge</h1>
    <p>Joint-block structure &rarr; crystal aggregate &rarr; curvature-driven level-set weathering &rarr; dual contouring.</p>`;
  root.appendChild(head);

  const genBtn = el('button', 'primary', 'Generate batch');
  genBtn.onclick = ctx.onGenerate;
  root.appendChild(genBtn);

  const bar = el('div', 'bar');
  const barFill = el('div', 'bar-fill');
  bar.appendChild(barFill); root.appendChild(bar);

  const statsBox = el('div', 'stats');
  root.appendChild(statsBox);

  // ---- batch ------------------------------------------------------------
  const b = section(root, 'Batch');
  slider(b, 'Count', state, 'batch', 1, 24, 1);
  toggle(b, 'Progressive (draft pass first)', state, 'progressive');
  slider(b, 'Master seed', state.params, 'seed', 1, 9999, 1);
  slider(b, 'Size spread &sigma; (log-normal)', state.spread, 'sizeSigma', 0, 1.0, 0.01);
  slider(b, 'Weathering spread &sigma;', state.spread, 'weatherSigma', 0, 1.2, 0.01);
  multi(b, 'Lithology mix', Object.keys(ctx.LITHOLOGIES).map((k) => [k, ctx.LITHOLOGIES[k].label.split(' ')[0]]),
    [], (v) => { state.spread.lithoMix = v.length ? v : null; });
  multi(b, 'Joint-style mix', Object.keys(ctx.JOINT_STYLES).map((k) => [k, k]),
    [], (v) => { state.spread.styleMix = v.length ? v : null; });

  // ---- structure --------------------------------------------------------
  const st = section(root, 'Structure \u2014 jointing');
  select(st, 'Joint style', state.params, 'jointStyle',
    Object.keys(ctx.JOINT_STYLES).map((k) => [k, ctx.JOINT_STYLES[k].label]));
  slider(st, 'Block size (m)', state.params, 'size', 0.15, 3.0, 0.01);
  slider(st, 'Aspect variation', state.params, 'aspectVariation', 0, 0.9, 0.01);
  slider(st, 'Joint-surface roughness', state.params, 'jointRoughness', 0, 3.0, 0.02);
  slider(st, 'Hurst exponent H', state.params, 'hurst', 0.5, 1.0, 0.01);
  slider(st, 'Sheeting curvature', state.params, 'sheetingCurvature', 0, 1.5, 0.01);
  const note1 = el('p', 'note');
  note1.innerHTML = 'H&nbsp;&asymp;&nbsp;0.8 is the measured self-affine exponent of mode&#8209;I fracture surfaces in granite. Lower&nbsp;H = harsher, more angular fracture faces.';
  st.appendChild(note1);

  // ---- petrology --------------------------------------------------------
  const pt = section(root, 'Petrology');
  select(pt, 'Lithology', state.params, 'lithology',
    Object.keys(ctx.LITHOLOGIES).map((k) => [k, ctx.LITHOLOGIES[k].label]));
  const modeBox = el('div', 'modes');
  pt.appendChild(modeBox);
  const renderModes = () => {
    const L = ctx.LITHOLOGIES[state.params.lithology];
    modeBox.innerHTML = '';
    const total = Object.values(L.mode).reduce((a, c) => a + c, 0);
    for (const [k, v] of Object.entries(L.mode)) {
      if (v <= 0) continue;
      const r = el('div', 'mode');
      r.innerHTML = `<span>${k}</span><span class="mbar"><i style="width:${(v / total * 100).toFixed(1)}%"></i></span><b>${(v / total * 100).toFixed(0)}%</b>`;
      modeBox.appendChild(r);
    }
    const g = el('div', 'mode-note');
    g.innerHTML = `median grain <b>${(L.grain * 1000).toFixed(1)} mm</b> &middot; &sigma;<sub>ln</sub> ${L.grainSigma} &middot; ${L.phenocryst?.frac > 0.03 ? 'porphyritic' : 'equigranular'}${L.foliation ? ' &middot; foliated' : ''}`;
    modeBox.appendChild(g);
  };
  renderModes();
  pt.querySelector('select').addEventListener('change', renderModes);

  // ---- weathering -------------------------------------------------------
  const w = section(root, 'Weathering \u2014 level-set solver');
  const W = state.params.weathering;
  slider(w, 'Exposure age', W, 'years', 0, 3.0, 0.01);
  slider(w, 'Spheroidal rate A<sub>sph</sub>', W, 'spheroidal', 0, 3.0, 0.02);
  slider(w, 'Spheroidal exponent p', W, 'spheroidalPower', 0.6, 2.2, 0.02);
  slider(w, 'Cavernous (tafoni) A<sub>cav</sub>', W, 'cavernous', 0, 2.0, 0.02);
  slider(w, 'Cavernous exponent q', W, 'cavernousPower', 0.8, 3.0, 0.02);
  slider(w, 'Uniform lowering', W, 'uniform', 0, 0.5, 0.005);
  slider(w, 'Grussification (mineral selectivity)', W, 'grussification', 0, 2.0, 0.02);
  slider(w, 'Basal moisture gradient', W, 'moistureGradient', 0, 2.0, 0.02);
  slider(w, 'Buried fraction', W, 'buriedFraction', 0, 0.6, 0.01);
  slider(w, 'Insolation / aspect bias', W, 'insolation', 0, 1.0, 0.01);
  slider(w, 'Rindlet amplitude', W, 'rindlet', 0, 1.2, 0.01);
  slider(w, 'Rindlet spacing (mm)', W, 'rindletSpacing', 0.01, 0.12, 0.001, null, (v) => (v * 1000).toFixed(0));
  slider(w, 'Shelter radius (m)', W, 'shelterRadius', 0.05, 0.6, 0.005);
  slider(w, 'Solver steps', W, 'steps', 10, 240, 1);
  const note2 = el('p', 'note');
  note2.innerHTML = 'Rate &prop; mean curvature: corners see three joint faces of attack, edges two, faces one \u2014 the mechanism behind corestone rounding. Concave shelter feedback drives tafoni. Rindlet spacing of 35&ndash;50&nbsp;mm matches field measurement on granite corestones.';
  w.appendChild(note2);

  // ---- mesh -------------------------------------------------------------
  const m = section(root, 'Mesh', true);
  slider(m, 'Grid resolution', state.params, 'resolution', 32, 128, 4);
  slider(m, 'Arris sharpness (QEF)', state.params, 'sharpness', 0, 1, 0.01);
  slider(m, 'Taubin smoothing passes', state.params, 'smoothing', 0, 6, 1);
  slider(m, 'Grain micro-relief', state.params, 'microReliefAmount', 0, 3, 0.02);
  const note3 = el('p', 'note');
  note3.innerHTML = 'Dual contouring, not marching cubes: the QEF vertex placement keeps fresh joint arrises razor&#8209;sharp while weathered shoulders stay smooth. Cost is O(steps&nbsp;&times;&nbsp;N&sup3;) \u2014 128&sup3; is slow.';
  m.appendChild(note3);

  // ---- surface ----------------------------------------------------------
  const sh = section(root, 'Surface \u2014 shading');
  const S = state.shading;
  slider(sh, 'Weathering age (visual)', S, 'weatherAge', 0, 1, 0.01, ctx.onShading);
  slider(sh, 'Fe-oxide stain / retreat scale', S, 'retreatScale', 0, 30, 0.1, ctx.onShading);
  slider(sh, 'Case hardening', S, 'caseHardening', 0, 1, 0.01, ctx.onShading);
  slider(sh, 'Lichen cover', S, 'lichen', 0, 1, 0.01, ctx.onShading);
  slider(sh, 'Dust / soil contact', S, 'dust', 0, 1, 0.01, ctx.onShading);
  slider(sh, 'Wetness', S, 'wetness', 0, 1, 0.01, ctx.onShading);
  slider(sh, 'Micro-relief normals', S, 'microRelief', 0, 3, 0.02, ctx.onShading);

  // ---- environment ------------------------------------------------------
  const en = section(root, 'Environment', true);
  slider(en, 'Sun azimuth', state.env, 'sunAzimuth', 0, 360, 1, ctx.onEnv);
  slider(en, 'Sun elevation', state.env, 'sunElevation', 3, 89, 1, ctx.onEnv);
  slider(en, 'Exposure', state.env, 'exposure', 0.2, 3, 0.01, ctx.onEnv);
  toggle(en, 'Auto-rotate', state, 'autoRotate');
  toggle(en, 'Wireframe', state, 'wireframe', ctx.onShading);

  // ---- inspect ----------------------------------------------------------
  const dg = section(root, 'Inspect', true);
  const modes = [['0', 'Shaded'], ['1', 'Mineral map'], ['2', 'Surface retreat'], ['3', 'Shelter / AO'], ['4', 'Mean curvature']];
  const drow = el('div', 'chips');
  modes.forEach(([v, t]) => {
    const c = el('div', 'chip' + (v === '0' ? ' on' : ''), t);
    c.onclick = () => {
      drow.querySelectorAll('.chip').forEach((x) => x.classList.remove('on'));
      c.classList.add('on');
      state.debugMode = parseInt(v, 10);
      ctx.onShading();
    };
    drow.appendChild(c);
  });
  dg.appendChild(drow);

  // ---- export -----------------------------------------------------------
  const ex = section(root, 'Export');
  const erow = el('div', 'row');
  const selRow = el('div', 'row');
  selRow.appendChild(el('label', null, 'Selected index'));
  const selInp = el('input'); selInp.type = 'number'; selInp.min = 0; selInp.value = 0;
  selInp.oninput = () => { state.selected = Math.max(0, parseInt(selInp.value || '0', 10)); };
  selRow.appendChild(selInp);
  ex.appendChild(selRow);
  const mk = (label, fn) => { const btn = el('button', null, label); btn.onclick = fn; erow.appendChild(btn); };
  mk('OBJ', () => ctx.onExport('obj'));
  mk('PLY', () => ctx.onExport('ply'));
  mk('All OBJ', () => ctx.onExportAll('obj'));
  mk('All PLY', () => ctx.onExportAll('ply'));
  ex.appendChild(erow);
  const note4 = el('p', 'note');
  note4.innerHTML = 'PLY carries the solver output as vertex colour: R = surface retreat, G = shelter/AO, B = mean curvature. Rebuild the material downstream without re-running the solve.';
  ex.appendChild(note4);

  return {
    setBusy(v) { genBtn.disabled = v; genBtn.textContent = v ? 'Solving\u2026' : 'Generate batch'; },
    setProgress(f) { barFill.style.width = `${(f * 100).toFixed(1)}%`; },
    setStats(rocks, ms) {
      if (!rocks.length) { statsBox.innerHTML = ''; return; }
      const tris = rocks.reduce((a, m) => a + m.userData.stats.triangles, 0);
      const sph = rocks.reduce((a, m) => a + m.userData.stats.sphericity, 0) / rocks.length;
      const mass = rocks.reduce((a, m) => a + m.userData.stats.massKg, 0);
      const solve = rocks.reduce((a, m) => a + m.userData.stats.elapsedMs, 0);
      statsBox.innerHTML = `
        <div><b>${rocks.length}</b> boulders &middot; <b>${(tris / 1000).toFixed(0)}k</b> tris</div>
        <div>mean Wadell sphericity <b>${sph.toFixed(3)}</b> <span class="dim">(field corestones 0.75&ndash;0.90)</span></div>
        <div>total mass <b>${mass > 1000 ? (mass / 1000).toFixed(2) + ' t' : mass.toFixed(0) + ' kg'}</b> <span class="dim">@ 2680 kg/m&sup3;</span></div>
        <div>wall <b>${(ms / 1000).toFixed(2)} s</b> &middot; solver CPU <b>${(solve / 1000).toFixed(2)} s</b></div>`;
    },
  };
}
