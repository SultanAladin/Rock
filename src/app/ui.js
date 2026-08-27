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
    <p>Joint-block structure &rarr; crystal aggregate &rarr; curvature-driven level-set weathering,
       solved and raymarched entirely on the GPU.</p>
    <p class="build">build ${ctx.BUILD}${ctx.gpu ? ' &middot; ' + ctx.gpu : ''}</p>`;
  root.appendChild(head);

  const genBtn = el('button', 'primary', 'New boulder');
  genBtn.onclick = ctx.onGenerate;
  root.appendChild(genBtn);

  const bar = el('div', 'bar');
  const barFill = el('div', 'bar-fill');
  bar.appendChild(barFill); root.appendChild(bar);

  const statsBox = el('div', 'stats');
  root.appendChild(statsBox);

  // ---- live solve transport ---------------------------------------------
  // The solver runs in the render loop, so these are transport controls over a
  // simulation that is already running, not a job queue.
  let scrubbing = false;
  const solveBox = el('div', 'solve');
  const trow = el('div', 'transport');
  const playBtn = el('button', 'tbtn', '\u23f8');
  playBtn.title = 'Pause / resume the erosion';
  let playing = true;
  playBtn.onclick = () => {
    playing = !playing;
    playBtn.textContent = playing ? '\u23f8' : '\u25b6';
    ctx.onPlayPause && ctx.onPlayPause(playing);
  };
  const stepBtn = el('button', 'tbtn', '\u23ed');
  stepBtn.title = 'Advance one iteration';
  stepBtn.onclick = () => {
    if (playing) playBtn.onclick();
    ctx.onStepOnce && ctx.onStepOnce();
  };
  const restartBtn = el('button', 'tbtn', '\u21ba');
  restartBtn.title = 'Restart the solve from the fresh joint block';
  restartBtn.onclick = ctx.onGenerate;
  trow.appendChild(playBtn); trow.appendChild(stepBtn); trow.appendChild(restartBtn);

  const scrub = el('input');
  scrub.type = 'range'; scrub.min = 0; scrub.max = 1000; scrub.step = 1; scrub.value = 0;
  scrub.className = 'scrub';
  scrub.oninput = () => { scrubbing = true; };
  scrub.onchange = () => {
    ctx.onScrub && ctx.onScrub(parseInt(scrub.value, 10) / 1000);
    scrubbing = false;
  };
  trow.appendChild(scrub);
  solveBox.appendChild(trow);
  const solveStats = el('div', 'solve-stats');
  solveBox.appendChild(solveStats);
  root.appendChild(solveBox);

  const perf = section(root, 'Performance');
  slider(perf, 'Iterations per frame', state, 'stepsPerFrame', 1, 32, 1);
  slider(perf, 'Render scale', state, 'quality', 0.4, 1.5, 0.05);
  const noteP = el('p', 'note');
  noteP.innerHTML = 'Every iteration you see is a real level-set step on the GPU: the raymarcher reads the same buffer the solver writes, so nothing is baked and nothing is re&#8209;meshed to display it.';
  perf.appendChild(noteP);

  // ---- batch ------------------------------------------------------------
  const b = section(root, 'Batch');
  slider(b, 'Count', state, 'batch', 1, 24, 1, () => { renderBatch(); ctx.onParams(); });
  slider(b, 'Master seed', state.params, 'seed', 1, 9999, 1, ctx.onParams);
  // Variant picker. A "batch" here is a set of parameter variants; each solves
  // in milliseconds on the GPU, so switching is instant rather than a queue.
  const batchRow = el('div', 'chips');
  b.appendChild(batchRow);
  const renderBatch = () => {
    batchRow.innerHTML = '';
    if (state.batch <= 1) { batchRow.style.display = 'none'; return; }
    batchRow.style.display = '';
    for (let i = 0; i < state.batch; i++) {
      const c = el('div', 'chip' + (i === state.selected ? ' on' : ''), String(i + 1));
      c.onclick = () => {
        state.selected = i;
        renderBatch();
        ctx.onSelect && ctx.onSelect(i);
      };
      batchRow.appendChild(c);
    }
  };
  renderBatch();
  slider(b, 'Size spread &sigma; (log-normal)', state.spread, 'sizeSigma', 0, 1.0, 0.01);
  slider(b, 'Weathering spread &sigma;', state.spread, 'weatherSigma', 0, 1.2, 0.01);
  slider(b, 'Size spread &sigma; (log-normal)', state.spread, 'sizeSigma', 0, 1.0, 0.01, ctx.onParams);
  slider(b, 'Weathering spread &sigma;', state.spread, 'weatherSigma', 0, 1.2, 0.01, ctx.onParams);
  multi(b, 'Lithology mix', Object.keys(ctx.LITHOLOGIES).map((k) => [k, ctx.LITHOLOGIES[k].label.split(' ')[0]]),
    [], (v) => { state.spread.lithoMix = v.length ? v : null; });
  multi(b, 'Joint-style mix', Object.keys(ctx.JOINT_STYLES).map((k) => [k, k]),
    [], (v) => { state.spread.styleMix = v.length ? v : null; });

  // ---- structure --------------------------------------------------------
  const st = section(root, 'Structure \u2014 jointing');
  select(st, 'Joint style', state.params, 'jointStyle',
    Object.keys(ctx.JOINT_STYLES).map((k) => [k, ctx.JOINT_STYLES[k].label]), ctx.onParams);
  slider(st, 'Block size (m)', state.params, 'size', 0.15, 3.0, 0.01, ctx.onParams);
  slider(st, 'Aspect variation', state.params, 'aspectVariation', 0, 0.9, 0.01, ctx.onParams);
  slider(st, 'Joint-surface roughness', state.params, 'jointRoughness', 0, 3.0, 0.02, ctx.onParams);
  slider(st, 'Hurst exponent H', state.params, 'hurst', 0.5, 1.0, 0.01, ctx.onParams);
  slider(st, 'Sheeting curvature', state.params, 'sheetingCurvature', 0, 1.5, 0.01, ctx.onParams);
  const note1 = el('p', 'note');
  note1.innerHTML = 'H&nbsp;&asymp;&nbsp;0.8 is the measured self-affine exponent of mode&#8209;I fracture surfaces in granite. Lower&nbsp;H = harsher, more angular fracture faces.';
  st.appendChild(note1);

  // ---- petrology --------------------------------------------------------
  const pt = section(root, 'Petrology');
  select(pt, 'Lithology', state.params, 'lithology',
    Object.keys(ctx.LITHOLOGIES).map((k) => [k, ctx.LITHOLOGIES[k].label]), ctx.onParams);
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
  slider(w, 'Exposure age', W, 'years', 0, 3.0, 0.01, ctx.onParams);
  slider(w, 'Spheroidal rate A<sub>sph</sub>', W, 'spheroidal', 0, 3.0, 0.02, ctx.onParams);
  slider(w, 'Spheroidal exponent p', W, 'spheroidalPower', 0.6, 2.2, 0.02, ctx.onParams);
  slider(w, 'Cavernous (tafoni) A<sub>cav</sub>', W, 'cavernous', 0, 2.0, 0.02, ctx.onParams);
  slider(w, 'Cavernous exponent q', W, 'cavernousPower', 0.8, 3.0, 0.02, ctx.onParams);
  slider(w, 'Uniform lowering', W, 'uniform', 0, 0.5, 0.005, ctx.onParams);
  slider(w, 'Grussification (mineral selectivity)', W, 'grussification', 0, 2.0, 0.02, ctx.onParams);
  slider(w, 'Basal moisture gradient', W, 'moistureGradient', 0, 2.0, 0.02, ctx.onParams);
  slider(w, 'Buried fraction', W, 'buriedFraction', 0, 0.6, 0.01, ctx.onParams);
  slider(w, 'Insolation / aspect bias', W, 'insolation', 0, 1.0, 0.01, ctx.onParams);
  slider(w, 'Rindlet amplitude', W, 'rindlet', 0, 1.2, 0.01, ctx.onParams);
  slider(w, 'Rindlet spacing (mm)', W, 'rindletSpacing', 0.01, 0.12, 0.001, ctx.onParams, (v) => (v * 1000).toFixed(0));
  slider(w, 'Shelter radius (m)', W, 'shelterRadius', 0.05, 0.6, 0.005, ctx.onParams);
  const note2 = el('p', 'note');
  note2.innerHTML = 'Rate &prop; mean curvature: corners see three joint faces of attack, edges two, faces one \u2014 the mechanism behind corestone rounding. Concave shelter feedback drives tafoni. Rindlet spacing of 35&ndash;50&nbsp;mm matches field measurement on granite corestones.';
  w.appendChild(note2);

  // ---- mesh -------------------------------------------------------------
  const m = section(root, 'Solver grid & mesh export', true);
  slider(m, 'Grid resolution', state.params, 'resolution', 32,
    ctx.maxResolution || 128, 4, ctx.onParams);
  slider(m, 'Arris sharpness (QEF)', state.params, 'sharpness', 0, 1, 0.01);
  slider(m, 'Taubin smoothing passes', state.params, 'smoothing', 0, 6, 1);
  const note3 = el('p', 'note');
  note3.innerHTML = 'The viewport raymarches the signed&#8209;distance field directly, so resolution costs solver accuracy, not frame time. Dual contouring &mdash; QEF vertex placement, which keeps fresh joint arrises razor&#8209;sharp while weathered shoulders stay smooth &mdash; runs only when you export.';
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
    setBusy(v, label) {
      genBtn.disabled = v;
      genBtn.textContent = v ? (label || 'Working\u2026') : 'New boulder';
    },
    setProgress(f) { barFill.style.width = `${(f * 100).toFixed(1)}%`; },

    /** Live solver readout. Called a couple of times a second, not per frame. */
    setSolve(s) {
      const frac = s.totalSteps ? s.step / s.totalSteps : 0;
      barFill.style.width = `${(frac * 100).toFixed(1)}%`;
      if (!scrubbing) scrub.value = Math.round(frac * 1000);
      const status = s.stoppedEarly
        ? '<b class="warn">stopped early</b> <span class="dim">(volume floor reached &mdash; the corestone would have become a cobble and been transported away)</span>'
        : s.done ? '<b>complete</b>' : `<b>${s.stepsPerSecond}</b> iterations/s`;
      solveStats.innerHTML = `
        <div>iteration <b>${s.step}</b> / ${s.totalSteps} &middot; ${status}</div>
        <div>grid <b>${s.resolution}&sup3;</b> = ${(s.resolution ** 3 / 1000).toFixed(0)}k cells
             &middot; solid <b>${(s.volumeFraction * 100).toFixed(1)}%</b> of fresh block</div>`;
    },

    setExportStats(e) {
      const mass = e.massKg > 1000 ? `${(e.massKg / 1000).toFixed(2)} t` : `${e.massKg.toFixed(0)} kg`;
      statsBox.innerHTML = `
        <div>exported <b>${(e.triangles / 1000).toFixed(1)}k</b> triangles &middot;
             <b>${(e.vertices / 1000).toFixed(1)}k</b> vertices</div>
        <div>Wadell sphericity <b>${e.sphericity.toFixed(3)}</b>
             <span class="dim">(field corestones 0.75&ndash;0.90)</span></div>
        <div>volume <b>${e.volume.toFixed(4)} m&sup3;</b> &middot; mass <b>${mass}</b>
             <span class="dim">@ 2680 kg/m&sup3;</span></div>`;
    },
  };
}
