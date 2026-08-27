/*
 * main.js — bootstraps the app, wires the erosion "time" scrubber to the
 * renderer, and animates the frame loop.
 */
(function () {
  const M = window.RockMaterials;
  const UI = window.UI;
  const Renderer = window.Renderer;

  const canvas = document.getElementById('glcanvas');

  // graceful fallback if WebGL is unavailable
  try {
    Renderer.init(canvas);
  } catch (err) {
    console.error(err);
    const d = document.createElement('div');
    d.className = 'webgl-fallback';
    d.innerHTML =
      '<h2>WebGL not available</h2>' +
      '<p>The SDF rock renderer needs WebGL. Please use a modern browser ' +
      'with hardware acceleration enabled (e.g. Chrome with GPU on).</p>' +
      '<code>' + (err && err.message ? err.message : '') + '</code>';
    document.body.appendChild(d);
    return;
  }

  UI.build();
  UI.onChange(() => { /* uniforms are recomputed from UI.params every frame */ });

  const erosionSlider = document.getElementById('erosion-scrub');
  const erosionPercent = document.getElementById('erosion-percent');
  const erosionProcess = document.getElementById('erosion-process');
  const rockReadout = document.getElementById('rock-readout');
  const fpsEl = document.getElementById('fps');

  function clamp01(v) { return Math.max(0, Math.min(1, v)); }

  let lastEroName = '';
  function updateReadouts() {
    const p = UI.params();
    const pct = clamp01(p.erosion);

    erosionPercent.textContent = Math.round(pct * 100) + '%';
    erosionProcess.textContent = 'Erosion: ' + p.erosionName;
    rockReadout.textContent = p.rockName + '  ·  ' + p.rockType + ' rock';
    if (p.rockName !== lastEroName) {
      // when the rock changes, don't jump the slider — just refresh text
    }
    lastEroName = p.rockName;
  }

  function onScrub() {
    const pct = clamp01(erosionSlider.value / 100);
    UI.state.erosion = pct;
    stopPlay();
    updateReadouts();
  }
  erosionSlider.addEventListener('input', onScrub);
  erosionSlider.addEventListener('change', onScrub);

  // ---- auto-play: scrub erosion over geologic time ----------------------
  const playBtn = document.getElementById('erosion-play');
  let playing = false, playStart = 0, playDur = 9000; // 9 s full cycle
  function setPlaying(v) {
    playing = v;
    playBtn.classList.toggle('playing', v);
    playBtn.textContent = v ? '❚❚' : '▶';
  }
  function stopPlay() { if (playing) setPlaying(false); }
  function tickPlay(now) {
    if (!playing) return;
    const t = ((now - playStart) % playDur) / playDur; // 0..1 looping
    const pct = t * 100;
    erosionSlider.value = pct;
    UI.state.erosion = t;
    updateReadouts();
    if (t >= 0.999) { setPlaying(false); erosionSlider.value = 100; UI.state.erosion = 1; updateReadouts(); }
  }
  playBtn.addEventListener('click', () => {
    if (playing) { setPlaying(false); return; }
    playStart = performance.now();
    setPlaying(true);
  });

  // surface any runtime error so it isn't a silent black canvas
  function showError(msg) {
    const el = document.getElementById('gl-error');
    if (el) { el.textContent = msg; el.style.display = 'block'; return; }
    const d = document.createElement('div');
    d.id = 'gl-error';
    d.className = 'webgl-fallback';
    d.innerHTML = '<h2>Renderer error</h2><code>' + msg + '</code>';
    document.body.appendChild(d);
  }
  window.addEventListener('error', (e) => { if (e && e.message) showError(e.message); });

  // simple FPS meter + adaptive quality
  let fpsLast = 0, fpsCount = 0;
  function loop(now) {
    tickPlay(now);
    try {
      const p = UI.params();
      Renderer.render(now, p);
    } catch (err) {
      showError((err && err.message) ? err.message : String(err));
      return;                    // stop the loop; the error is visible
    }
    if (now - fpsLast > 800) {
      const fps = Math.round(fpsCount * 1000 / (now - fpsLast));
      fpsEl.textContent = fps + ' fps';
      Renderer.setQuality(fps);
      fpsLast = now; fpsCount = 0;
    }
    fpsCount++;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  updateReadouts();
  window.__rockUpdateReadouts = updateReadouts; // let ui.js refresh after picks
})();
