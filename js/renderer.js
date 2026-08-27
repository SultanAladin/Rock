/*
 * renderer.js — the WebGL engine. Sets up the GL context, compiles the
 * raymarching shaders, reads params from UI each frame, and draws the rock.
 * Also owns the little drag-to-autofit camera orbit.
 */

const Renderer = (function () {
  const ROCK_LIGHT = Object.freeze({
    dir: [-0.55, 0.62, 0.35],        // normalized in code
    color: [1.0, 0.96, 0.88],
    skyTop: [0.28, 0.44, 0.74],       // deep blue horizon->zenith gradient
    skyHorizon: [0.72, 0.78, 0.86],
    sun: [1.0, 0.94, 0.82]
  });

  let gl = null;
  let program = null;
  let uniforms = null;
  let canvas = null;
  let clock = { t: 0 };

  const VIEW = { fov: 0.52, orbit: 0.0, pitch: 0.38, dist: 5.2, target: [0, 0, 0] };

  /* ---- shader helpers ---- */
  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s);
      console.error('Shader compile error:\n' + log);
      throw new Error('Shader compile failed');
    }
    return s;
  }

  function link(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.error('Program link error:\n' + gl.getProgramInfoLog(p));
      throw new Error('Program link failed');
    }
    return p;
  }

  // normalized direction
  function nrm(v) {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  }

  function init(canvasEl) {
    canvas = canvasEl;
    gl = canvas.getContext('webgl', { antialias: true, preserveDrawingBuffer: false });
    if (!gl) gl = canvas.getContext('experimental-webgl');
    if (!gl) throw new Error('WebGL not supported');

    const vs = compile(gl.VERTEX_SHADER, window.RockShaders.vert);
    const fs = compile(gl.FRAGMENT_SHADER, window.RockShaders.frag);
    program = link(vs, fs);
    gl.useProgram(program);

    // fullscreen triangle
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // cache uniform locations
    const names = [
      'uRes', 'uTime', 'uCamPos', 'uCamTarget', 'uFov',
      'uE', 'uErodeType', 'uShape', 'uShapeRough', 'uSeed', 'uRockPos',
      'uC1', 'uC2', 'uC3', 'uSpeck', 'uBand', 'uBandFreq',
      'uMoss', 'uWeather', 'uWeatherColor', 'uDetailAmp', 'uDetailFreq',
      'uCarveAmt', 'uCarveFreq', 'uStriAmt', 'uStriFreq',
      'uShininess', 'uSpecAmt',
      'uLightDir', 'uLightColor', 'uSkyTop', 'uSkyHorizon', 'uSunColor'
    ];
    uniforms = {};
    names.forEach((n) => { uniforms[n] = gl.getUniformLocation(program, n); });

    resize(canvas);
    window.addEventListener('resize', () => resize(canvas));

    // pointer orbit (drag) + pinch/scroll zoom stored back into UI via callback
    setupControls();

    return { gl, program };
  }

  // Adaptive render resolution keeps the raymarcher real-time on any GPU.
  // We cap the drawing buffer and, when FPS drops, we shrink it further.
  let renderScale = 1.0;
  function resize(c) {
    let dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    // Fall back to the window size if the canvas isn't laid out yet, and
    // never allow a 0-sized buffer (would silently render nothing).
    const cw = c.clientWidth  || window.innerWidth  || 1;
    const ch = c.clientHeight || window.innerHeight || 1;
    // cap total pixels
    let clientPx = cw * ch || 1;
    const maxPx = 640000;
    if (clientPx * dpr * dpr > maxPx) {
      dpr *= Math.sqrt(maxPx / (clientPx * dpr * dpr));
    }
    dpr *= renderScale;
    dpr = Math.max(dpr, 0.25);
    let w = Math.max(1, Math.floor(cw * dpr));
    let h = Math.max(1, Math.floor(ch * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }

  // called by the frame loop when FPS is low; clamps the internal scale
  function setQuality(fps) {
    let target = 1.0;
    if (fps < 45) target = 0.8;
    if (fps < 32) target = 0.62;
    if (fps < 22) target = 0.5;
    if (Math.abs(target - renderScale) > 0.05) {
      renderScale = target;
      resize(canvas);
    }
  }

  function setupControls() {
    let dragging = false, lx = 0, ly = 0;
    canvas.addEventListener('mousedown', (e) => { dragging = true; lx = e.clientX; ly = e.clientY; });
    window.addEventListener('mouseup', () => { dragging = false; });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      const ui = window.UI;
      const st = ui.state;
      // manual drag disables auto; sync the UI orbit to avoid a jump
      if (st.cameraAuto) { st.cameraOrbit = VIEW.orbit; st.cameraAuto = false; }
      st.cameraOrbit += dx * 0.006;
      VIEW.orbit = st.cameraOrbit;
      st.cameraPitch = clamp(st.cameraPitch + dy * 0.004, 0.05, 1.45);
      VIEW.pitch = st.cameraPitch;
      const s = document.getElementById('cam-pitch'); if (s) s.value = st.cameraPitch;
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const st = window.UI.state;
      st.cameraDist = clamp(st.cameraDist + e.deltaY * 0.003, 2.3, 12);
      VIEW.dist = st.cameraDist;
      const s = document.getElementById('cam-dist'); if (s) s.value = st.cameraDist;
    }, { passive: false });
  }

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  function setVec3(name, v) { gl.uniform3f(uniforms[name], v[0], v[1], v[2]); }
  function setVec2(name, v) { gl.uniform2f(uniforms[name], v[0], v[1]); }

  function render(now, p) {
    clock.t += 1 / 60;
    // Re-sync size each frame: if the canvas was 0-sized at init (e.g. the
    // preview iframe hadn't laid out yet) this recovers it once it has. resize
    // is a cheap no-op while the buffer already matches.
    resize(canvas);
    const w = canvas.width, h = canvas.height;
    // Clear to a visible sky colour so a failure is never a silent black box.
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.72, 0.78, 0.86, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(uniforms.uRes, w, h);
    gl.uniform1f(uniforms.uTime, now);

    // camera
    VIEW.target = [0, p.rockY || 0, 0];
    if (p.cameraAuto) {
      VIEW.orbit += 0.004;
      p.cameraOrbit = VIEW.orbit;
    }
    VIEW.orbit = p.cameraOrbit;
    VIEW.pitch = p.cameraPitch;
    VIEW.dist = p.cameraDist;
    const oy = Math.sin(VIEW.orbit), ox = Math.cos(VIEW.orbit);
    const cy = Math.cos(VIEW.pitch), sy = Math.sin(VIEW.pitch);
    const cam = [
      VIEW.target[0] + VIEW.dist * ox * cy,
      VIEW.target[1] + VIEW.dist * sy,
      VIEW.target[2] + VIEW.dist * oy * cy
    ];
    setVec3('uCamPos', cam);
    setVec3('uCamTarget', VIEW.target);
    gl.uniform1f(uniforms.uFov, VIEW.fov);

    // erosion + shape
    gl.uniform1f(uniforms.uE, p.erosion);
    gl.uniform1f(uniforms.uErodeType, p.erosErodeType);
    gl.uniform1f(uniforms.uShape, p.shape);
    gl.uniform1f(uniforms.uShapeRough, p.shapeRough);
    gl.uniform1f(uniforms.uSeed, p.seed);
    // sit the rock on the ground plane
    gl.uniform3f(uniforms.uRockPos, 0, p.rockY || 0, 0);

    // material
    setVec3('uC1', p.c1); setVec3('uC2', p.c2); setVec3('uC3', p.c3);
    gl.uniform1f(uniforms.uSpeck, p.speck);
    gl.uniform1f(uniforms.uBand, p.band);
    gl.uniform1f(uniforms.uBandFreq, p.bandFreq);
    gl.uniform1f(uniforms.uMoss, p.moss);
    gl.uniform1f(uniforms.uWeather, p.weather);
    setVec3('uWeatherColor', p.weatherColor);
    gl.uniform1f(uniforms.uDetailAmp, p.detailAmp);
    gl.uniform1f(uniforms.uDetailFreq, p.detailFreq);
    gl.uniform1f(uniforms.uCarveAmt, p.carveAmt);
    gl.uniform1f(uniforms.uCarveFreq, p.carveFreq);
    gl.uniform1f(uniforms.uStriAmt, p.striAmt);
    gl.uniform1f(uniforms.uStriFreq, p.striFreq);
    gl.uniform1f(uniforms.uShininess, p.shine);
    gl.uniform1f(uniforms.uSpecAmt, p.specAmt);

    // lights
    setVec3('uLightDir', nrm(ROCK_LIGHT.dir));
    setVec3('uLightColor', ROCK_LIGHT.color);
    setVec3('uSkyTop', ROCK_LIGHT.skyTop);
    setVec3('uSkyHorizon', ROCK_LIGHT.skyHorizon);
    setVec3('uSunColor', ROCK_LIGHT.sun);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  return { init, render, VIEW, setQuality };
})();

window.Renderer = Renderer;
