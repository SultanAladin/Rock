# Rock — Real-Time GPU Rock Erosion with SDFs

A browser app that renders realistic rocks as **signed distance fields (SDFs)** and
erodes them **in real time on the GPU**, driven by a **time slider** so you can
scrub a rock from fresh to fully weathered. It uses three.js-free, raw **WebGL 1**
(fragment-shader raymarching) so it runs everywhere with no build step.

Open `index.html` (or serve the folder) and it just runs.

---

## What it does

- **Realistic SDF raymarching** renderer (Inigo Quílez-style distance-field sphere
  tracing, soft shadows, ambient occlusion, fresnel, sky + sun lighting).
- **Multiple real erosion processes** that "ride the rock over time":
  1. **Abrasion** (sand / glacial wear) — smooths micro-roughness, light etch.
  2. **Hydraulic** (running water) — carves directional grooves/striations.
  3. **Chemical** (dissolution + tafoni) — honeycomb / karst pit dissolution.
  4. **Wind / Aeolian** — strong layered sandblasting striations.
  5. **Frost / freeze-thaw** — deep cellular jointing and cracks.
  6. **Combined** — a natural blended weathering profile.
- A **geologic-time scrubber** that animates erosion over a period (▶ play), and
  lets you **drag the slider to freeze/stop at any erosion state** you want.
- **Real rock colours** (researched values): Granite, Pink Granite, Basalt,
  Obsidian, Pumice, Sandstone, Red Sandstone, Limestone, Marble, Slate, Gneiss —
  each with its mineral speckle, banding, weathering stains and biological moss.
- A **shape generator** for basic rock silhouettes: Round Boulder, Boulder
  Cluster, Block/Crag, Slab/Ledge, Crystal Shard, Pebble/Cobble, Shelf/Strata,
  Porous Rock — with a random **seed** slider.
- Drag to orbit, scroll to zoom, and an adaptive-resolution/FPS system to keep it
  real-time on any GPU.

## How to run

Serve the folder and open the page (a static host avoids file:// shader nuances):

```bash
python3 -m http.server 8000 --bind 0.0.0.0
# open http://localhost:8000
```

Requires a browser with WebGL (Chrome, Edge, Firefox, Safari). No dependencies.

## Controls

| Control | Description |
| --- | --- |
| **Rock type** chips | Choose the rock (colour/material + its erosion behaviour). |
| **Shape generator** chips | Choose the base rock silhouette. |
| **Seed / variation** | Randomise the rock shape and mineral pattern. |
| **Erosion process** chips | Choose which erosion process wears the rock. |
| Micro roughness / detail freq / carve / striation | Fine-tune the surface. |
| **Timeline slider** (bottom) | Scrub erosion from 0 % (fresh) to 100 % (fully weathered). |
| **▶ / ❚❚** button | Auto-play the erosion over ~9 s; drag the slider to stop at a point. |
| **Drag on the canvas** | Orbit the camera. **Scroll** to zoom. |

## Architecture

```
index.html
  js/shaders.js     GLSL: vertex + fragment (SDF raymarcher, erosion, colour)
  js/materials.js   researched rock colour data + shape/erosion presets
  js/renderer.js    WebGL setup, uniforms, camera, adaptive resolution
  js/ui.js          control panel -> params
  js/main.js        boots the app, timeline scrubber, play/pause, FPS
```

### The erosion model

The rock is an analytic SDF. Each point is warped by 3D noise into an irregular
boulder, then carved by a weighted combination of three procedural patterns that
are **scaled by the erosion slider `uE`**:

- **Pits** — Voronoi F1 (honeycomb / tafoni / karst dissolution).
- **Cracks** — Voronoi F2−F1 difference (jointing / frost fracturing).
- **Striations** — directional FBM-modulated grooves (water / wind).

The same carve mask is reused in the colour shader so the colour of the holes
matches the geometry, and weathering stains + moss accumulate *in* those crevices
over time. Micro-roughness is worn away as `uE` rises, which is the abrasion term.

All of it is evaluated per-pixel on the GPU, so changing the slider costs no
recompilation — it is a single uniform.

### Colour research

Rock palettes are sampled from real geological colours: feldspar/quartz granites
(grey/pink/white-speckled), basalt near-black, obsidian glassy black with a
mahogany sheen, sandstone tan/cream with iron-oxide banding, limestone cream with
karst varnish, marble white with grey veining, slate dark foliated, gneiss
black/white banded. Each material also sets its carve intensity, striation,
specular and moss tendency to match the real rock's weathering character.
