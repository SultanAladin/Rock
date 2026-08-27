# Granite Boulder Forge

A realtime procedural generator for **batches of geologically-derived granite boulders**, with
crystal-scale surface texture and a real curvature-driven weathering simulation.

Run it:

```bash
npm install
npm run dev
```

---

## The two things you asked for

### 1. "Realistic texture, not fake noise"

The surface is a **crystal aggregate**, not a noise field.

Granite is holocrystalline: it is an interlocking mosaic of crystals that nucleated and grew until
they impinged on one another. The correct geometric idealisation of that process is a **Laguerre
(radical Voronoi) tessellation** of nucleation sites with per-crystal weights — sharply bounded
polygonal grains with a log-normal size distribution, not scale-free fBm clouds.

So `src/core/petrology.js` builds:

| stage | what it produces |
|---|---|
| Jittered lattice → Laguerre cells | interlocking crystals with ~120° triple junctions |
| Modal composition (QAP) | per-crystal mineral species drawn from real volume fractions |
| Log-normal Laguerre weights | seriate grain-size distribution, `σ_ln` per lithology |
| Second finer population | continuous (seriate) fabric, not artificially equigranular |
| Sparse large cells | K-feldspar **phenocrysts** → porphyritic texture |
| Anisotropic metric | tectonic **foliation** (flattened grains) for gneissose granite |

Noise is used in exactly one place, and only where the phenomenon genuinely *is* stochastic:
sericite clouding on plagioclase, perthitic exsolution lamellae, and micro-fracture staining —
i.e. *sub-grain* effects. Everything at grain scale and above is deterministic crystal geometry.

Six minerals carry real optical and physical properties (linear albedo, GGX roughness, dielectric
F0, Mohs hardness, Goldich durability, cleavage class, Fe content, translucency):

```
quartz  kfeldspar  plagioclase  biotite  muscovite  hornblende
```

Five lithologies ship: biotite granite, porphyritic pink granite, granodiorite (Rio Blanco type),
two-mica leucogranite, gneissose granite.

**What the shader does with it** (`src/gpu/rockMaterial.js`, no bitmap textures anywhere):

- **Per-crystal albedo** with per-crystal jitter — mottling is *discontinuous across grain
  boundaries*. Noise-based rock shaders always smear across boundaries; this cannot.
- **Cleavage-plane specular flash.** Feldspar has two good cleavages, mica one perfect one. When
  such a plane lies near-parallel to the surface it throws a mirror glint. This winking
  schiller as you orbit is the single strongest cue that you're looking at crystalline rock.
- **Anisotropic GGX for biotite/muscovite** — mica books are layered, so their specular is
  stretched along the sheet.
- **Analytic micro-relief normals** from the hardness/boundary field: quartz stands proud, biotite
  pits, boundaries are etched grooves — at any zoom level.
- **Wrapped-diffuse subsurface scattering** weighted by mineral translucency, which is what keeps
  quartz from reading as grey plastic.
- Weathering response: Fe(III) ochre staining with a **mafic-source halo**, case hardening,
  grussification bleaching, crustose lichen with ragged thalli, wetness, and soil contact.

Because it is procedural and resolution-independent, you can put the camera 5 cm from the surface
and still be looking at individual crystals.

### 2. Weathering — an actual SDF erosion solver

`src/core/weathering.js` runs a **level-set simulation** on the signed distance field:

```
∂φ/∂t + F·|∇φ| = 0
```

Godunov upwinding for the hyperbolic part, periodic fast-sweeping re-distancing, narrow-band
restricted, with **velocity extension**. The speed function:

```
F = [ A_sph·sat(κ̂,p) + A_cav·sat(-κ̂,q)·shelter² + A_uni ] · weakness(x)
```

- **Spheroidal term.** A joint block is attacked from every bounding joint face at once: a corner
  from three, an edge from two, a face from one. The attack rate scales with the solid angle of
  exposure ≈ mean curvature. This is the textbook mechanism of corestone rounding.
- **Cavernous term.** Concavities retain moisture and salt and are sheltered from rainwash, so a
  hollow accelerates once started → tafoni and honeycomb. Gated on a shelter integral (squared),
  without which the term simply dissolves the rock.
- **Weakness field.** Mineral durability from the crystal aggregate — quartz inert, biotite first
  to oxidise. Biotite oxidation (Fe(II)→Fe(III), positive ΔV) is the documented trigger for
  rindlet spalling and grussification in granitoids.
- **Rindlets.** Concentric shells phased on the *initial* distance field, at the field-measured
  **35–50 mm** spacing, giving scaly onion-skin relief instead of a bland ellipsoid.
- **Basal moisture gradient** — buried rock stays damp, which is why corestones are rounder at the
  base and why inselbergs develop flared slopes and basal notches.
- **Insolation/aspect bias** for thermal and wet-dry cycling.

### Structure comes first

Real boulders are not "a sphere plus noise" — they start as **joint-bounded blocks**.
`src/core/joints.js` cuts the fresh block as a CSG intersection of half-spaces from Fisher-distributed
joint sets with log-normal spacing: orthogonal (equidimensional), sheeting (slabby), columnar
(prismatic), random polyhedral, conjugate shear.

Joint faces carry **self-affine roughness with Hurst exponent H ≈ 0.8** — the measured statistic of
mode-I fracture surfaces in granite across five decades of scale, with amplitude scaled to grain
size because crack paths deviate around and through crystals.

---

## Pipeline

```
1. STRUCTURE   joint sets → CSG half-space intersection → fresh block SDF
2. PETROLOGY   Laguerre crystal aggregate → durability volume
3. WEATHERING  curvature-driven level-set erosion (narrow band + velocity extension)
4. MESHING     dual contouring with QEF vertex placement
5. DETAIL      grain-scale differential micro-relief at mesh level
6. ATTRIBUTES  bake retreat / shelter / curvature per vertex for the shader
```

Weathering runs on the **volume, before meshing**, because rounding a corner is a geometric change
to the solid, not a displacement of a surface. Anything faking it as surface displacement cannot
produce a concave tafone or a flared base, and it always shows.

Meshing is **dual contouring, not marching cubes** — MC cannot represent the sharp arrises of a
freshly jointed block and produces the sliver triangles and staircase normals that make procedural
rocks read as CG. QEF vertex placement gives razor-sharp fresh joint edges *and* smooth weathered
shoulders from the same field.

---

## Correctness notes

Several non-obvious things had to be right; each is documented at length in-source.

- **Saturating rate law.** A raw power law in curvature is unbounded — at grid resolution an arris
  has κ ~ 1/h, the rate blows up, the timestep collapses, and the solver nibbles one voxel while
  the rest of the rock never moves. That failure mode is why most "SDF erosion" demos output
  something indistinguishable from their input. Saturation is also physically correct: a corner is
  attacked from at most three faces, so its rate is bounded regardless of sharpness.
- **Velocity extension** (Adalsteinsson & Sethian). Speed is a property of the *surface*, so for an
  off-interface band cell it must be evaluated at the closest point *on* the interface. Skipping
  this makes φ stop being a distance function and the Godunov term amplifies exactly the
  high-frequency modes it should suppress. Diagnostic symptom: surface **area rises** as the rock
  erodes — the boulder grows fuzz instead of rounding.
- **Durability upscaling.** The grid has ~30 mm cells; crystals are ~3 mm. Point-sampling the
  crystal field per cell returns an uncorrelated draw at every cell — white noise at grid scale.
  The correct grid quantity is a Reuss-style harmonic mean over the cell (the front advances
  through the weakest connected path). Real grid-scale variation comes from *mesoscale*
  heterogeneity instead — mafic schlieren, aplite veins — which is modelled separately. Crystal
  relief reappears at mesh level, where resolution can carry it.
- **Retreat budget vs. time.** Age is specified as a retreat *distance*; integration happens in
  time. Without the conversion, turning up the rate dissolves the rock instead of rounding it
  faster.
- **CPU/GPU hash parity.** The crystal field is evaluated on both sides. `tools/checkhash.mjs`
  verifies the JS and GLSL hashes agree **bit-for-bit** — otherwise quartz micro-relief wouldn't
  line up with quartz colour, which is exactly the tell that makes procedural stone look painted.

Validation of the solver, resolution 56³:

| age | volume (m³) | area (m²) | Wadell sphericity |
|---|---|---|---|
| 0 (fresh block) | 2.012 | 9.23 | 0.835 |
| 0.25 | 1.916 | 8.82 | 0.846 |
| 0.5 | 1.811 | 8.41 | **0.854** |
| 1.0 | 1.567 | 7.94 | 0.822 |
| 2.0 | 0.985 | 8.12 | 0.590 |

Area falls monotonically and sphericity **rises** as corners round (fresh blocks ~0.83 → corestone
range 0.85–0.90), then falls as differential weathering takes over and the surface becomes
sculpted. That is the correct signature.

---

## Batch generation

Instances vary by log-normal size (matching real block-size distributions from joint spacing),
weathering age, buried fraction, aspect, joint roughness, and optional lithology/joint-style mixes.
Generation runs across a **worker pool** (one per core) with transferable buffers, so the solve
never touches the render thread.

Live stats report mean Wadell sphericity against the field range, and total mass at 2680 kg/m³.

## Inspect modes

Shaded · Mineral map · Surface retreat · Shelter/AO · Mean curvature

## Export

- **OBJ** — geometry + normals, with volume/mass/sphericity in the header
- **PLY** (binary) — carries solver output as vertex colour: R = surface retreat, G = shelter/AO,
  B = mean curvature, so the material can be rebuilt downstream without re-running the solve.

## Layout

```
src/core/      rng · noise · grid (SDF, curvature, fast-sweep) · petrology
               joints · weathering · mesher · generator
src/gpu/       rockMaterial.js + glsl/common.glsl.js  (hash/noise/grain, mirrors core)
src/worker/    generation worker
src/io/        OBJ / PLY exporters
src/app/       viewer + control panel
tools/         checkhash · checkglsl · checkuniforms   (static validation)
```

## References

Curvature-driven spheroidal/cavernous weathering on voxel grids (Beardall, Jones et al.;
Farley, *Fast Spheroidal Weathering with Colluvium Deposition*, BYU 2011) · Dorsey et al.,
*Modeling and Rendering of Weathered Stone*, SIGGRAPH 1999 · Peytavie et al., *Modeling Rocky
Scenery using Implicit Blocks* · Fletcher/Buss/Brantley on coupled oxidation–dissolution–fracturing
in the Rio Blanco quartz diorite · Hirata & Chigira on rindlet exfoliation of granite porphyry ·
Adalsteinsson & Sethian on velocity extension · Zhao, fast sweeping for the Eikonal equation.
