/*
 * materials.js — real rock colour data + per-rock shader parameters.
 *
 * Colours come from research on real rocks (granite, basalt, sandstone,
 * limestone, marble, slate, gneiss, obsidian). Each entry packages the mineral
 * palette plus the erosion behaviour that is characteristic of that rock.
 *
 * All colours are linear-ish RGB in [0,1] (the shader applies a gamma curve).
 */

/* Convert a CSS hex (#rrggbb) to a linear-ish [r,g,b] triplet. */
function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const n = parseInt(h, 16);
  return [
    ((n >> 16) & 255) / 255,
    ((n >> 8) & 255) / 255,
    (n & 255) / 255
  ];
}

/*
 * ROCK TYPES
 *   name      : display name
 *   type      : igneous / sedimentary / metamorphic / special
 *   c1,c2,c3  : base, variation, accent colours
 *   speck     : mineral speckle amount (0..1)
 *   band      : sedimentary/foliation banding amount (0..1)
 *   bandFreq  : band spatial frequency
 *   roughness : default macro/carve detail strength
 *   shine, specAmt : crystalline / glassy specular response
 *   detailAmp, detailFreq : micro roughness texture
 *   carveAmt, carveFreq   : how fiercely erosion carves this rock
 *   striAmt, striFreq     : striation strength/frequency
 *   erodeType : which erosion process it weathers by default
 *   weather   : weathering stain (iron oxide / varnish / salt)
 *   weatherColor
 *   moss      : biological weathering tendency
 */
const ROCK_TYPES = [
  {
    name: 'Granite',
    type: 'igneous',
    c1: '#c2c2c8',          // light gray feldspar
    c2: '#8c8b90',          // dark mica / hornblende
    c3: '#f0eee9',          // white quartz specks
    speck: 0.55, band: 0.0, bandFreq: 4.0,
    detailAmp: 0.11, detailFreq: 6.5,
    carveAmt: 0.18, carveFreq: 3.2,
    striAmt: 0.10, striFreq: 5.0,
    shine: 80.0, specAmt: 0.12,
    erodeType: 0, weather: 0.35, weatherColor: '#7e6852',
    moss: 0.30, rough: 0.5
  },
  {
    name: 'Pink Granite',
    type: 'igneous',
    c1: '#d9b8a6',
    c2: '#a9866f',
    c3: '#f2e8dd',
    speck: 0.55, band: 0.0, bandFreq: 4.0,
    detailAmp: 0.11, detailFreq: 6.5,
    carveAmt: 0.20, carveFreq: 3.2,
    striAmt: 0.10, striFreq: 5.0,
    shine: 80.0, specAmt: 0.12,
    erodeType: 0, weather: 0.40, weatherColor: '#9c6b52',
    moss: 0.30, rough: 0.5
  },
  {
    name: 'Basalt',
    type: 'igneous',
    c1: '#2b2b30',
    c2: '#3a3a41',
    c3: '#5a5a63',
    speck: 0.30, band: 0.0, bandFreq: 4.0,
    detailAmp: 0.10, detailFreq: 7.0,
    carveAmt: 0.16, carveFreq: 2.6,
    striAmt: 0.12, striFreq: 3.5,
    shine: 60.0, specAmt: 0.10,
    erodeType: 4, weather: 0.35, weatherColor: '#4a4a40',
    moss: 0.40, rough: 0.4
  },
  {
    name: 'Obsidian',
    type: 'igneous',
    c1: '#101015',
    c2: '#1d1d24',
    c3: '#6d2f2f',          // subtle mahogany obsidian sheen
    speck: 0.12, band: 0.0, bandFreq: 4.0,
    detailAmp: 0.05, detailFreq: 6.0,
    carveAmt: 0.12, carveFreq: 2.4,
    striAmt: 0.06, striFreq: 4.0,
    shine: 520.0, specAmt: 0.55,   // glassy
    erodeType: 4, weather: 0.15, weatherColor: '#26262b',
    moss: 0.10, rough: 0.15
  },
  {
    name: 'Pumice',
    type: 'igneous',
    c1: '#c6c2b4',
    c2: '#b2ada0',
    c3: '#e7e3d6',
    speck: 0.25, band: 0.0, bandFreq: 4.0,
    detailAmp: 0.16, detailFreq: 5.5,
    carveAmt: 0.40, carveFreq: 5.0,   // very porous / vesicular
    striAmt: 0.06, striFreq: 5.0,
    shine: 8.0, specAmt: 0.02,
    erodeType: 2, weather: 0.25, weatherColor: '#a89a84',
    moss: 0.15, rough: 0.7
  },
  {
    name: 'Sandstone',
    type: 'sedimentary',
    c1: '#c2a06c',          // tan
    c2: '#a9834d',          // darker band
    c3: '#e0c49a',          // pale band
    speck: 0.35, band: 0.9, bandFreq: 5.5,
    detailAmp: 0.10, detailFreq: 8.0,
    carveAmt: 0.26, carveFreq: 3.4,
    striAmt: 0.45, striFreq: 6.0,
    shine: 20.0, specAmt: 0.04,
    erodeType: 2, weather: 0.55, weatherColor: '#9c5f34', // iron oxide
    moss: 0.35, rough: 0.5
  },
  {
    name: 'Red Sandstone',
    type: 'sedimentary',
    c1: '#b07a4e',
    c2: '#8f5a36',
    c3: '#d0a075',
    speck: 0.35, band: 0.9, bandFreq: 4.5,
    detailAmp: 0.10, detailFreq: 8.0,
    carveAmt: 0.28, carveFreq: 3.4,
    striAmt: 0.45, striFreq: 5.5,
    shine: 20.0, specAmt: 0.04,
    erodeType: 2, weather: 0.55, weatherColor: '#7a3f24',
    moss: 0.30, rough: 0.5
  },
  {
    name: 'Limestone',
    type: 'sedimentary',
    c1: '#d8d2c2',          // cream
    c2: '#bcb4a0',
    c3: '#ede8da',
    speck: 0.25, band: 0.35, bandFreq: 3.5,
    detailAmp: 0.09, detailFreq: 6.0,
    carveAmt: 0.34, carveFreq: 3.8,
    striAmt: 0.12, striFreq: 5.0,
    shine: 30.0, specAmt: 0.06,
    erodeType: 2, weather: 0.45, weatherColor: '#a89a82', // karst varnish
    moss: 0.40, rough: 0.4
  },
  {
    name: 'Marble',
    type: 'metamorphic',
    c1: '#eae6dd',          // white calcite
    c2: '#cfc8bc',
    c3: '#8a8f8c',          // gray veining
    speck: 0.12, band: 0.55, bandFreq: 6.0,
    detailAmp: 0.05, detailFreq: 6.5,
    carveAmt: 0.22, carveFreq: 3.0,
    striAmt: 0.12, striFreq: 5.0,
    shine: 140.0, specAmt: 0.22,
    erodeType: 2, weather: 0.30, weatherColor: '#a89b8c',
    moss: 0.25, rough: 0.3
  },
  {
    name: 'Slate',
    type: 'metamorphic',
    c1: '#4a4a50',
    c2: '#33333a',
    c3: '#6b6b74',
    speck: 0.20, band: 0.9, bandFreq: 7.0,   // intense foliation
    detailAmp: 0.06, detailFreq: 7.0,
    carveAmt: 0.18, carveFreq: 2.4,
    striAmt: 0.30, striFreq: 5.0,
    shine: 50.0, specAmt: 0.10,
    erodeType: 4, weather: 0.35, weatherColor: '#565e57',
    moss: 0.40, rough: 0.3
  },
  {
    name: 'Gneiss',
    type: 'metamorphic',
    c1: '#d9d6cf',
    c2: '#5c5c62',          // dark band
    c3: '#efede8',          // light band
    speck: 0.30, band: 1.0, bandFreq: 3.0,   // bold bands
    detailAmp: 0.09, detailFreq: 6.0,
    carveAmt: 0.18, carveFreq: 3.0,
    striAmt: 0.15, striFreq: 5.0,
    shine: 70.0, specAmt: 0.12,
    erodeType: 0, weather: 0.35, weatherColor: '#8a7d6a',
    moss: 0.25, rough: 0.45
  }
];

/*
 * SHAPE PRESETS for the generator. Each yields the shader uniforms:
 *   uShape (which primitive), uShapeRough (macro irregularity),
 *   plus an autofit scale so generated shapes sit on the ground nicely.
 */
/*
 * yOff lowers/raises the shape so its base sits on the ground (y = -1.05).
 * Values are tuned per primitive; rocks look natural with a slight embed.
 */
const SHAPE_PRESETS = [
  { id: 'round',    name: 'Round Boulder',   shape: 0, rough: 0.30, yOff: -0.12 },
  { id: 'cluster',  name: 'Boulder Cluster', shape: 6, rough: 0.55, yOff:  0.15 },
  { id: 'block',    name: 'Block / Crag',    shape: 1, rough: 0.30, yOff: -0.20 },
  { id: 'slab',     name: 'Slab / Ledge',    shape: 2, rough: 0.28, yOff: -0.60 },
  { id: 'crystal',  name: 'Crystal Shard',   shape: 3, rough: 0.18, yOff:  0.00 },
  { id: 'pebble',   name: 'Pebble / Cobble', shape: 4, rough: 0.24, yOff: -0.32 },
  { id: 'shelf',    name: 'Shelf / Strata',  shape: 5, rough: 0.30, yOff: -0.40 },
  { id: 'pumice',   name: 'Porous Rock',     shape: 6, rough: 0.75, yOff:  0.15 }
];

/*
 * EROSION PROCESSES (the "different types of rock erosion that rode the rock
 * over time" that the user asked about). Each maps to a shader index and
 * pops sliders that are meaningful for that process.
 */
const EROSION_TYPES = [
  { id: 'abrasion',   name: 'Abrasion (sand/glacial wear)',     idx: 0 },
  { id: 'hydraulic',  name: 'Hydraulic (running water)',        idx: 1 },
  { id: 'chemical',   name: 'Chemical (dissolution / tafoni)',  idx: 2 },
  { id: 'wind',       name: 'Wind / Aeolian (sandblasting)',    idx: 3 },
  { id: 'frost',      name: 'Frost / Freeze-thaw fracturing',   idx: 4 },
  { id: 'combined',   name: 'Combined (natural weathering)',    idx: 5 }
];

window.RockMaterials = { ROCK_TYPES, SHAPE_PRESETS, EROSION_TYPES, hexToRgb };
