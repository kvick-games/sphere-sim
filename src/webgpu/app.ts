import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

declare global {
  interface Window {
    __sphereSimBackend?: string;
  }
}

type NumericParamKey = { [K in keyof Params]: Params[K] extends number ? K : never }[keyof Params];
type TextureDebugKey = 'base' | 'emissive' | 'geo' | 'topo';

interface Params {
  speed: number;
  coriolis: number;
  heating: number;
  solar: number;
  cool: number;
  greenhouse: number;
  oceanInertia: number;
  curl: number;
  velDissipation: number;
  denDissipation: number;
  pressureIters: number;
  brush: number;
  emitters: boolean;
  spin: number;
  dayNight: boolean;
  viewMode: number;
  overlay: number;
  specular: number;
  atmosphere: number;
}

interface GpuField {
  read: GPUTexture;
  write: GPUTexture;
  readView: GPUTextureView;
  writeView: GPUTextureView;
}

interface LoadedTexture {
  texture: GPUTexture;
  view: GPUTextureView;
  source: CanvasImageSource;
}

interface Splat {
  dir: THREE.Vector3;
  delta: THREE.Vector3;
  color: THREE.Vector3;
}

type TouchMode = 'orbit' | 'paint';

const FACE = 384;
const DIAG_FACE = 64;
const DIAG_SAMPLES = DIAG_FACE * DIAG_FACE * 6;
const WORKGROUP = 8;
const VIEW: Record<string, number> = { smoke: 0, temperature: 1, wind: 2, vorticity: 3, pressure: 4, climate: 5, anomaly: 6 };
const DEFAULT_TIME_SCALE = 0.1;
const EARTH_RADIUS = 1.0;
const EARTH_RADIUS_KM = 6371.0;
const EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0;
const SUN_WORLD = new THREE.Vector3(0.6, 0.5, 0.8).normalize();
const MOON_RADIUS_KM = 1737.4;
const MOON_SEMI_MAJOR_AXIS_KM = 384400.0;
const SUN_RADIUS_KM = 696340.0;
const ASTRONOMICAL_UNIT_KM = 149597870.7;
const EARTH_SOLAR_DAY_SECONDS = 86400.0;
const EARTH_SIDEREAL_DAY_SECONDS = 0.99726968 * EARTH_SOLAR_DAY_SECONDS;
const EARTH_ROTATION_RATE_RAD_PER_SECOND = Math.PI * 2 / EARTH_SIDEREAL_DAY_SECONDS;
const MOON_SIDEREAL_ORBIT_SECONDS = 27.321661 * EARTH_SOLAR_DAY_SECONDS;
const PHYSICAL_SECONDS_PER_REAL_SECOND = 3600.0;
const MODEL_VELOCITY_UNIT_MPS = 50.0;
const CORIOLIS_OMEGA_CODE = EARTH_ROTATION_RATE_RAD_PER_SECOND * EARTH_RADIUS_M / MODEL_VELOCITY_UNIT_MPS;
const MOON_RADIUS = (MOON_RADIUS_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;
const MOON_ORBIT_A = (MOON_SEMI_MAJOR_AXIS_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;
const MOON_ORBIT_E = 0.0549;
const MOON_ORBIT_INCLINATION = THREE.MathUtils.degToRad(5.145);
const MOON_INITIAL_ANOMALY = 4.6;
const SUN_RADIUS = (SUN_RADIUS_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;
const SUN_DISTANCE = (ASTRONOMICAL_UNIT_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;

const TEX_URLS = {
  earth: 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
  night: 'https://assets.science.nasa.gov/content/dam/science/esd/eo/images/imagerecords/144000/144897/BlackMarble_2016_01deg_gray.jpg',
  topo: 'https://unpkg.com/three-globe/example/img/earth-topology.png',
};

const params: Params = {
  speed: DEFAULT_TIME_SCALE,
  coriolis: 1.0,
  heating: 6.0,
  solar: 62,
  cool: 40,
  greenhouse: 18,
  oceanInertia: 7,
  curl: 6,
  velDissipation: 0.12,
  denDissipation: 0.9,
  pressureIters: 16,
  brush: 1.2,
  emitters: false,
  spin: 1.0,
  dayNight: true,
  viewMode: VIEW.temperature,
  overlay: 0.45,
  specular: 3.0,
  atmosphere: 16,
};

const SIM_WGSL = /* wgsl */`
struct SimParams {
  simInfo: vec4<f32>,
  time: vec4<f32>,
  sunDir: vec4<f32>,
  brushDir: vec4<f32>,
  brushValue: vec4<f32>,
  brushMeta: vec4<f32>,
  controls: vec4<f32>,
  forces: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SimParams;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var geoTex: texture_2d<f32>;
@group(0) @binding(3) var topoTex: texture_2d<f32>;
@group(0) @binding(4) var velRead: texture_2d_array<f32>;
@group(0) @binding(5) var dyeRead: texture_2d_array<f32>;
@group(0) @binding(6) var tempRead: texture_2d_array<f32>;
@group(0) @binding(7) var pressRead: texture_2d_array<f32>;
@group(0) @binding(8) var velWrite: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(9) var dyeWrite: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(10) var tempWrite: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(11) var pressWrite: texture_storage_2d_array<rgba16float, write>;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn face_dir(face: u32, xy: vec2<u32>) -> vec3<f32> {
  let size = u.simInfo.x;
  let p = (vec2<f32>(xy) + vec2<f32>(0.5)) / size * 2.0 - vec2<f32>(1.0);
  let x = p.x;
  let y = p.y;
  var d: vec3<f32>;
  if (face == 0u) {
    d = vec3<f32>(1.0, -y, -x);
  } else if (face == 1u) {
    d = vec3<f32>(-1.0, -y, x);
  } else if (face == 2u) {
    d = vec3<f32>(x, 1.0, y);
  } else if (face == 3u) {
    d = vec3<f32>(x, -1.0, -y);
  } else if (face == 4u) {
    d = vec3<f32>(x, -y, 1.0);
  } else {
    d = vec3<f32>(-x, -y, -1.0);
  }
  return normalize(d);
}

fn dir_to_face_uv(n: vec3<f32>) -> vec3<f32> {
  let ax = abs(n.x);
  let ay = abs(n.y);
  let az = abs(n.z);
  var face = 0.0;
  var uv = vec2<f32>(0.0);
  if (ax >= ay && ax >= az) {
    if (n.x >= 0.0) {
      face = 0.0;
      uv = vec2<f32>(-n.z / ax, -n.y / ax);
    } else {
      face = 1.0;
      uv = vec2<f32>(n.z / ax, -n.y / ax);
    }
  } else if (ay >= ax && ay >= az) {
    if (n.y >= 0.0) {
      face = 2.0;
      uv = vec2<f32>(n.x / ay, n.z / ay);
    } else {
      face = 3.0;
      uv = vec2<f32>(n.x / ay, -n.z / ay);
    }
  } else if (n.z >= 0.0) {
    face = 4.0;
    uv = vec2<f32>(n.x / az, -n.y / az);
  } else {
    face = 5.0;
    uv = vec2<f32>(-n.x / az, -n.y / az);
  }
  return vec3<f32>(uv * 0.5 + vec2<f32>(0.5), face);
}

fn sample_cube(tex: texture_2d_array<f32>, n: vec3<f32>) -> vec4<f32> {
  let fuv = dir_to_face_uv(normalize(n));
  return textureSampleLevel(tex, linearSampler, fuv.xy, i32(fuv.z), 0.0);
}

fn dir_to_equirect(n: vec3<f32>) -> vec2<f32> {
  var phi = atan2(n.z, -n.x);
  var uu = phi / (2.0 * 3.141592653589793);
  if (uu < 0.0) { uu = uu + 1.0; }
  let vv = 1.0 - acos(clamp(n.y, -1.0, 1.0)) / 3.141592653589793;
  return vec2<f32>(uu, vv);
}

fn clim_baseline(n: vec3<f32>) -> f32 {
  let uv = dir_to_equirect(n);
  let lat = asin(clamp(n.y, -1.0, 1.0));
  let c = max(cos(lat), 0.0);
  var t = 0.04 + 0.78 * pow(c, 1.35);
  let elev = textureSampleLevel(topoTex, linearSampler, uv, 0.0).r;
  t = t - 0.65 * elev;
  let g = textureSampleLevel(geoTex, linearSampler, uv, 0.0).rgb;
  let ocean = smoothstep(0.015, 0.12, g.b - max(g.r, g.g));
  let oceanClimate = 0.14 + 0.56 * pow(c, 1.05);
  t = mix(t, mix(t, oceanClimate, 0.35), ocean);
  let ice = smoothstep(0.55, 0.85, min(g.r, min(g.g, g.b)));
  t = t - ice * 0.18;
  return clamp(t, 0.0, 1.0);
}

fn tbasis(n: vec3<f32>) -> mat2x3<f32> {
  let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
  let t1 = normalize(cross(up, n));
  let t2 = cross(n, t1);
  return mat2x3<f32>(t1, t2);
}

fn daily_mean(n: vec3<f32>, sun: vec3<f32>) -> f32 {
  let lat = asin(clamp(n.y, -1.0, 1.0));
  let decl = asin(clamp(sun.y, -1.0, 1.0));
  let sinLat = sin(lat);
  let cosLat = max(abs(cos(lat)), 0.0001);
  let sinDec = sin(decl);
  let cosDec = max(abs(cos(decl)), 0.0001);
  let x = -(sinLat * sinDec) / (cosLat * cosDec);
  let h0 = acos(clamp(x, -1.0, 1.0));
  return max((h0 * sinLat * sinDec + cosLat * cosDec * sin(h0)) / 3.141592653589793, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = u32(u.simInfo.x);
  if (gid.x >= size || gid.y >= size || gid.z >= 6u) { return; }
  let n = face_dir(gid.z, gid.xy);
  let temp = clamp(235.0 + 85.0 * clim_baseline(n), 185.0, 330.0);
  textureStore(velWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(0.0, 0.0, 0.0, 1.0));
  textureStore(dyeWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(0.0, 0.0, 0.0, 1.0));
  textureStore(tempWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(temp, 0.0, 0.0, 1.0));
  textureStore(pressWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = u32(u.simInfo.x);
  if (gid.x >= size || gid.y >= size || gid.z >= 6u) { return; }

  let n = face_dir(gid.z, gid.xy);
  let h = u.simInfo.y;
  let advectDt = u.simInfo.z;
  let thermalDt = u.simInfo.w;
  let basis = tbasis(n);
  let t1 = basis[0];
  let t2 = basis[1];
  let sun = normalize(u.sunDir.xyz);

  var vel = textureLoad(velRead, vec2<i32>(gid.xy), i32(gid.z), 0).xyz;
  let back = normalize(n - advectDt * vel);
  vel = sample_cube(velRead, back).xyz / (1.0 + u.forces.w * advectDt);
  var dye = sample_cube(dyeRead, back).xyz / (1.0 + u.brushMeta.w * advectDt);
  var temp = sample_cube(tempRead, back).x;

  let tp1 = sample_cube(tempRead, normalize(n + h * t1)).x;
  let tm1 = sample_cube(tempRead, normalize(n - h * t1)).x;
  let tp2 = sample_cube(tempRead, normalize(n + h * t2)).x;
  let tm2 = sample_cube(tempRead, normalize(n - h * t2)).x;
  let gradT = (((tp1 - tm1) / (2.0 * h)) * t1 + ((tp2 - tm2) / (2.0 * h)) * t2) / 80.0;
  let thermal = u.forces.y * cross(n, gradT);
  let coriolis = -2.0 * u.forces.x * n.y * cross(n, vel);
  vel = vel + (thermal + coriolis) * advectDt;

  let g = textureSampleLevel(geoTex, linearSampler, dir_to_equirect(n), 0.0).rgb;
  let ocean = smoothstep(0.015, 0.12, g.b - max(g.r, g.g));
  let ice = smoothstep(0.55, 0.85, min(g.r, min(g.g, g.b)));
  let annual = 235.0 + 85.0 * clim_baseline(n);
  let instantMu = max(dot(n, sun), 0.0);
  let meanMu = daily_mean(n, sun);
  let solarScale = u.controls.x / 62.0;
  let greenhouseBias = (u.controls.z - 18.0) * 0.35;
  var diurnalAmp = mix(16.0, 3.0, ocean);
  diurnalAmp = mix(diurnalAmp, 5.0, ice);
  let diurnal = select(0.0, instantMu - meanMu, u.time.z > 0.5);
  let thermalTarget = annual + greenhouseBias + diurnalAmp * solarScale * diurnal;
  let coolScale = max(u.controls.y / 40.0, 0.08);
  var tauDays = mix(0.85, max(u.controls.w, 1.0) * 3.2, ocean);
  tauDays = mix(tauDays, tauDays * 1.8, ice) / coolScale;
  let alpha = 1.0 - exp(-thermalDt / max(tauDays, 0.001));
  temp = temp + (thermalTarget - temp) * alpha;
  let lap = tp1 + tm1 + tp2 + tm2 - 4.0 * temp;
  temp = temp + lap * thermalDt * mix(0.20, 0.55, ocean);

  if (u.brushMeta.x > 0.5) {
    let ang = acos(clamp(dot(n, normalize(u.brushDir.xyz)), -1.0, 1.0));
    let fall = exp(-(ang * ang) / max(u.brushMeta.y, 0.00001));
    vel = vel + fall * u.brushValue.xyz;
    dye = dye + fall * u.brushValue.www * vec3<f32>(u.brushMeta.z, 0.45, 0.95);
    temp = temp + fall * 18.0;
  }

  vel = vel - dot(vel, n) * n;
  let sp = length(vel);
  if (sp > 18.0) { vel = vel * (18.0 / sp); }

  let div = dot(sample_cube(velRead, normalize(n + h * t1)).xyz - sample_cube(velRead, normalize(n - h * t1)).xyz, t1) / (2.0 * h)
          + dot(sample_cube(velRead, normalize(n + h * t2)).xyz - sample_cube(velRead, normalize(n - h * t2)).xyz, t2) / (2.0 * h);
  let curl = dot(sample_cube(velRead, normalize(n + h * t1)).xyz - sample_cube(velRead, normalize(n - h * t1)).xyz, t2) / (2.0 * h)
           - dot(sample_cube(velRead, normalize(n + h * t2)).xyz - sample_cube(velRead, normalize(n - h * t2)).xyz, t1) / (2.0 * h);
  let pressure = mix(textureLoad(pressRead, vec2<i32>(gid.xy), i32(gid.z), 0).x, div, 0.18);

  textureStore(velWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(vel, 1.0));
  textureStore(dyeWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(max(dye, vec3<f32>(0.0)), 1.0));
  textureStore(tempWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(clamp(temp, 120.0, 360.0), curl, 0.0, 1.0));
  textureStore(pressWrite, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(pressure, div, curl, 1.0));
}

@group(1) @binding(0) var<storage, read_write> diagOut: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn diagnostics(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= 64u || gid.y >= 64u || gid.z >= 6u) { return; }
  let n = face_dir(gid.z, gid.xy);
  let temp = textureLoad(tempRead, vec2<i32>(gid.xy * 6u), i32(gid.z), 0).x;
  let base = 235.0 + 85.0 * clim_baseline(n);
  let idx = gid.z * 64u * 64u + gid.y * 64u + gid.x;
  diagOut[idx] = vec4<f32>(temp, base, temp - base, 1.0);
}
`;

const RENDER_WGSL = /* wgsl */`
struct RenderParams {
  invViewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  sunWorld: vec4<f32>,
  sunObj: vec4<f32>,
  moonWorld: vec4<f32>,
  renderMeta: vec4<f32>,
  visual: vec4<f32>,
  flags: vec4<f32>,
};

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: RenderParams;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var baseTex: texture_2d<f32>;
@group(0) @binding(3) var nightTex: texture_2d<f32>;
@group(0) @binding(4) var geoTex: texture_2d<f32>;
@group(0) @binding(5) var topoTex: texture_2d<f32>;
@group(0) @binding(6) var velTex: texture_2d_array<f32>;
@group(0) @binding(7) var dyeTex: texture_2d_array<f32>;
@group(0) @binding(8) var tempTex: texture_2d_array<f32>;
@group(0) @binding(9) var pressTex: texture_2d_array<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(3.0, 1.0),
    vec2<f32>(-1.0, 1.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(p[vi], 0.0, 1.0);
  out.uv = p[vi] * 0.5 + vec2<f32>(0.5);
  return out;
}

fn rot_y(v: vec3<f32>, a: f32) -> vec3<f32> {
  let c = cos(a);
  let s = sin(a);
  return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn dir_to_face_uv(n: vec3<f32>) -> vec3<f32> {
  let ax = abs(n.x);
  let ay = abs(n.y);
  let az = abs(n.z);
  var face = 0.0;
  var uv = vec2<f32>(0.0);
  if (ax >= ay && ax >= az) {
    if (n.x >= 0.0) { face = 0.0; uv = vec2<f32>(-n.z / ax, -n.y / ax); }
    else { face = 1.0; uv = vec2<f32>(n.z / ax, -n.y / ax); }
  } else if (ay >= ax && ay >= az) {
    if (n.y >= 0.0) { face = 2.0; uv = vec2<f32>(n.x / ay, n.z / ay); }
    else { face = 3.0; uv = vec2<f32>(n.x / ay, -n.z / ay); }
  } else if (n.z >= 0.0) {
    face = 4.0; uv = vec2<f32>(n.x / az, -n.y / az);
  } else {
    face = 5.0; uv = vec2<f32>(-n.x / az, -n.y / az);
  }
  return vec3<f32>(uv * 0.5 + vec2<f32>(0.5), face);
}

fn sample_cube(tex: texture_2d_array<f32>, n: vec3<f32>) -> vec4<f32> {
  let fuv = dir_to_face_uv(normalize(n));
  return textureSampleLevel(tex, linearSampler, fuv.xy, i32(fuv.z), 0.0);
}

fn dir_to_equirect(n: vec3<f32>) -> vec2<f32> {
  var phi = atan2(n.z, -n.x);
  var uu = phi / 6.283185307179586;
  if (uu < 0.0) { uu = uu + 1.0; }
  return vec2<f32>(uu, 1.0 - acos(clamp(n.y, -1.0, 1.0)) / 3.141592653589793);
}

fn clim_baseline(n: vec3<f32>) -> f32 {
  let uv = dir_to_equirect(n);
  let lat = asin(clamp(n.y, -1.0, 1.0));
  let c = max(cos(lat), 0.0);
  var t = 0.04 + 0.78 * pow(c, 1.35);
  let elev = textureSampleLevel(topoTex, linearSampler, uv, 0.0).r;
  t = t - 0.65 * elev;
  let g = textureSampleLevel(geoTex, linearSampler, uv, 0.0).rgb;
  let ocean = smoothstep(0.015, 0.12, g.b - max(g.r, g.g));
  let oceanClimate = 0.14 + 0.56 * pow(c, 1.05);
  t = mix(t, mix(t, oceanClimate, 0.35), ocean);
  let ice = smoothstep(0.55, 0.85, min(g.r, min(g.g, g.b)));
  return clamp(t - ice * 0.18, 0.0, 1.0);
}

fn cmap_temp(x0: f32) -> vec3<f32> {
  let x = clamp(x0, 0.0, 1.0);
  let c0 = vec3<f32>(0.05, 0.18, 0.55);
  let c1 = vec3<f32>(0.10, 0.55, 0.85);
  let c2 = vec3<f32>(0.95, 0.93, 0.70);
  let c3 = vec3<f32>(0.92, 0.45, 0.12);
  let c4 = vec3<f32>(0.75, 0.06, 0.06);
  if (x < 0.25) { return mix(c0, c1, x / 0.25); }
  if (x < 0.50) { return mix(c1, c2, (x - 0.25) / 0.25); }
  if (x < 0.75) { return mix(c2, c3, (x - 0.50) / 0.25); }
  return mix(c3, c4, (x - 0.75) / 0.25);
}

fn cmap_div(x0: f32) -> vec3<f32> {
  let x = clamp(x0, 0.0, 1.0);
  let neg = vec3<f32>(0.15, 0.40, 0.90);
  let mid = vec3<f32>(0.04, 0.05, 0.07);
  let pos = vec3<f32>(0.95, 0.35, 0.18);
  if (x < 0.5) { return mix(neg, mid, x * 2.0); }
  return mix(mid, pos, (x - 0.5) * 2.0);
}

fn cmap_speed(x0: f32) -> vec3<f32> {
  let x = clamp(x0, 0.0, 1.0);
  let a = vec3<f32>(0.02, 0.03, 0.09);
  let b = vec3<f32>(0.10, 0.40, 0.60);
  let c = vec3<f32>(0.40, 0.85, 0.95);
  let d = vec3<f32>(1.0, 1.0, 0.92);
  if (x < 0.4) { return mix(a, b, x / 0.4); }
  if (x < 0.75) { return mix(b, c, (x - 0.4) / 0.35); }
  return mix(c, d, (x - 0.75) / 0.25);
}

fn hash13(p: vec3<f32>) -> f32 {
  let q = fract(p * 0.1031);
  let d = dot(q, q.yzx + vec3<f32>(33.33));
  let r = q + d;
  return fract((r.x + r.y) * r.z);
}

fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> vec2<f32> {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - radius * radius;
  let h = b * b - c;
  if (h < 0.0) { return vec2<f32>(-1.0, -1.0); }
  let s = sqrt(h);
  return vec2<f32>(-b - s, -b + s);
}

fn city_light(uv: vec2<f32>) -> vec3<f32> {
  let raw = textureSampleLevel(nightTex, linearSampler, vec2<f32>(fract(uv.x), clamp(uv.y, 0.0, 1.0)), 0.0).rgb;
  let radiance = max(raw.r, max(raw.g, raw.b));
  var city = max(radiance - 0.018, 0.0) / 0.982;
  city = pow(clamp(city, 0.0, 1.0), 1.35) * smoothstep(0.018, 0.08, radiance);
  return vec3<f32>(1.0, 0.64, 0.34) * city * 1.85;
}

fn field_color(dir: vec3<f32>, uv: vec2<f32>, sunAmount: f32, viewDir: vec3<f32>) -> vec3<f32> {
  let dayNight = u.visual.x;
  let viewMode = u.renderMeta.z;
  let overlay = u.renderMeta.w;
  let lit = mix(1.0, clamp(sunAmount, 0.0, 1.0), dayNight);
  var earth = vec3<f32>(0.012, 0.018, 0.028);
  if (u.visual.w > 0.5) {
    let dayTex = pow(textureSampleLevel(baseTex, linearSampler, uv, 0.0).rgb, vec3<f32>(2.2));
    let nightAlbedo = dayTex * vec3<f32>(0.018, 0.023, 0.032) * dayNight * (1.0 - lit);
    earth = dayTex * (1.28 * lit) + nightAlbedo + city_light(uv) * dayNight * (1.0 - smoothstep(-0.14, 0.07, sunAmount));
  }

  let geo = textureSampleLevel(geoTex, linearSampler, uv, 0.0).rgb;
  let ocean = smoothstep(0.015, 0.12, geo.b - max(geo.r, geo.g));
  let halfDir = normalize(normalize(u.sunObj.xyz) + viewDir);
  let ndoth = max(dot(dir, halfDir), 0.0);
  let spec = pow(ndoth, mix(30.0, 780.0, ocean)) * max(sunAmount, 0.0) * u.visual.y * (0.03 + ocean);
  earth = earth + vec3<f32>(1.0, 0.95, 0.84) * spec;

  var fc = vec3<f32>(0.0);
  var inten = overlay;
  if (viewMode < 0.5) {
    let smoke = max(sample_cube(dyeTex, dir).rgb, vec3<f32>(0.0));
    return earth + smoke * (0.9 + 0.6 * clamp(length(smoke), 0.0, 1.0));
  } else if (viewMode < 1.5) {
    let tk = sample_cube(tempTex, dir).x;
    fc = cmap_temp((tk - 245.0) / 65.0);
    inten = max(inten, 0.68);
  } else if (viewMode < 2.5) {
    let s = length(sample_cube(velTex, dir).xyz) * 0.55;
    fc = cmap_speed(s);
    inten = inten * clamp(s, 0.0, 1.0);
  } else if (viewMode < 3.5) {
    let w = sample_cube(tempTex, dir).y * 0.06;
    fc = cmap_div(0.5 + 0.5 * w);
    inten = inten * clamp(abs(w), 0.0, 1.0);
  } else if (viewMode < 4.5) {
    let p = sample_cube(pressTex, dir).x * 1.6;
    fc = cmap_div(0.5 + 0.5 * p);
    inten = inten * clamp(abs(p), 0.0, 1.0);
  } else if (viewMode < 5.5) {
    fc = cmap_temp(clim_baseline(dir));
  } else {
    let tk = sample_cube(tempTex, dir).x;
    let base = 235.0 + 85.0 * clim_baseline(dir);
    fc = cmap_div(0.5 + (tk - base) / 48.0);
    inten = max(inten, 0.74);
  }
  if (viewMode < 1.5 || viewMode > 5.5) {
    return mix(earth, fc, inten);
  }
  return mix(earth, fc * lit, inten * mix(1.0, smoothstep(-0.05, 0.22, sunAmount), dayNight));
}

fn scene_color(uvIn: vec2<f32>, unwrap: bool) -> vec4<f32> {
  var rd: vec3<f32>;
  var ro = u.cameraPos.xyz;
  if (unwrap) {
    let phi = uvIn.x * 6.283185307179586;
    let theta = (1.0 - uvIn.y) * 3.141592653589793;
    let st = sin(theta);
    let dir = vec3<f32>(-cos(phi) * st, cos(theta), sin(phi) * st);
    let sun = dot(dir, normalize(u.sunObj.xyz));
    return vec4<f32>(field_color(dir, uvIn, sun, vec3<f32>(0.0, 0.0, 1.0)), 1.0);
  }

  let ndc = uvIn * 2.0 - vec2<f32>(1.0);
  let far = u.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  let world = far.xyz / far.w;
  rd = normalize(world - ro);

  var col = vec3<f32>(0.0, 0.0, 0.0);
  let cell = floor(rd * 700.0);
  let rnd = hash13(cell);
  if (rnd > 0.993) {
    let local = fract(rd * 700.0) - vec3<f32>(0.5);
    let star = exp(-dot(local, local) * 18.0) * pow((rnd - 0.993) / 0.007, 3.0);
    col = col + mix(vec3<f32>(0.55, 0.72, 1.0), vec3<f32>(1.0, 0.82, 0.55), hash13(cell + 9.2)) * star * 1.2;
  }

  let shp = ray_sphere(ro, rd, 1.0);
  let atmo = ray_sphere(ro, rd, 1.075);
  if (atmo.y > 0.0) {
    let limb = pow(clamp(1.0 - abs(dot(normalize(ro + max(atmo.x, 0.0) * rd), -rd)), 0.0, 1.0), 2.3);
    col = col + vec3<f32>(0.18, 0.46, 1.0) * limb * u.visual.z * 0.018;
  }

  let moonDir = normalize(u.moonWorld.xyz - ro);
  let moonAng = acos(clamp(dot(rd, moonDir), -1.0, 1.0));
  let moonRadius = u.moonWorld.w;
  if (shp.x < 0.0 && moonAng < moonRadius) {
    let m = smoothstep(moonRadius, moonRadius * 0.65, moonAng);
    let phase = clamp(dot(moonDir, normalize(u.sunWorld.xyz)), 0.05, 1.0);
    col = mix(col, vec3<f32>(0.55, 0.55, 0.52) * phase, m);
  }

  if (shp.x > 0.0) {
    let p = ro + shp.x * rd;
    let dir = normalize(rot_y(p, -u.flags.w));
    let euv = dir_to_equirect(dir);
    let sun = dot(dir, normalize(u.sunObj.xyz));
    col = field_color(dir, euv, sun, -rd);
  }

  let sunAng = acos(clamp(dot(rd, normalize(u.sunWorld.xyz)), -1.0, 1.0));
  let sunRadius = max(u.sunWorld.w, 0.0036);
  if (shp.x < 0.0) {
    let disk = smoothstep(sunRadius * 1.08, sunRadius * 0.82, sunAng);
    let coronaCore = exp(-max(sunAng - sunRadius, 0.0) / (sunRadius * 2.6));
    let coronaMask = 1.0 - smoothstep(sunRadius * 3.0, sunRadius * 18.0, sunAng);
    let corona = coronaCore * coronaMask;
    col = col + vec3<f32>(34.0, 29.0, 20.0) * disk;
    col = col + vec3<f32>(4.0, 2.8, 1.35) * corona * (1.0 - disk);
  }

  return vec4<f32>(col, 1.0);
}

@fragment
fn fs_scene(in: VSOut) -> @location(0) vec4<f32> {
  return scene_color(in.uv, false);
}

@fragment
fn fs_unwrap(in: VSOut) -> @location(0) vec4<f32> {
  return scene_color(in.uv, true);
}
`;

const POST_WGSL = /* wgsl */`
struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};
struct PostParams {
  resolution: vec4<f32>,
  sun: vec4<f32>,
  lens: vec4<f32>,
};
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> u: PostParams;
@group(0) @binding(4) var<uniform> blurDir: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(3.0, 1.0),
    vec2<f32>(-1.0, 1.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(p[vi], 0.0, 1.0);
  out.uv = p[vi] * 0.5 + vec2<f32>(0.5);
  return out;
}

@fragment
fn fs_bright(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(hdrTex, linearSampler, in.uv, 0.0).rgb;
  let l = max(c.r, max(c.g, c.b));
  let bright = smoothstep(2.2, 8.0, l);
  return vec4<f32>(c * bright, 1.0);
}

@fragment
fn fs_blur(in: VSOut) -> @location(0) vec4<f32> {
  let texel = blurDir.xy / u.resolution.xy;
  var c = textureSampleLevel(hdrTex, linearSampler, in.uv, 0.0).rgb * 0.227027;
  c = c + textureSampleLevel(hdrTex, linearSampler, in.uv + texel * 1.384615, 0.0).rgb * 0.316216;
  c = c + textureSampleLevel(hdrTex, linearSampler, in.uv - texel * 1.384615, 0.0).rgb * 0.316216;
  c = c + textureSampleLevel(hdrTex, linearSampler, in.uv + texel * 3.230769, 0.0).rgb * 0.070270;
  c = c + textureSampleLevel(hdrTex, linearSampler, in.uv - texel * 3.230769, 0.0).rgb * 0.070270;
  return vec4<f32>(c, 1.0);
}

fn lens_gaussian(uv: vec2<f32>, center: vec2<f32>, radius: f32, aspect: f32) -> f32 {
  let d = (uv - center) * vec2<f32>(aspect, 1.0);
  return exp(-dot(d, d) / max(radius * radius, 0.000001));
}

@fragment
fn fs_final(in: VSOut) -> @location(0) vec4<f32> {
  var col = textureSampleLevel(hdrTex, linearSampler, in.uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTex, linearSampler, in.uv, 0.0).rgb;
  let bloomVis = u.lens.x;
  let directVis = u.lens.y;
  let aspect = u.resolution.x / max(u.resolution.y, 1.0);
  let sunUv = u.sun.xy;
  let center = vec2<f32>(0.5);
  col = col + bloom * 0.55 * bloomVis;

  let sunVector = (in.uv - sunUv) * vec2<f32>(aspect, 1.0);
  let sunDist = length(sunVector);
  let veil = exp(-sunDist * 8.5) * bloomVis * 0.055;
  col = col + vec3<f32>(1.0, 0.78, 0.46) * veil;

  let axis = center - sunUv;
  let offAxis = length(axis * vec2<f32>(aspect, 1.0));
  let ghostGate = directVis * (1.0 - smoothstep(0.82, 1.25, offAxis));
  var ghosts = vec3<f32>(0.0);
  ghosts = ghosts + vec3<f32>(1.0, 0.62, 0.35) * lens_gaussian(in.uv, center + axis * 0.85, 0.070, aspect) * 0.070;
  ghosts = ghosts + vec3<f32>(0.42, 0.74, 1.0) * lens_gaussian(in.uv, center + axis * 1.45, 0.105, aspect) * 0.040;
  ghosts = ghosts + vec3<f32>(0.70, 1.0, 0.72) * lens_gaussian(in.uv, center + axis * 2.12, 0.048, aspect) * 0.030;
  let vignette = 1.0 - smoothstep(0.62, 1.05, length((in.uv - center) * vec2<f32>(aspect, 1.0)));
  col = col + ghosts * ghostGate * vignette;

  col = col / (col + vec3<f32>(1.0));
  col = pow(col, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(col, 1.0);
}
`;

export async function startWebGPUApp() {
  const nav = window.navigator as Navigator & { gpu?: GPU };
  if (!nav.gpu) {
    throw new Error('WebGPU required: this browser does not expose navigator.gpu.');
  }
  const adapter = await nav.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('WebGPU required: no compatible GPU adapter was found.');
  }
  const device = await adapter.requestDevice();
  let deviceErrorShown = false;
  device.addEventListener('uncapturederror', (event) => {
    console.error('WebGPU uncaptured error:', event.error.message);
    if (deviceErrorShown || document.querySelector('.fatal-webgpu')) return;
    deviceErrorShown = true;
    const panel = document.createElement('div');
    panel.className = 'fatal-webgpu';
    panel.textContent = `WebGPU runtime error: ${event.error.message}`;
    document.body.appendChild(panel);
  });
  const app = new WebGPUSphereSim(device, nav.gpu);
  await app.start();
}

class WebGPUSphereSim {
  private readonly device: GPUDevice;
  private readonly gpu: GPU;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly sampler: GPUSampler;
  private readonly nearestSampler: GPUSampler;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly simModule: GPUShaderModule;
  private readonly renderModule: GPUShaderModule;
  private readonly postModule: GPUShaderModule;
  private readonly simPipeline: GPUComputePipeline;
  private readonly initPipeline: GPUComputePipeline;
  private readonly diagPipeline: GPUComputePipeline;
  private readonly scenePipeline: GPURenderPipeline;
  private readonly unwrapPipeline: GPURenderPipeline;
  private readonly brightPipeline: GPURenderPipeline;
  private readonly blurPipeline: GPURenderPipeline;
  private readonly finalPipeline: GPURenderPipeline;
  private readonly simBindGroupLayout: GPUBindGroupLayout;
  private readonly diagBindGroupLayout: GPUBindGroupLayout;
  private readonly renderBindGroupLayout: GPUBindGroupLayout;
  private readonly postBindGroupLayout: GPUBindGroupLayout;
  private readonly simUniform: GPUBuffer;
  private readonly renderUniform: GPUBuffer;
  private readonly postUniform: GPUBuffer;
  private readonly blurUniform: GPUBuffer;
  private readonly diagStorage: GPUBuffer;
  private readonly diagReadback: GPUBuffer;
  private readonly fields: { velocity: GpuField; dye: GpuField; temperature: GpuField; pressure: GpuField };
  private readonly textureCache = new Map<string, LoadedTexture>();
  private readonly textureDebug = {
    visible: false,
    el: null as HTMLDivElement | null,
    cards: {} as Partial<Record<TextureDebugKey, HTMLCanvasElement>>,
    sources: {} as Partial<Record<TextureDebugKey, CanvasImageSource>>,
  };
  private readonly rayTmp = new THREE.Ray();
  private readonly ndcNear = new THREE.Vector3();
  private readonly ndcFar = new THREE.Vector3();
  private readonly viewProj = new THREE.Matrix4();
  private readonly invViewProj = new THREE.Matrix4();
  private readonly sunObj = new THREE.Vector3();
  private readonly moonPos = new THREE.Vector3();
  private readonly splats: Splat[] = [];
  private readonly palette = [
    new THREE.Vector3(0.18, 0.55, 0.95),
    new THREE.Vector3(0.95, 0.35, 0.55),
    new THREE.Vector3(0.55, 0.9, 0.55),
    new THREE.Vector3(0.95, 0.7, 0.25),
    new THREE.Vector3(0.7, 0.45, 0.95),
    new THREE.Vector3(0.25, 0.85, 0.85),
  ];
  private readonly blackTexture: LoadedTexture;
  private readonly gridTexture: Promise<LoadedTexture>;

  private baseTexture: LoadedTexture;
  private nightTexture: LoadedTexture;
  private geoTexture: LoadedTexture;
  private topoTexture: LoadedTexture;
  private hdrTexture: GPUTexture | null = null;
  private hdrView: GPUTextureView | null = null;
  private bloomA: GPUTexture | null = null;
  private bloomB: GPUTexture | null = null;
  private bloomAView: GPUTextureView | null = null;
  private bloomBView: GPUTextureView | null = null;
  private sceneBindGroup: GPUBindGroup | null = null;
  private postBindGroupBright: GPUBindGroup | null = null;
  private postBindGroupBlurA: GPUBindGroup | null = null;
  private postBindGroupBlurB: GPUBindGroup | null = null;
  private postBindGroupFinal: GPUBindGroup | null = null;
  private textureBindDirty = true;
  private sizeDirty = true;
  private contextConfigured = false;
  private painting = false;
  private prevDir: THREE.Vector3 | null = null;
  private latestProbePointer: { x: number; y: number } | null = null;
  private readonly orbitInputEvents = new WeakSet<PointerEvent>();
  private desktopOrbitFlip: { pointerId: number; startY: number } | null = null;
  private readonly coarsePointerQuery = window.matchMedia('(pointer: coarse)');
  private hasCoarsePointer = this.coarsePointerQuery.matches;
  private touchMode: TouchMode = 'orbit';
  private colorIdx = Math.floor(Math.random() * 6);
  private showUnwrap = false;
  private lastTime = performance.now();
  private realSeconds = 0;
  private physicalSeconds = 0;
  private earthRotationRadians = 0;
  private lastDiagAt = -Infinity;
  private diagPending = false;
  private simBindGroup: GPUBindGroup | null = null;
  private diagBindGroup: GPUBindGroup | null = null;

  constructor(device: GPUDevice, gpu: GPU) {
    this.device = device;
    this.gpu = gpu;
    window.__sphereSimBackend = 'webgpu';
    document.body.dataset.backend = 'webgpu';

    const appEl = document.getElementById('app');
    if (!appEl) throw new Error('Missing #app mount element.');

    this.canvas = document.createElement('canvas');
    this.canvas.id = 'webgpu-canvas';
    appEl.appendChild(this.canvas);

    const context = this.canvas.getContext('webgpu');
    if (!context) throw new Error('WebGPU required: failed to create a GPUCanvasContext.');
    this.context = context;
    this.format = this.gpu.getPreferredCanvasFormat();

    this.simUniform = this.device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.renderUniform = this.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.postUniform = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blurUniform = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.diagStorage = this.device.createBuffer({
      size: DIAG_SAMPLES * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.diagReadback = this.device.createBuffer({
      size: DIAG_SAMPLES * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
    });
    this.nearestSampler = this.device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100000);
    this.camera.position.set(0, 0.4, 3.4);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.16;
    this.controls.minDistance = 1.8;
    this.controls.maxDistance = SUN_DISTANCE * 1.6;
    this.controls.rotateSpeed = 0.6;
    this.controls.enablePan = false;
    this.controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };

    this.blackTexture = this.createSolidTexture(0, 0, 0, 255);
    this.gridTexture = this.createGridTexture();
    this.baseTexture = this.blackTexture;
    this.nightTexture = this.blackTexture;
    this.geoTexture = this.blackTexture;
    this.topoTexture = this.blackTexture;

    this.fields = {
      velocity: this.createField(),
      dye: this.createField(),
      temperature: this.createField(),
      pressure: this.createField(),
    };
    this.simBindGroupLayout = this.device.createBindGroupLayout({
      label: 'sphere-sim-bind-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
      ],
    });
    this.diagBindGroupLayout = this.device.createBindGroupLayout({
      label: 'sphere-diag-bind-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.renderBindGroupLayout = this.device.createBindGroupLayout({
      label: 'sphere-render-bind-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      ],
    });
    this.postBindGroupLayout = this.device.createBindGroupLayout({
      label: 'sphere-post-bind-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const simPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.simBindGroupLayout] });
    const diagPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.simBindGroupLayout, this.diagBindGroupLayout] });
    const renderPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.renderBindGroupLayout] });
    const postPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.postBindGroupLayout] });

    this.simModule = this.device.createShaderModule({ label: 'sphere-sim-compute', code: SIM_WGSL });
    this.renderModule = this.device.createShaderModule({ label: 'sphere-scene-render', code: RENDER_WGSL });
    this.postModule = this.device.createShaderModule({ label: 'sphere-postprocess', code: POST_WGSL });
    this.initPipeline = this.device.createComputePipeline({ layout: diagPipelineLayout, compute: { module: this.simModule, entryPoint: 'init' } });
    this.simPipeline = this.device.createComputePipeline({ layout: diagPipelineLayout, compute: { module: this.simModule, entryPoint: 'step' } });
    this.diagPipeline = this.device.createComputePipeline({ layout: diagPipelineLayout, compute: { module: this.simModule, entryPoint: 'diagnostics' } });
    this.scenePipeline = this.device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: { module: this.renderModule, entryPoint: 'vs' },
      fragment: { module: this.renderModule, entryPoint: 'fs_scene', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.unwrapPipeline = this.device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: { module: this.renderModule, entryPoint: 'vs' },
      fragment: { module: this.renderModule, entryPoint: 'fs_unwrap', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.brightPipeline = this.device.createRenderPipeline({
      layout: postPipelineLayout,
      vertex: { module: this.postModule, entryPoint: 'vs' },
      fragment: { module: this.postModule, entryPoint: 'fs_bright', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.blurPipeline = this.device.createRenderPipeline({
      layout: postPipelineLayout,
      vertex: { module: this.postModule, entryPoint: 'vs' },
      fragment: { module: this.postModule, entryPoint: 'fs_blur', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.finalPipeline = this.device.createRenderPipeline({
      layout: postPipelineLayout,
      vertex: { module: this.postModule, entryPoint: 'vs' },
      fragment: { module: this.postModule, entryPoint: 'fs_final', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.device.lost.then((info) => {
      this.showFatal(`WebGPU device lost: ${info.message || info.reason}`);
    });
  }

  async start() {
    this.configureCanvas();
    this.bindUi();
    this.bindPointer();
    this.resetSimulation();
    await this.setUnderlay((document.getElementById('sel-tex') as HTMLSelectElement).value);
    void this.setupClimate();
    requestAnimationFrame((t) => this.frame(t));
  }

  private createField(): GpuField {
    const make = () => this.device.createTexture({
      size: { width: FACE, height: FACE, depthOrArrayLayers: 6 },
      dimension: '2d',
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
    const read = make();
    const write = make();
    return {
      read,
      write,
      readView: read.createView({ dimension: '2d-array', arrayLayerCount: 6 }),
      writeView: write.createView({ dimension: '2d-array', arrayLayerCount: 6 }),
    };
  }

  private swapField(field: GpuField) {
    const read = field.read;
    const readView = field.readView;
    field.read = field.write;
    field.readView = field.writeView;
    field.write = read;
    field.writeView = readView;
  }

  private createSolidTexture(r: number, g: number, b: number, a: number): LoadedTexture {
    const texture = this.device.createTexture({
      size: { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture({ texture }, new Uint8Array([r, g, b, a]), { bytesPerRow: 4 }, { width: 1, height: 1 });
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    canvas.getContext('2d')!.fillRect(0, 0, 1, 1);
    return { texture, view: texture.createView(), source: canvas };
  }

  private async createGridTexture(): Promise<LoadedTexture> {
    const c = document.createElement('canvas');
    c.width = 1024;
    c.height = 512;
    const g = c.getContext('2d')!;
    g.fillStyle = '#0a2030';
    g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#143a4f';
    for (let y = 0; y < 12; y++) {
      for (let x = 0; x < 24; x++) {
        if ((x + y) % 2 === 0) g.fillRect(x * (c.width / 24), y * (c.height / 12), c.width / 24, c.height / 12);
      }
    }
    g.strokeStyle = 'rgba(120,200,230,0.5)';
    for (let x = 0; x <= 24; x++) {
      g.beginPath();
      g.moveTo(x * (c.width / 24), 0);
      g.lineTo(x * (c.width / 24), c.height);
      g.stroke();
    }
    const bitmap = await createImageBitmap(c);
    return this.createTextureFromSource(bitmap);
  }

  private createTextureFromSource(source: ImageBitmap | HTMLCanvasElement): LoadedTexture {
    const width = source.width;
    const height = source.height;
    const texture = this.device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture({ source }, { texture }, { width, height });
    return { texture, view: texture.createView(), source };
  }

  private async loadTexture(key: keyof typeof TEX_URLS): Promise<LoadedTexture> {
    const cached = this.textureCache.get(key);
    if (cached) return cached;
    try {
      const response = await fetch(TEX_URLS[key], { mode: 'cors' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const blob = await response.blob();
      const bitmap = await createImageBitmap(blob, { imageOrientation: 'none' });
      const texture = this.createTextureFromSource(bitmap);
      this.textureCache.set(key, texture);
      return texture;
    } catch (error) {
      console.warn(`Texture load failed for ${key}`, error);
      return this.blackTexture;
    }
  }

  private async setUnderlay(key: string) {
    if (key === 'none') {
      this.baseTexture = this.blackTexture;
      this.nightTexture = this.blackTexture;
    } else if (key === 'grid') {
      this.baseTexture = await this.gridTexture;
      this.nightTexture = this.blackTexture;
    } else {
      this.baseTexture = await this.loadTexture('earth');
      this.nightTexture = await this.loadTexture('night');
    }
    this.textureDebug.sources.base = this.baseTexture.source;
    this.textureDebug.sources.emissive = this.nightTexture.source;
    this.updateTextureDebugPanel();
    this.textureBindDirty = true;
  }

  private async setupClimate() {
    this.geoTexture = await this.loadTexture('earth');
    this.topoTexture = await this.loadTexture('topo');
    this.textureDebug.sources.geo = this.geoTexture.source;
    this.textureDebug.sources.topo = this.topoTexture.source;
    this.updateTextureDebugPanel();
    this.simBindGroup = null;
    this.textureBindDirty = true;
    this.resetSimulation();
  }

  private bindUi() {
    const bindSlider = (id: string, valId: string, key: NumericParamKey, fmt?: (x: number) => string) => {
      const slider = document.getElementById(id) as HTMLInputElement;
      const val = document.getElementById(valId) as HTMLElement;
      const update = () => {
        params[key] = parseFloat(slider.value);
        val.textContent = fmt ? fmt(params[key]) : slider.value;
      };
      slider.addEventListener('input', update);
      update();
    };
    bindSlider('s-speed', 'v-speed', 'speed', this.formatTimeScale);
    bindSlider('s-spin', 'v-spin', 'spin', (x) => `${x.toFixed(2)}x`);
    bindSlider('s-overlay', 'v-overlay', 'overlay', (x) => x.toFixed(2));
    bindSlider('s-glint', 'v-glint', 'specular', (x) => x.toFixed(1));
    bindSlider('s-atmo', 'v-atmo', 'atmosphere', (x) => x.toFixed(0));
    bindSlider('s-sun', 'v-sun', 'solar', (x) => x.toFixed(0));
    bindSlider('s-cool', 'v-cool', 'cool', (x) => x.toFixed(0));
    bindSlider('s-green', 'v-green', 'greenhouse', (x) => x.toFixed(0));
    bindSlider('s-ocean', 'v-ocean', 'oceanInertia', (x) => x.toFixed(0));
    bindSlider('s-coriolis', 'v-coriolis', 'coriolis', (x) => `${x.toFixed(1)}x`);
    bindSlider('s-heat', 'v-heat', 'heating', (x) => x.toFixed(1));
    bindSlider('s-curl', 'v-curl', 'curl');
    bindSlider('s-veldis', 'v-veldis', 'velDissipation', (x) => x.toFixed(2));
    bindSlider('s-dendis', 'v-dendis', 'denDissipation', (x) => x.toFixed(2));
    bindSlider('s-iter', 'v-iter', 'pressureIters');
    bindSlider('s-radius', 'v-radius', 'brush', (x) => x.toFixed(1));

    const selView = document.getElementById('sel-view') as HTMLSelectElement;
    selView.addEventListener('change', () => { params.viewMode = VIEW[selView.value]; });
    params.viewMode = VIEW[selView.value];

    const selTex = document.getElementById('sel-tex') as HTMLSelectElement;
    selTex.addEventListener('change', () => { void this.setUnderlay(selTex.value); });

    const bEmit = document.getElementById('b-emit') as HTMLButtonElement;
    const bDay = document.getElementById('b-daynight') as HTMLButtonElement;
    const bUnwrap = document.getElementById('b-unwrap') as HTMLButtonElement;
    const bTexDebug = document.getElementById('b-texdebug') as HTMLButtonElement;
    const bMobileOrbit = document.getElementById('b-mobile-orbit') as HTMLButtonElement;
    const bMobilePaint = document.getElementById('b-mobile-paint') as HTMLButtonElement;
    bEmit.addEventListener('click', () => { params.emitters = !params.emitters; bEmit.classList.toggle('active', params.emitters); });
    bDay.addEventListener('click', () => { params.dayNight = !params.dayNight; bDay.classList.toggle('active', params.dayNight); });
    bUnwrap.addEventListener('click', () => {
      this.showUnwrap = !this.showUnwrap;
      bUnwrap.classList.toggle('active', this.showUnwrap);
      document.getElementById('unwrap-frame')!.classList.toggle('show', this.showUnwrap);
    });
    bTexDebug.addEventListener('click', () => {
      this.textureDebug.visible = !this.textureDebug.visible;
      bTexDebug.classList.toggle('active', this.textureDebug.visible);
      const panel = this.makeTextureDebugPanel();
      panel.classList.toggle('show', this.textureDebug.visible);
      if (this.textureDebug.visible) this.updateTextureDebugPanel();
    });
    document.getElementById('b-reset')!.addEventListener('click', () => this.resetSimulation());
    bMobileOrbit.addEventListener('click', () => this.setTouchMode('orbit'));
    bMobilePaint.addEventListener('click', () => this.setTouchMode('paint'));
    this.coarsePointerQuery.addEventListener('change', (event) => {
      this.hasCoarsePointer = event.matches;
      this.applyTouchMode();
    });
    this.applyTouchMode();
  }

  private bindPointer() {
    this.canvas.style.touchAction = 'none';
    this.canvas.style.userSelect = 'none';
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.bindDesktopOrbitAxisFix();
    this.canvas.addEventListener('pointerdown', (e) => {
      if (!this.shouldPaintFromPointer(e)) return;
      this.blockTouchOrbitIfPainting(e);
      const dir = this.dirAtPointer(e);
      if (!dir) return;
      this.painting = true;
      this.prevDir = null;
      this.colorIdx = (this.colorIdx + 1) % this.palette.length;
      try { this.canvas.setPointerCapture(e.pointerId); } catch {}
      this.queuePaint(dir);
    }, { capture: true });
    this.canvas.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'touch' && this.touchMode !== 'paint') return;
      this.blockTouchOrbitIfPainting(e);
      const dir = this.dirAtPointer(e);
      if (this.painting && dir) this.queuePaint(dir);
      if (this.painting && !dir) this.prevDir = null;
    }, { capture: true });
    this.canvas.addEventListener('pointerleave', () => {
      this.latestProbePointer = null;
      this.painting = false;
      this.prevDir = null;
    });
    window.addEventListener('pointerup', (e) => {
      if (e.pointerType === 'touch' && this.painting) this.blockTouchOrbitIfPainting(e);
      this.painting = false;
      this.prevDir = null;
      try { this.canvas.releasePointerCapture(e.pointerId); } catch {}
    }, { capture: true });
    window.addEventListener('pointercancel', (e) => {
      if (e.pointerType === 'touch' && this.painting) this.blockTouchOrbitIfPainting(e);
      if (this.desktopOrbitFlip?.pointerId === e.pointerId) this.desktopOrbitFlip = null;
      this.painting = false;
      this.prevDir = null;
      try { this.canvas.releasePointerCapture(e.pointerId); } catch {}
    }, { capture: true });
    window.addEventListener('resize', () => {
      this.sizeDirty = true;
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
    });
  }

  private bindDesktopOrbitAxisFix() {
    this.canvas.addEventListener('pointerdown', (e) => {
      if (this.orbitInputEvents.has(e) || e.pointerType !== 'mouse' || e.button !== 2) return;
      this.desktopOrbitFlip = { pointerId: e.pointerId, startY: e.clientY };
    }, { capture: true });
    this.canvas.addEventListener('pointermove', (e) => {
      if (this.orbitInputEvents.has(e) || e.pointerType !== 'mouse') return;
      const active = this.desktopOrbitFlip;
      if (!active || active.pointerId !== e.pointerId || (e.buttons & 2) === 0) return;

      const mirrored = new PointerEvent(e.type, {
        bubbles: true,
        cancelable: true,
        composed: true,
        pointerId: e.pointerId,
        pointerType: e.pointerType,
        isPrimary: e.isPrimary,
        width: e.width,
        height: e.height,
        pressure: e.pressure,
        tangentialPressure: e.tangentialPressure,
        tiltX: e.tiltX,
        tiltY: e.tiltY,
        twist: e.twist,
        button: e.button,
        buttons: e.buttons,
        clientX: e.clientX,
        clientY: active.startY - (e.clientY - active.startY),
        screenX: e.screenX,
        screenY: e.screenY,
        ctrlKey: e.ctrlKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        metaKey: e.metaKey,
      });
      this.orbitInputEvents.add(mirrored);
      e.preventDefault();
      e.stopImmediatePropagation();
      this.canvas.dispatchEvent(mirrored);
    }, { capture: true });
    window.addEventListener('pointerup', (e) => {
      if (this.desktopOrbitFlip?.pointerId === e.pointerId) this.desktopOrbitFlip = null;
    }, { capture: true });
  }

  private setTouchMode(mode: TouchMode) {
    this.touchMode = mode;
    this.painting = false;
    this.prevDir = null;
    this.applyTouchMode();
  }

  private applyTouchMode() {
    const paintMode = this.hasCoarsePointer && this.touchMode === 'paint';
    this.controls.enabled = !paintMode;
    document.body.dataset.touchMode = this.touchMode;
    const bMobileOrbit = document.getElementById('b-mobile-orbit') as HTMLButtonElement | null;
    const bMobilePaint = document.getElementById('b-mobile-paint') as HTMLButtonElement | null;
    bMobileOrbit?.classList.toggle('active', this.touchMode === 'orbit');
    bMobilePaint?.classList.toggle('active', this.touchMode === 'paint');
    bMobileOrbit?.setAttribute('aria-pressed', String(this.touchMode === 'orbit'));
    bMobilePaint?.setAttribute('aria-pressed', String(this.touchMode === 'paint'));
  }

  private shouldPaintFromPointer(e: PointerEvent) {
    if (e.pointerType === 'touch') return this.touchMode === 'paint';
    if (e.button === 2) e.preventDefault();
    return e.button === 0;
  }

  private blockTouchOrbitIfPainting(e: PointerEvent) {
    if (e.pointerType !== 'touch' || this.touchMode !== 'paint') return;
    e.preventDefault();
    e.stopImmediatePropagation();
  }

  private queuePaint(dir: THREE.Vector3) {
    const delta = new THREE.Vector3();
    if (this.prevDir) delta.subVectors(dir, this.prevDir).multiplyScalar(36.0);
    const color = this.palette[this.colorIdx].clone();
    this.splats.push({ dir: dir.clone(), delta, color });
    if (this.splats.length > 8) this.splats.shift();
    this.prevDir = dir.clone();
  }

  private dirAtPointer(e: PointerEvent) {
    this.latestProbePointer = { x: e.clientX, y: e.clientY };
    return this.dirAtScreen(e.clientX, e.clientY);
  }

  private dirAtScreen(x: number, y: number): THREE.Vector3 | null {
    this.camera.updateMatrixWorld();
    const ndcX = (x / window.innerWidth) * 2 - 1;
    const ndcY = -(y / window.innerHeight) * 2 + 1;
    this.ndcNear.set(ndcX, ndcY, -1).unproject(this.camera);
    this.ndcFar.set(ndcX, ndcY, 1).unproject(this.camera);
    const origin = this.ndcNear.clone();
    const direction = this.ndcFar.clone().sub(this.ndcNear).normalize();
    this.rayTmp.set(origin, direction);
    const hit = new THREE.Vector3();
    const ok = this.rayTmp.intersectSphere(new THREE.Sphere(new THREE.Vector3(), 1), hit);
    if (!ok) return null;
    return this.rotateY(hit.normalize(), -this.earthRotationRadians);
  }

  private rotateY(v: THREE.Vector3, angle: number) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new THREE.Vector3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
  }

  private formatTimeScale(x: number) {
    const minutesPerSecond = x * PHYSICAL_SECONDS_PER_REAL_SECOND / 60;
    if (minutesPerSecond < 60) return `${minutesPerSecond.toFixed(0)}m/s`;
    const hoursPerSecond = minutesPerSecond / 60;
    return `${hoursPerSecond % 1 === 0 ? hoursPerSecond.toFixed(0) : hoursPerSecond.toFixed(1)}h/s`;
  }

  private configureCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.max(1, Math.floor(window.innerWidth * dpr));
    const height = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.canvas.style.width = '100vw';
      this.canvas.style.height = '100vh';
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.sizeDirty = true;
    }
    if (!this.contextConfigured || this.sizeDirty) {
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: 'opaque',
      });
      this.contextConfigured = true;
      if (this.sizeDirty) this.recreateRenderTargets(width, height);
      this.sizeDirty = false;
    }
  }

  private recreateRenderTargets(width: number, height: number) {
    this.hdrTexture?.destroy();
    this.bloomA?.destroy();
    this.bloomB?.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.hdrTexture = this.device.createTexture({ label: 'scene-hdr', size: { width, height }, format: 'rgba16float', usage });
    this.bloomA = this.device.createTexture({ label: 'bloom-a', size: { width, height }, format: 'rgba16float', usage });
    this.bloomB = this.device.createTexture({ label: 'bloom-b', size: { width, height }, format: 'rgba16float', usage });
    this.hdrView = this.hdrTexture.createView();
    this.bloomAView = this.bloomA.createView();
    this.bloomBView = this.bloomB.createView();
    this.postBindGroupBright = null;
    this.postBindGroupBlurA = null;
    this.postBindGroupBlurB = null;
    this.postBindGroupFinal = null;
  }

  private resetSimulation() {
    this.writeSimUniform(0, 0, null);
    const encoder = this.device.createCommandEncoder({ label: 'reset-simulation' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.initPipeline);
    pass.setBindGroup(0, this.getSimBindGroup(this.initPipeline));
    pass.setBindGroup(1, this.getDiagBindGroup());
    pass.dispatchWorkgroups(Math.ceil(FACE / WORKGROUP), Math.ceil(FACE / WORKGROUP), 6);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    for (const field of Object.values(this.fields)) this.swapField(field);
    this.simBindGroup = null;
    this.diagBindGroup = null;
    this.textureBindDirty = true;
  }

  private frame(now: number) {
    requestAnimationFrame((t) => this.frame(t));
    this.configureCanvas();

    let elapsed = (now - this.lastTime) / 1000;
    this.lastTime = now;
    elapsed = Math.min(Math.max(elapsed, 1 / 240), 1 / 20);
    const deltaPhysicalSeconds = elapsed * PHYSICAL_SECONDS_PER_REAL_SECOND * params.speed;
    this.realSeconds += elapsed;
    this.physicalSeconds += deltaPhysicalSeconds;
    this.earthRotationRadians += (Math.PI * 2) * (deltaPhysicalSeconds / EARTH_SOLAR_DAY_SECONDS) * params.spin;

    this.controls.update();
    this.updateMoon(this.physicalSeconds);
    this.updateSunObject();
    this.runEmitters(deltaPhysicalSeconds / EARTH_SOLAR_DAY_SECONDS);
    this.stepSimulation(deltaPhysicalSeconds);
    this.updateDiagnostics();
    this.render(now);
  }

  private stepSimulation(deltaPhysicalSeconds: number) {
    const advectDt = deltaPhysicalSeconds * MODEL_VELOCITY_UNIT_MPS / EARTH_RADIUS_M;
    const thermalDt = deltaPhysicalSeconds / EARTH_SOLAR_DAY_SECONDS;
    const splat = this.splats.shift() ?? null;
    this.writeSimUniform(advectDt, thermalDt, splat);

    const encoder = this.device.createCommandEncoder({ label: 'sim-step' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.simPipeline);
    pass.setBindGroup(0, this.getSimBindGroup(this.simPipeline));
    pass.setBindGroup(1, this.getDiagBindGroup());
    pass.dispatchWorkgroups(Math.ceil(FACE / WORKGROUP), Math.ceil(FACE / WORKGROUP), 6);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    for (const field of Object.values(this.fields)) this.swapField(field);
    this.simBindGroup = null;
    this.diagBindGroup = null;
    this.textureBindDirty = true;
  }

  private getSimBindGroup(pipeline: GPUComputePipeline) {
    if (this.simBindGroup) return this.simBindGroup;
    this.simBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.geoTexture.view },
        { binding: 3, resource: this.topoTexture.view },
        { binding: 4, resource: this.fields.velocity.readView },
        { binding: 5, resource: this.fields.dye.readView },
        { binding: 6, resource: this.fields.temperature.readView },
        { binding: 7, resource: this.fields.pressure.readView },
        { binding: 8, resource: this.fields.velocity.writeView },
        { binding: 9, resource: this.fields.dye.writeView },
        { binding: 10, resource: this.fields.temperature.writeView },
        { binding: 11, resource: this.fields.pressure.writeView },
      ],
    });
    return this.simBindGroup;
  }

  private getDiagBindGroup() {
    if (this.diagBindGroup) return this.diagBindGroup;
    this.diagBindGroup = this.device.createBindGroup({
      layout: this.diagBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.diagStorage } }],
    });
    return this.diagBindGroup;
  }

  private writeSimUniform(advectDt: number, thermalDt: number, splat: Splat | null) {
    const brushRadius = params.brush * 0.06;
    const dir = splat?.dir ?? new THREE.Vector3(1, 0, 0);
    const delta = splat?.delta ?? new THREE.Vector3();
    const color = splat?.color ?? new THREE.Vector3();
    const data = new Float32Array(40);
    data.set([FACE, 1.5 / FACE, advectDt, thermalDt], 0);
    data.set([this.physicalSeconds / EARTH_SOLAR_DAY_SECONDS, this.earthRotationRadians, params.dayNight ? 1 : 0, params.emitters ? 1 : 0], 4);
    data.set([this.sunObj.x, this.sunObj.y, this.sunObj.z, 0], 8);
    data.set([dir.x, dir.y, dir.z, 0], 12);
    data.set([delta.x, delta.y, delta.z, 1.0], 16);
    data.set([splat ? 1 : 0, brushRadius * brushRadius + 0.0004, color.x, params.denDissipation], 20);
    data.set([params.solar, params.cool, params.greenhouse, params.oceanInertia], 24);
    data.set([CORIOLIS_OMEGA_CODE * params.coriolis, params.heating, params.curl, params.velDissipation], 28);
    this.device.queue.writeBuffer(this.simUniform, 0, data);
  }

  private updateMoon(physicalSeconds: number) {
    const anomaly = MOON_INITIAL_ANOMALY + (physicalSeconds / MOON_SIDEREAL_ORBIT_SECONDS) * Math.PI * 2;
    const r = MOON_ORBIT_A * (1 - MOON_ORBIT_E * MOON_ORBIT_E) / (1 + MOON_ORBIT_E * Math.cos(anomaly));
    const x = Math.cos(anomaly) * r;
    const z = Math.sin(anomaly) * r;
    this.moonPos.set(x, Math.sin(MOON_ORBIT_INCLINATION) * z, Math.cos(MOON_ORBIT_INCLINATION) * z);
  }

  private updateSunObject() {
    this.sunObj.copy(this.rotateY(SUN_WORLD, -this.earthRotationRadians)).normalize();
  }

  private runEmitters(physicalDays: number) {
    if (!params.emitters) return;
    const t = this.realSeconds + physicalDays * 20;
    const dirs = [
      new THREE.Vector3(0.35, 0.15, 0.92).normalize(),
      new THREE.Vector3(-0.45, -0.1, 0.88).normalize(),
    ];
    for (let i = 0; i < dirs.length; i++) {
      const dir = dirs[i];
      const tangent = new THREE.Vector3(-dir.z, 0, dir.x).normalize();
      const delta = tangent.multiplyScalar(0.5 + 0.2 * Math.sin(t + i));
      this.splats.push({ dir, delta, color: this.palette[(i + 2) % this.palette.length] });
    }
  }

  private render(now: number) {
    if (!this.hdrView || !this.bloomAView || !this.bloomBView) return;
    this.writeRenderUniform(now);
    this.writePostUniform(now);
    this.ensureSceneBindGroup();
    this.ensurePostBindGroups();

    const encoder = this.device.createCommandEncoder({ label: 'render-frame' });
    const scenePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.hdrView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    scenePass.setPipeline(this.scenePipeline);
    scenePass.setBindGroup(0, this.sceneBindGroup);
    scenePass.draw(3);
    scenePass.end();

    this.drawFullscreen(encoder, this.brightPipeline, this.postBindGroupBright!, this.bloomAView, 'clear');
    this.device.queue.writeBuffer(this.blurUniform, 0, new Float32Array([1, 0, 0, 0]));
    this.drawFullscreen(encoder, this.blurPipeline, this.postBindGroupBlurA!, this.bloomBView, 'clear');
    this.device.queue.writeBuffer(this.blurUniform, 0, new Float32Array([0, 1, 0, 0]));
    this.drawFullscreen(encoder, this.blurPipeline, this.postBindGroupBlurB!, this.bloomAView, 'clear');

    const swapTexture = this.context.getCurrentTexture();
    const swapView = swapTexture.createView();
    this.drawFullscreen(encoder, this.finalPipeline, this.postBindGroupFinal!, swapView, 'clear');
    if (this.showUnwrap) this.drawUnwrap(encoder, swapView);
    this.device.queue.submit([encoder.finish()]);
  }

  private drawFullscreen(encoder: GPUCommandEncoder, pipeline: GPURenderPipeline, bindGroup: GPUBindGroup, view: GPUTextureView, loadOp: GPULoadOp) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp, storeOp: 'store' }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  private drawUnwrap(encoder: GPUCommandEncoder, swapView: GPUTextureView) {
    const frame = document.getElementById('unwrap-view');
    if (!frame || !this.sceneBindGroup) return;
    const rect = frame.getBoundingClientRect();
    if (rect.width <= 1 || rect.height <= 1) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const x = Math.max(0, Math.floor(rect.left * dpr));
    const y = Math.max(0, Math.floor((window.innerHeight - rect.bottom) * dpr));
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: swapView, loadOp: 'load', storeOp: 'store' }],
    });
    pass.setViewport(x, y, w, h, 0, 1);
    pass.setScissorRect(x, y, w, h);
    pass.setPipeline(this.unwrapPipeline);
    pass.setBindGroup(0, this.sceneBindGroup);
    pass.draw(3);
    pass.end();
  }

  private writeRenderUniform(now: number) {
    this.camera.updateMatrixWorld();
    this.viewProj.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this.invViewProj.copy(this.viewProj).invert();

    const moonVec = this.moonPos.clone();
    const moonAngular = Math.asin(Math.min(MOON_RADIUS / Math.max(moonVec.length(), 1e-4), 1));
    const sunAngular = Math.asin(Math.min(SUN_RADIUS / SUN_DISTANCE, 1));
    const data = new Float32Array(64);
    data.set(this.invViewProj.elements, 0);
    data.set([this.camera.position.x, this.camera.position.y, this.camera.position.z, 1], 16);
    data.set([SUN_WORLD.x, SUN_WORLD.y, SUN_WORLD.z, sunAngular], 20);
    data.set([this.sunObj.x, this.sunObj.y, this.sunObj.z, now * 0.001], 24);
    data.set([moonVec.x, moonVec.y, moonVec.z, moonAngular], 28);
    data.set([this.canvas.width, this.canvas.height, params.viewMode, params.overlay], 32);
    data.set([params.dayNight ? 1 : 0, params.specular, params.atmosphere, this.baseTexture === this.blackTexture ? 0 : 1], 36);
    data.set([this.nightTexture === this.blackTexture ? 0 : 1, this.geoTexture === this.blackTexture ? 0 : 1, this.topoTexture === this.blackTexture ? 0 : 1, this.earthRotationRadians], 40);
    this.device.queue.writeBuffer(this.renderUniform, 0, data);
  }

  private writePostUniform(now: number) {
    const sunState = this.sunScreenState();
    const data = new Float32Array(16);
    data.set([this.canvas.width, this.canvas.height, 0, 0], 0);
    data.set([sunState.uv.x, sunState.uv.y, 0, 0], 4);
    data.set([sunState.bloom, sunState.ghost, now * 0.001, 0], 8);
    this.device.queue.writeBuffer(this.postUniform, 0, data);
  }

  private sunScreenState() {
    const sunPos = SUN_WORLD.clone().multiplyScalar(SUN_DISTANCE);
    const view = sunPos.clone().applyMatrix4(this.camera.matrixWorldInverse);
    const uv = new THREE.Vector2(0.5, 0.5);
    if (view.z >= -this.camera.near) return { uv, bloom: 0, ghost: 0 };
    const projected = sunPos.clone().project(this.camera);
    uv.set(projected.x * 0.5 + 0.5, projected.y * 0.5 + 0.5);
    const margin = 0.28;
    const off = Math.max(Math.abs(uv.x - 0.5) - 0.5, Math.abs(uv.y - 0.5) - 0.5);
    const offFade = 1 - THREE.MathUtils.smoothstep(off, 0, margin);
    const cam = this.camera.position;
    const toSun = sunPos.clone().sub(cam);
    const sunRay = toSun.clone().normalize();
    const tClosest = -cam.dot(sunRay);
    let occult = 0;
    if (tClosest > 0 && tClosest < toSun.length()) {
      const closest = cam.clone().add(sunRay.clone().multiplyScalar(tClosest)).length();
      occult = 1 - THREE.MathUtils.smoothstep(closest, 0.98, 1.08);
    }
    const visibility = Math.max(0, 1 - occult) * offFade;
    const screenEdge = Math.max(Math.abs(uv.x - 0.5), Math.abs(uv.y - 0.5));
    const onScreen = 1 - THREE.MathUtils.smoothstep(screenEdge, 0.46, 0.58);
    return { uv, bloom: Math.pow(visibility, 0.55), ghost: visibility * onScreen };
  }

  private ensureSceneBindGroup() {
    if (this.sceneBindGroup && !this.textureBindDirty) return;
    this.sceneBindGroup = this.device.createBindGroup({
      layout: this.renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.renderUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.baseTexture.view },
        { binding: 3, resource: this.nightTexture.view },
        { binding: 4, resource: this.geoTexture.view },
        { binding: 5, resource: this.topoTexture.view },
        { binding: 6, resource: this.fields.velocity.readView },
        { binding: 7, resource: this.fields.dye.readView },
        { binding: 8, resource: this.fields.temperature.readView },
        { binding: 9, resource: this.fields.pressure.readView },
      ],
    });
    this.textureBindDirty = false;
  }

  private ensurePostBindGroups() {
    if (this.postBindGroupFinal && this.postBindGroupBright && this.postBindGroupBlurA && this.postBindGroupBlurB) return;
    const layout = this.postBindGroupLayout;
    const make = (hdrView: GPUTextureView, bloomView: GPUTextureView) => this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: hdrView },
        { binding: 2, resource: bloomView },
        { binding: 3, resource: { buffer: this.postUniform } },
        { binding: 4, resource: { buffer: this.blurUniform } },
      ],
    });
    this.postBindGroupBright = make(this.hdrView!, this.blackTexture.view);
    this.postBindGroupBlurA = make(this.bloomAView!, this.blackTexture.view);
    this.postBindGroupBlurB = make(this.bloomBView!, this.blackTexture.view);
    this.postBindGroupFinal = make(this.hdrView!, this.bloomAView!);
  }

  private async updateDiagnostics() {
    if (this.diagPending || this.realSeconds - this.lastDiagAt < 0.33) return;
    this.diagPending = true;
    this.lastDiagAt = this.realSeconds;

    try {
      const encoder = this.device.createCommandEncoder({ label: 'temperature-diagnostics' });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.diagPipeline);
      pass.setBindGroup(0, this.getSimBindGroup(this.diagPipeline));
      pass.setBindGroup(1, this.getDiagBindGroup());
      pass.dispatchWorkgroups(Math.ceil(DIAG_FACE / WORKGROUP), Math.ceil(DIAG_FACE / WORKGROUP), 6);
      pass.end();
      encoder.copyBufferToBuffer(this.diagStorage, 0, this.diagReadback, 0, DIAG_SAMPLES * 16);
      this.device.queue.submit([encoder.finish()]);
      await this.diagReadback.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(this.diagReadback.getMappedRange()).slice();
      this.diagReadback.unmap();
      this.applyDiagnostics(bytes);
    } catch (error) {
      console.warn('Temperature diagnostics failed', error);
    } finally {
      this.diagPending = false;
    }
  }

  private applyDiagnostics(bytes: Uint8Array) {
    const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    let minT = Infinity;
    let maxT = -Infinity;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < floats.length; i += 4) {
      const t = floats[i];
      minT = Math.min(minT, t);
      maxT = Math.max(maxT, t);
      sum += t;
      count++;
    }
    const set = (id: string, value: string) => { const el = document.getElementById(id); if (el) el.textContent = value; };
    set('temp-min', this.formatKelvin(minT));
    set('temp-mean', this.formatKelvin(sum / Math.max(count, 1)));
    set('temp-max', this.formatKelvin(maxT));
    set('temp-spread', this.formatRange(maxT - minT));
    const probe = this.latestProbePointer ? this.dirAtScreen(this.latestProbePointer.x, this.latestProbePointer.y) : null;
    if (!probe) {
      set('temp-probe', '--');
      set('temp-anomaly', '--');
      return;
    }
    const p = this.directionToCubeFacePixel(probe, DIAG_FACE);
    const offset = (p.face * DIAG_FACE * DIAG_FACE + p.y * DIAG_FACE + p.x) * 4;
    const temp = floats[offset];
    const anomaly = floats[offset + 2];
    set('temp-probe', this.formatKelvin(temp));
    set('temp-anomaly', `${anomaly >= 0 ? '+' : ''}${anomaly.toFixed(1)} K`);
  }

  private directionToCubeFacePixel(dir: THREE.Vector3, size: number) {
    const ax = Math.abs(dir.x);
    const ay = Math.abs(dir.y);
    const az = Math.abs(dir.z);
    let face = 0;
    let u = 0;
    let v = 0;
    if (ax >= ay && ax >= az) {
      if (dir.x >= 0) { face = 0; u = -dir.z / ax; v = -dir.y / ax; }
      else { face = 1; u = dir.z / ax; v = -dir.y / ax; }
    } else if (ay >= ax && ay >= az) {
      if (dir.y >= 0) { face = 2; u = dir.x / ay; v = dir.z / ay; }
      else { face = 3; u = dir.x / ay; v = -dir.z / ay; }
    } else if (dir.z >= 0) {
      face = 4; u = dir.x / az; v = -dir.y / az;
    } else {
      face = 5; u = -dir.x / az; v = -dir.y / az;
    }
    return {
      face,
      x: THREE.MathUtils.clamp(Math.floor((u * 0.5 + 0.5) * size), 0, size - 1),
      y: THREE.MathUtils.clamp(Math.floor((v * 0.5 + 0.5) * size), 0, size - 1),
    };
  }

  private formatKelvin(value: number) {
    return Number.isFinite(value) ? `${value.toFixed(1)} K` : '--';
  }

  private formatRange(value: number) {
    return Number.isFinite(value) ? `${value.toFixed(1)} K` : '--';
  }

  private makeTextureDebugPanel() {
    if (this.textureDebug.el) return this.textureDebug.el;
    const panel = document.createElement('div');
    panel.id = 'tex-debug';
    const addCard = (key: TextureDebugKey, label: string) => {
      const card = document.createElement('div');
      card.className = 'tex-card';
      const title = document.createElement('div');
      title.className = 'tex-label';
      title.textContent = label;
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 256;
      card.append(title, canvas);
      panel.appendChild(card);
      this.textureDebug.cards[key] = canvas;
    };
    addCard('base', 'uBase day albedo');
    addCard('emissive', 'uNight Black Marble');
    addCard('geo', 'uGeo climate geography');
    addCard('topo', 'uTopo elevation');
    document.body.appendChild(panel);
    this.textureDebug.el = panel;
    return panel;
  }

  private updateTextureDebugPanel() {
    if (!this.textureDebug.el) return;
    for (const [key, canvas] of Object.entries(this.textureDebug.cards)) {
      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const src = this.textureDebug.sources[key as TextureDebugKey];
      if (!src) {
        ctx.fillStyle = '#657284';
        ctx.font = '12px ui-monospace, Menlo, Consolas, monospace';
        ctx.fillText('not bound', 14, 24);
      } else {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
      }
    }
  }

  private showFatal(message: string) {
    const panel = document.createElement('div');
    panel.className = 'fatal-webgpu';
    panel.textContent = message;
    document.body.appendChild(panel);
  }
}
