// Atmospheric circulation on a sphere — a cubemap Navier–Stokes solver turned
// into a toy "general circulation model" of a rotating planet.
//
// On top of the incompressible surface flow we add the physics that makes
// circulation look Earth-like:
//   • Coriolis force  a = -f (n × v),  f = 2Ω·sin(lat)   → cyclones, jets, trades
//   • a temperature field T advected by the wind, relaxed toward a solar
//     equilibrium (hot equator / cold poles, day/night from the sun direction,
//     land heating faster than ocean — sampled from the Earth texture)
//   • thermal-wind forcing  a = κ (n × ∇T)  → wind blows along isotherms.
//     (A plain pressure-gradient force would be removed by the incompressible
//      projection; the rotational n×∇T term survives it and is the physically
//      correct geostrophic/thermal-wind behaviour.)
//
// The fluid can be visualised as temperature, wind speed, vorticity (storms),
// pressure, or the original painted smoke tracer.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const FACE = 384;          // resolution per cube face — higher = finer detail, heavier GPU
const H = 1.5 / FACE;

// visualisation scaling (field value -> colormap input)
const WIND_SCALE = 0.55;
const VORT_SCALE = 0.06;
const PRESS_SCALE = 1.6;

const VIEW = { smoke: 0, temperature: 1, wind: 2, vorticity: 3, pressure: 4, climate: 5 };
const DEFAULT_TIME_SCALE = 0.1;  // 6 simulated minutes per real second; one Earth day takes 4 real minutes

const params = {
  speed: DEFAULT_TIME_SCALE,
  coriolis: 1.0,         // 1.0 = Earth's physical Coriolis rate
  heating: 6.0,
  solar: 62,             // absorbed-sunlight strength (heat in)
  cool: 40,              // outgoing-longwave strength (T^4 radiative cooling)
  greenhouse: 18,        // greenhouse back-radiation floor (keeps night/poles bounded)
  oceanInertia: 7,       // ocean heat capacity vs land (thermal lag)
  curl: 6,
  velDissipation: 0.12,
  denDissipation: 0.9,
  pressureIters: 16,
  brush: 1.2,
  emitters: false,
  spin: 1.0,             // 1.0 = 24-hour solar day; exposed as a multiplier
  dayNight: true,
  viewMode: VIEW.temperature,
  overlay: 0.45,
  specular: 3.0,         // sun-glint (specular) strength
  atmosphere: 16,        // atmospheric scattering brightness
};

// ---------------------------------------------------------------------------
// Renderer / scene
// ---------------------------------------------------------------------------
const appEl = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x050608, 1);
appEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(0, 0.4, 3.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.16;
controls.minDistance = 1.8;
controls.maxDistance = 90;
controls.rotateSpeed = 0.6;
controls.enablePan = false;
controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };

renderer.domElement.style.touchAction = 'none';
renderer.domElement.style.userSelect = 'none';
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (e.button === 2) e.preventDefault();
}, { capture: true });
renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

const SUN_WORLD = new THREE.Vector3(0.6, 0.5, 0.8).normalize();
const EARTH_RADIUS = 1.0;
const EARTH_RADIUS_KM = 6371.0;
const EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0;
const MOON_RADIUS_KM = 1737.4;
const MOON_SEMI_MAJOR_AXIS_KM = 384400.0;
const SUN_RADIUS_KM = 696340.0;
const ASTRONOMICAL_UNIT_KM = 149597870.7;
const EARTH_SOLAR_DAY_SECONDS = 86400.0;
const EARTH_SIDEREAL_DAY_SECONDS = 0.99726968 * EARTH_SOLAR_DAY_SECONDS;
const EARTH_ROTATION_RATE_RAD_PER_SECOND = Math.PI * 2 / EARTH_SIDEREAL_DAY_SECONDS;
const MOON_SIDEREAL_ORBIT_SECONDS = 27.321661 * EARTH_SOLAR_DAY_SECONDS;
const PHYSICAL_SECONDS_PER_REAL_SECOND = 3600.0;      // time-scale multiplier basis: 1.0 = 1 physical hour per real second
const MODEL_VELOCITY_UNIT_MPS = 50.0;                 // one shader velocity unit ~= jet-stream-scale wind
const CORIOLIS_OMEGA_CODE = EARTH_ROTATION_RATE_RAD_PER_SECOND * EARTH_RADIUS_M / MODEL_VELOCITY_UNIT_MPS;
const MOON_RADIUS = (MOON_RADIUS_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;
const MOON_ORBIT_A = (MOON_SEMI_MAJOR_AXIS_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;
const MOON_ORBIT_E = 0.0549;                           // real orbital eccentricity
const MOON_ORBIT_INCLINATION = THREE.MathUtils.degToRad(5.145);
const MOON_INITIAL_ANOMALY = 4.6;
const SUN_RADIUS = (SUN_RADIUS_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;
const SUN_DISTANCE = (ASTRONOMICAL_UNIT_KM / EARTH_RADIUS_KM) * EARTH_RADIUS;
controls.maxDistance = SUN_DISTANCE * 1.6;
const texType = THREE.HalfFloatType;

// 1x1 fallback so sampler2D uniforms are always bound to a valid texture.
const blackTex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, THREE.RGBAFormat);
blackTex.needsUpdate = true;

// ---------------------------------------------------------------------------
// GPGPU plumbing
// ---------------------------------------------------------------------------
const quadScene = new THREE.Scene();
const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quadMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
quadScene.add(quadMesh);

function makeCube() {
  return new THREE.WebGLCubeRenderTarget(FACE, {
    type: texType, format: THREE.RGBAFormat,
    magFilter: THREE.LinearFilter, minFilter: THREE.LinearFilter,
    generateMipmaps: false, depthBuffer: false, stencilBuffer: false,
  });
}
function makeFBO() {
  return { read: makeCube(), write: makeCube(), swap() { const t = this.read; this.read = this.write; this.write = t; } };
}

const velocity = makeFBO();
const dye = makeFBO();
const temperature = makeFBO();
const pressure = makeFBO();
const divergenceRT = makeCube();
const curlRT = makeCube();

function runFaces(material, rt) {
  quadMesh.material = material;
  for (let f = 0; f < 6; f++) {
    material.uniforms.uFace.value = f;
    renderer.setRenderTarget(rt, f);
    renderer.render(quadScene, quadCamera);
  }
}

// ---------------------------------------------------------------------------
// Shader scaffolding
// ---------------------------------------------------------------------------
const VERT = /* glsl */`
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

// Baked-in climatological baseline that MAPS TO THE WORLD'S GEOMETRY:
//   T(lat, elevation, land/ocean) — a simplified but physically-motivated model
//   of annual-mean surface temperature. Shared by the heat sim, the init pass,
//   and the "climate" view so they agree exactly. Needs uGeo (land/ocean colour)
//   and uTopo (elevation) bound; falls back to latitude-only if absent.
const CLIM = /* glsl */`
  vec2 dirToEquirect(vec3 n) {
    float phi = atan(n.z, -n.x);
    float u = phi / (2.0 * PI); if (u < 0.0) u += 1.0;
    float v = 1.0 - acos(clamp(n.y, -1.0, 1.0)) / PI;
    return vec2(u, v);
  }
  float climBaseline(vec3 n) {
    vec2 uv = dirToEquirect(n);
    float lat = asin(clamp(n.y, -1.0, 1.0));
    float c = max(cos(lat), 0.0);

    // 1) sea-level annual-mean temperature by latitude. Values are normalized
    //    so 0..1 maps to roughly 235..320 K in the live slab model.
    float Tsea = 0.04 + 0.78 * pow(c, 1.35);

    // 2) elevation lapse rate: higher terrain is colder (Himalaya, Andes,
    //    Rockies, Greenland & Antarctic ice sheets, East-African highlands).
    float elev = uHasTopo > 0.5 ? texture2D(uTopo, uv).r : 0.0;
    float T = Tsea - 0.65 * elev;

    // 3) continentality: oceans moderate temperature without collapsing every
    //    ocean latitude toward the same value.
    if (uHasGeo > 0.5) {
      vec3 g = texture2D(uGeo, uv).rgb;
      float ocean = smoothstep(0.015, 0.12, g.b - max(g.r, g.g));
      float oceanClimate = 0.14 + 0.56 * pow(c, 1.05);
      T = mix(T, mix(T, oceanClimate, 0.35), ocean);

      float ice = smoothstep(0.55, 0.85, min(g.r, min(g.g, g.b)));
      T -= ice * 0.18;
    }
    return clamp(T, 0.0, 1.0);
  }
`;

const PRELUDE = /* glsl */`
  precision highp float;
  #define PI 3.141592653589793
  varying vec2 vUv;
  uniform float uFace;
  uniform float uH;
  uniform sampler2D uGeo, uTopo;
  uniform float uHasGeo, uHasTopo;
  vec3 faceDir(float face, vec2 uv) {
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;
    vec3 d;
    if      (face < 0.5) d = vec3( 1.0,  -v,  -u);
    else if (face < 1.5) d = vec3(-1.0,  -v,   u);
    else if (face < 2.5) d = vec3(  u, 1.0,    v);
    else if (face < 3.5) d = vec3(  u,-1.0,   -v);
    else if (face < 4.5) d = vec3(  u,  -v,  1.0);
    else                 d = vec3( -u,  -v, -1.0);
    return normalize(d);
  }
  void tbasis(vec3 n, out vec3 t1, out vec3 t2) {
    vec3 up = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    t1 = normalize(cross(up, n));
    t2 = cross(n, t1);
  }
` + CLIM;

function makeMat(fragment, uniforms) {
  return new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: PRELUDE + fragment,
    uniforms: Object.assign({
      uFace: { value: 0 }, uH: { value: H },
      uGeo: { value: blackTex }, uTopo: { value: blackTex },
      uHasGeo: { value: 0 }, uHasTopo: { value: 0 },
    }, uniforms),
    depthTest: false, depthWrite: false,
  });
}

// ---------------------------------------------------------------------------
// Simulation passes
// ---------------------------------------------------------------------------
const advectMat = makeMat(/* glsl */`
  uniform samplerCube uVelocity;
  uniform samplerCube uSource;
  uniform float uDt, uDissipation, uProjectVel;
  void main() {
    vec3 n = faceDir(uFace, vUv);
    vec3 vel = textureCube(uVelocity, n).xyz;
    vec3 back = normalize(n - uDt * vel);
    vec4 s = textureCube(uSource, back);
    vec3 outv = s.xyz;
    if (uProjectVel > 0.5) outv -= dot(outv, n) * n;
    gl_FragColor = vec4(outv / (1.0 + uDissipation * uDt), s.w);
  }
`, {
  uVelocity: { value: null }, uSource: { value: null },
  uDt: { value: 0 }, uDissipation: { value: 0 }, uProjectVel: { value: 0 },
});

// Coriolis + thermal-wind forcing.
const forcesMat = makeMat(/* glsl */`
  uniform samplerCube uVelocity;
  uniform samplerCube uTemp;
  uniform float uOmega, uThermal, uDt;
  void main() {
    vec3 n = faceDir(uFace, vUv); vec3 t1, t2; tbasis(n, t1, t2);
    vec3 v = textureCube(uVelocity, n).xyz;
    float Tp1 = textureCube(uTemp, normalize(n + uH * t1)).x;
    float Tm1 = textureCube(uTemp, normalize(n - uH * t1)).x;
    float Tp2 = textureCube(uTemp, normalize(n + uH * t2)).x;
    float Tm2 = textureCube(uTemp, normalize(n - uH * t2)).x;
    vec3 gradT = ((Tp1 - Tm1) / (2.0 * uH)) * t1 + ((Tp2 - Tm2) / (2.0 * uH)) * t2;
    gradT *= (1.0 / 80.0);                        // T is in Kelvin; gentler so weather perturbs (not shreds) the gradient
    vec3 thermal = uThermal * cross(n, gradT);   // wind along isotherms
    float f = 2.0 * uOmega * n.y;                // Coriolis parameter (axis = +Y)
    vec3 cor = -f * cross(n, v);
    v += (thermal + cor) * uDt;
    v -= dot(v, n) * n;
    float sp = length(v); if (sp > 18.0) v *= 18.0 / sp;
    gl_FragColor = vec4(v, 1.0);
  }
`, { uVelocity: { value: null }, uTemp: { value: null }, uOmega: { value: 0 }, uThermal: { value: 0 }, uDt: { value: 0 } });

// Surface temperature slab model (temperature stored in Kelvin per texel).
// The target climate is the existing geography/elevation baseline plus a
// physical diurnal term around the daily-mean insolation. The slab then relaxes
// toward that target with land/ocean heat-capacity time scales. Diffusion is
// scaled by physical days; applying it every frame was flattening the field.
const heatMat = makeMat(/* glsl */`
  uniform samplerCube uTemp;
  uniform float uSolar, uCool, uGreenhouse, uOceanC, uDt, uDayNight;
  uniform vec3 uSun;

  float dailyMeanInsolation(vec3 n, vec3 sunDir) {
    float lat = asin(clamp(n.y, -1.0, 1.0));
    float decl = asin(clamp(sunDir.y, -1.0, 1.0));
    float sinLat = sin(lat), cosLat = max(abs(cos(lat)), 1e-4);
    float sinDec = sin(decl), cosDec = max(abs(cos(decl)), 1e-4);
    float x = -(sinLat * sinDec) / (cosLat * cosDec);
    float h0 = acos(clamp(x, -1.0, 1.0));
    return max((h0 * sinLat * sinDec + cosLat * cosDec * sin(h0)) / PI, 0.0);
  }

  void main() {
    vec3 n = faceDir(uFace, vUv);
    float T = textureCube(uTemp, n).x;

    float ocean = 0.0;
    float ice = 0.0;
    if (uHasGeo > 0.5) {
      vec3 g = texture2D(uGeo, dirToEquirect(n)).rgb;
      ocean = smoothstep(0.015, 0.12, g.b - max(g.r, g.g));
      ice = smoothstep(0.55, 0.85, min(g.r, min(g.g, g.b)));
    }

    float annual = 235.0 + 85.0 * climBaseline(n);
    float instantMu = max(dot(n, normalize(uSun)), 0.0);
    float meanMu = dailyMeanInsolation(n, normalize(uSun));
    float solarScale = uSolar / 62.0;
    float greenhouseBias = (uGreenhouse - 18.0) * 0.35;
    float diurnalAmp = mix(16.0, 3.0, ocean);
    diurnalAmp = mix(diurnalAmp, 5.0, ice);
    float diurnal = uDayNight > 0.5 ? (instantMu - meanMu) : 0.0;
    float target = annual + greenhouseBias + diurnalAmp * solarScale * diurnal;

    // Land surface air responds on roughly day-scale timing; ocean mixed-layer
    // temperature changes over weeks. The cool slider controls the relaxation
    // rate around the target rather than forcing the whole globe to cool.
    float coolScale = max(uCool / 40.0, 0.08);
    float tauDays = mix(0.85, max(uOceanC, 1.0) * 3.2, ocean);
    tauDays = mix(tauDays, tauDays * 1.8, ice) / coolScale;
    float alpha = 1.0 - exp(-uDt / max(tauDays, 1e-3));
    T += (target - T) * alpha;

    // horizontal thermal diffusion (sub-grid heat mixing) — smooths sharp fronts
    vec3 t1, t2; tbasis(n, t1, t2);
    float lap = textureCube(uTemp, normalize(n + uH * t1)).x
              + textureCube(uTemp, normalize(n - uH * t1)).x
              + textureCube(uTemp, normalize(n + uH * t2)).x
              + textureCube(uTemp, normalize(n - uH * t2)).x - 4.0 * T;
    T += lap * uDt * mix(0.20, 0.55, ocean);

    gl_FragColor = vec4(clamp(T, 120.0, 360.0), 0.0, 0.0, 1.0);
  }
`, {
  uTemp: { value: null },
  uSolar: { value: 0 }, uCool: { value: 0 }, uGreenhouse: { value: 0 }, uOceanC: { value: 12 },
  uDt: { value: 0 }, uDayNight: { value: 1 }, uSun: { value: new THREE.Vector3() },
});

// Initialise the temperature field from the same geography/elevation baseline
// used by the live slab model, in Kelvin.
const tempInitMat = makeMat(/* glsl */`
  void main() {
    vec3 n = faceDir(uFace, vUv);
    float t = 235.0 + 85.0 * climBaseline(n);
    gl_FragColor = vec4(clamp(t, 185.0, 330.0), 0.0, 0.0, 1.0);
  }
`, {});

const curlMat = makeMat(/* glsl */`
  uniform samplerCube uVelocity;
  void main() {
    vec3 n = faceDir(uFace, vUv); vec3 t1, t2; tbasis(n, t1, t2);
    vec3 vp1 = textureCube(uVelocity, normalize(n + uH * t1)).xyz;
    vec3 vm1 = textureCube(uVelocity, normalize(n - uH * t1)).xyz;
    vec3 vp2 = textureCube(uVelocity, normalize(n + uH * t2)).xyz;
    vec3 vm2 = textureCube(uVelocity, normalize(n - uH * t2)).xyz;
    float w = (dot(vp1, t2) - dot(vm1, t2)) / (2.0 * uH)
            - (dot(vp2, t1) - dot(vm2, t1)) / (2.0 * uH);
    gl_FragColor = vec4(w, 0.0, 0.0, 1.0);
  }
`, { uVelocity: { value: null } });

const vorticityMat = makeMat(/* glsl */`
  uniform samplerCube uVelocity;
  uniform samplerCube uCurlTex;
  uniform float uCurlStrength, uDt;
  void main() {
    vec3 n = faceDir(uFace, vUv); vec3 t1, t2; tbasis(n, t1, t2);
    float wp1 = abs(textureCube(uCurlTex, normalize(n + uH * t1)).x);
    float wm1 = abs(textureCube(uCurlTex, normalize(n - uH * t1)).x);
    float wp2 = abs(textureCube(uCurlTex, normalize(n + uH * t2)).x);
    float wm2 = abs(textureCube(uCurlTex, normalize(n - uH * t2)).x);
    float w = textureCube(uCurlTex, n).x;
    vec3 grad = ((wp1 - wm1) / (2.0 * uH)) * t1 + ((wp2 - wm2) / (2.0 * uH)) * t2;
    grad /= (length(grad) + 1e-5);
    vec3 force = uCurlStrength * uH * w * cross(n, grad);
    vec3 vel = textureCube(uVelocity, n).xyz + force * uDt;
    vel -= dot(vel, n) * n;
    gl_FragColor = vec4(vel, 1.0);
  }
`, { uVelocity: { value: null }, uCurlTex: { value: null }, uCurlStrength: { value: 0 }, uDt: { value: 0 } });

const divergenceMat = makeMat(/* glsl */`
  uniform samplerCube uVelocity;
  void main() {
    vec3 n = faceDir(uFace, vUv); vec3 t1, t2; tbasis(n, t1, t2);
    vec3 vp1 = textureCube(uVelocity, normalize(n + uH * t1)).xyz;
    vec3 vm1 = textureCube(uVelocity, normalize(n - uH * t1)).xyz;
    vec3 vp2 = textureCube(uVelocity, normalize(n + uH * t2)).xyz;
    vec3 vm2 = textureCube(uVelocity, normalize(n - uH * t2)).xyz;
    float div = (dot(vp1, t1) - dot(vm1, t1)) / (2.0 * uH)
              + (dot(vp2, t2) - dot(vm2, t2)) / (2.0 * uH);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
  }
`, { uVelocity: { value: null } });

const pressureMat = makeMat(/* glsl */`
  uniform samplerCube uPressure;
  uniform samplerCube uDivergence;
  void main() {
    vec3 n = faceDir(uFace, vUv); vec3 t1, t2; tbasis(n, t1, t2);
    float pp1 = textureCube(uPressure, normalize(n + uH * t1)).x;
    float pm1 = textureCube(uPressure, normalize(n - uH * t1)).x;
    float pp2 = textureCube(uPressure, normalize(n + uH * t2)).x;
    float pm2 = textureCube(uPressure, normalize(n - uH * t2)).x;
    float div = textureCube(uDivergence, n).x;
    gl_FragColor = vec4((pp1 + pm1 + pp2 + pm2 - uH * uH * div) * 0.25, 0.0, 0.0, 1.0);
  }
`, { uPressure: { value: null }, uDivergence: { value: null } });

const gradientMat = makeMat(/* glsl */`
  uniform samplerCube uPressure;
  uniform samplerCube uVelocity;
  void main() {
    vec3 n = faceDir(uFace, vUv); vec3 t1, t2; tbasis(n, t1, t2);
    float pp1 = textureCube(uPressure, normalize(n + uH * t1)).x;
    float pm1 = textureCube(uPressure, normalize(n - uH * t1)).x;
    float pp2 = textureCube(uPressure, normalize(n + uH * t2)).x;
    float pm2 = textureCube(uPressure, normalize(n - uH * t2)).x;
    vec3 grad = ((pp1 - pm1) / (2.0 * uH)) * t1 + ((pp2 - pm2) / (2.0 * uH)) * t2;
    vec3 vel = textureCube(uVelocity, n).xyz - grad;
    vel -= dot(vel, n) * n;
    gl_FragColor = vec4(vel, 1.0);
  }
`, { uPressure: { value: null }, uVelocity: { value: null } });

const splatMat = makeMat(/* glsl */`
  uniform samplerCube uTarget;
  uniform vec3 uPoint, uValue;
  uniform float uRadius, uProject;
  void main() {
    vec3 n = faceDir(uFace, vUv);
    float ang = acos(clamp(dot(n, uPoint), -1.0, 1.0));
    float fall = exp(-(ang * ang) / uRadius);
    vec3 add = uValue;
    if (uProject > 0.5) add -= dot(add, n) * n;
    gl_FragColor = vec4(textureCube(uTarget, n).xyz + fall * add, 1.0);
  }
`, {
  uTarget: { value: null }, uPoint: { value: new THREE.Vector3() },
  uValue: { value: new THREE.Vector3() }, uRadius: { value: 0.01 }, uProject: { value: 0 },
});

const clearMat = makeMat(`uniform float uValue; void main(){ gl_FragColor = vec4(uValue); }`, { uValue: { value: 0.0 } });

// ---------------------------------------------------------------------------
// Solver step
// ---------------------------------------------------------------------------
function splat(fbo, point, value, radius, project) {
  splatMat.uniforms.uTarget.value = fbo.read.texture;
  splatMat.uniforms.uPoint.value.copy(point);
  splatMat.uniforms.uValue.value.copy(value);
  splatMat.uniforms.uRadius.value = radius;
  splatMat.uniforms.uProject.value = project ? 1 : 0;
  runFaces(splatMat, fbo.write);
  fbo.swap();
}

function step(advectDt, thermalDt) {
  // advect velocity
  advectMat.uniforms.uVelocity.value = velocity.read.texture;
  advectMat.uniforms.uSource.value = velocity.read.texture;
  advectMat.uniforms.uDt.value = advectDt;
  advectMat.uniforms.uDissipation.value = params.velDissipation;
  advectMat.uniforms.uProjectVel.value = 1;
  runFaces(advectMat, velocity.write); velocity.swap();

  // Coriolis + thermal wind
  forcesMat.uniforms.uVelocity.value = velocity.read.texture;
  forcesMat.uniforms.uTemp.value = temperature.read.texture;
  forcesMat.uniforms.uOmega.value = CORIOLIS_OMEGA_CODE * params.coriolis;
  forcesMat.uniforms.uThermal.value = params.heating;
  forcesMat.uniforms.uDt.value = advectDt;
  runFaces(forcesMat, velocity.write); velocity.swap();

  // vorticity confinement (small-scale detail)
  curlMat.uniforms.uVelocity.value = velocity.read.texture;
  runFaces(curlMat, curlRT);
  vorticityMat.uniforms.uVelocity.value = velocity.read.texture;
  vorticityMat.uniforms.uCurlTex.value = curlRT.texture;
  vorticityMat.uniforms.uCurlStrength.value = params.curl;
  vorticityMat.uniforms.uDt.value = advectDt;
  runFaces(vorticityMat, velocity.write); velocity.swap();

  // projection -> divergence-free
  divergenceMat.uniforms.uVelocity.value = velocity.read.texture;
  runFaces(divergenceMat, divergenceRT);
  runFaces(clearMat, pressure.write); pressure.swap();
  for (let i = 0; i < params.pressureIters; i++) {
    pressureMat.uniforms.uPressure.value = pressure.read.texture;
    pressureMat.uniforms.uDivergence.value = divergenceRT.texture;
    runFaces(pressureMat, pressure.write); pressure.swap();
  }
  gradientMat.uniforms.uPressure.value = pressure.read.texture;
  gradientMat.uniforms.uVelocity.value = velocity.read.texture;
  runFaces(gradientMat, velocity.write); velocity.swap();

  // advect temperature (pure transport), then heat/relax
  advectMat.uniforms.uVelocity.value = velocity.read.texture;
  advectMat.uniforms.uSource.value = temperature.read.texture;
  advectMat.uniforms.uDissipation.value = 0;
  advectMat.uniforms.uProjectVel.value = 0;
  runFaces(advectMat, temperature.write); temperature.swap();

  heatMat.uniforms.uTemp.value = temperature.read.texture;
  heatMat.uniforms.uSolar.value = params.solar;
  heatMat.uniforms.uCool.value = params.cool;
  heatMat.uniforms.uGreenhouse.value = params.greenhouse;
  heatMat.uniforms.uOceanC.value = params.oceanInertia;
  heatMat.uniforms.uDt.value = thermalDt;
  heatMat.uniforms.uDayNight.value = params.dayNight ? 1 : 0;
  heatMat.uniforms.uSun.value.copy(sunObject());
  runFaces(heatMat, temperature.write); temperature.swap();

  // advect dye tracer
  advectMat.uniforms.uVelocity.value = velocity.read.texture;
  advectMat.uniforms.uSource.value = dye.read.texture;
  advectMat.uniforms.uDissipation.value = params.denDissipation;
  advectMat.uniforms.uProjectVel.value = 0;
  runFaces(advectMat, dye.write); dye.swap();
}

// sun direction in the planet's (object) frame
const _q = new THREE.Quaternion();
const _sunObj = new THREE.Vector3();
function sunObject() {
  _q.copy(sphere.quaternion).invert();
  return _sunObj.copy(SUN_WORLD).applyQuaternion(_q);
}

function seedVelocity() {
  // small random tangential gusts to break zonal symmetry into eddies
  const p = new THREE.Vector3(), v = new THREE.Vector3(), r = new THREE.Vector3();
  for (let i = 0; i < 20; i++) {
    p.set(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1).normalize();
    r.set(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
    v.copy(r).addScaledVector(p, -p.dot(r)).normalize().multiplyScalar(1.2 + Math.random());
    splat(velocity, p, v, 0.01, true);
  }
}

function reset() {
  for (const fbo of [velocity, dye, pressure]) { runFaces(clearMat, fbo.read); runFaces(clearMat, fbo.write); }
  runFaces(clearMat, divergenceRT);
  runFaces(clearMat, curlRT);
  runFaces(tempInitMat, temperature.read);
  runFaces(tempInitMat, temperature.write);
  seedVelocity();
}

// ---------------------------------------------------------------------------
// Colormaps (shared by sphere + unwrap)
// ---------------------------------------------------------------------------
const CMAP = /* glsl */`
  vec3 cmapTemp(float x){
    x = clamp(x, 0.0, 1.0);
    vec3 c0=vec3(0.05,0.18,0.55), c1=vec3(0.10,0.55,0.85), c2=vec3(0.95,0.93,0.70),
         c3=vec3(0.92,0.45,0.12), c4=vec3(0.75,0.06,0.06);
    if (x<0.25) return mix(c0,c1,x/0.25);
    if (x<0.50) return mix(c1,c2,(x-0.25)/0.25);
    if (x<0.75) return mix(c2,c3,(x-0.50)/0.25);
    return mix(c3,c4,(x-0.75)/0.25);
  }
  vec3 cmapDiv(float x){
    x = clamp(x, 0.0, 1.0);
    vec3 neg=vec3(0.15,0.40,0.90), mid=vec3(0.04,0.05,0.07), pos=vec3(0.95,0.35,0.18);
    return x<0.5 ? mix(neg,mid,x*2.0) : mix(mid,pos,(x-0.5)*2.0);
  }
  vec3 cmapSpeed(float x){
    x = clamp(x, 0.0, 1.0);
    vec3 a=vec3(0.02,0.03,0.09), b=vec3(0.10,0.40,0.60), c=vec3(0.40,0.85,0.95), d=vec3(1.0,1.0,0.92);
    if (x<0.4) return mix(a,b,x/0.4);
    if (x<0.75) return mix(b,c,(x-0.4)/0.35);
    return mix(c,d,(x-0.75)/0.25);
  }
`;

// Physically-based (Cook-Torrance GGX) specular highlight from the sun, with
// per-pixel material: oceans are smooth (tight bright sun-glint, low F0 water),
// land is rough (broad & dim), snow/ice is in between. Needs uGeo/uHasGeo,
// uUseBase, uSpecStrength and #define PI in the including shader.
const SPEC = /* glsl */`
  vec3 sunSpecular(vec3 N, vec3 V, vec3 L, vec2 uv) {
    if (uUseBase < 0.5) return vec3(0.0);
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);

    float ocean = 0.0, snow = 0.0;
    if (uHasGeo > 0.5) {
      vec3 g = texture2D(uGeo, uv).rgb;
      ocean = smoothstep(0.015, 0.12, g.b - max(g.r, g.g));   // bluest = water
      float mn = min(g.r, min(g.g, g.b));
      snow = smoothstep(0.55, 0.85, mn);                      // bright+white = ice/snow
    }
    float rough = mix(0.62, 0.06, ocean);   // land rough, ocean glassy
    rough = mix(rough, 0.30, snow);
    float F0 = mix(0.04, 0.02, ocean);      // dielectric; water reflects ~2%

    vec3 H = normalize(L + V);
    float NdotV = max(dot(N, V), 1e-3);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float a = rough * rough; float a2 = a * a;
    float dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);                            // GGX distribution
    float k = (rough + 1.0); k = k * k * 0.125;               // Smith-Schlick geometry
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    float G = gv * gl;
    float F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);        // Schlick Fresnel
    float spec = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4) * NdotL;

    float matRefl = max(ocean, snow * 0.7) + 0.06;            // land barely glossy
    return vec3(1.0, 0.95, 0.85) * spec * matRefl * uSpecStrength;
  }
`;

// fragment body that, given a direction `dir` and equirect uv `bUv`, returns the
// final color. Shared logic so sphere + unwrap render identically.
function fieldBody(use3dLighting) {
  return /* glsl */`
    // How lit a point is. uDayNight=1 -> realistic terminator (night side goes
    // dark); uDayNight=0 -> evenly lit everywhere (for data viewing).
    float dayAmount(float sun) { return mix(1.0, smoothstep(-0.05, 0.22, sun), uDayNight); }
    float litAmount(float sun) { return mix(1.0, clamp(sun, 0.0, 1.0), uDayNight); }

    vec3 cityLightSample(vec2 uv) {
      vec3 raw = texture2D(uNight, vec2(fract(uv.x), clamp(uv.y, 0.0, 1.0))).rgb;
      float radiance = max(raw.r, max(raw.g, raw.b));
      float city = max(radiance - 0.018, 0.0) / 0.982;
      city = pow(clamp(city, 0.0, 1.0), 1.35);
      city *= smoothstep(0.018, 0.08, radiance);
      return vec3(1.0, 0.64, 0.34) * city;
    }

    vec3 nightEmission(vec2 bUv, float sun) {
      if (uHasNight < 0.5) return vec3(0.0);
      float night = uDayNight * (1.0 - smoothstep(-0.14, 0.07, sun));
      return cityLightSample(bUv) * 1.85 * night;
    }

    // Day/night Earth: the blue-marble texture remains the physical surface
    // albedo. uNight is a Black Marble radiance map used only for emission.
    vec3 earthBase(vec2 bUv, float sun, float fres) {
      float lit = litAmount(sun);
      vec3 col;
      if (uUseBase > 0.5) {
        vec3 dayTex = pow(texture2D(uBase, bUv).rgb, vec3(2.2));
        vec3 nightAlbedo = dayTex * vec3(0.018, 0.023, 0.032) * (uDayNight * (1.0 - lit));
        col = dayTex * (1.28 * lit) + nightAlbedo;
        col += nightEmission(bUv, sun);
      } else {
        float day = dayAmount(sun);
        col = mix(vec3(0.0), vec3(0.05,0.08,0.13), lit) * mix(1.0, day, uDayNight);
      }
      // (atmospheric limb glow is now a real scattering pass on a separate shell)
      return col;
    }

    vec3 fieldColor(vec3 dir, vec2 bUv, float sun, float fres, vec3 specCol) {
      vec3 earthCol = earthBase(bUv, sun, fres) + specCol;   // sun glint on the surface
      float day = dayAmount(sun);
      vec3 result = earthCol;
      if (uViewMode < 0.5) {
        vec3 smoke = max(textureCube(uSmoke, dir).rgb, 0.0);
        float d = clamp(length(smoke), 0.0, 1.0);
        result = earthCol + smoke * (0.9 + 0.6 * d) * mix(1.0, 0.01 + 0.99 * day, uDayNight);
      } else {
        vec3 fc = vec3(0.0);
        float inten = uOverlay;                       // how opaque the fluid is
        if (uViewMode < 1.5) {
          float Tk = textureCube(uTemp, dir).x;
          fc = cmapTemp((Tk - 245.0) / 65.0);   // live temp, tuned around Earth surface-air range
          inten = max(inten, 0.68);
        } else if (uViewMode < 2.5) {
          float s = length(textureCube(uVel, dir).xyz) * uWindScale;
          fc = cmapSpeed(s); inten *= clamp(s, 0.0, 1.0);
        } else if (uViewMode < 3.5) {
          float w = textureCube(uCurl, dir).x * uVortScale;
          fc = cmapDiv(0.5 + 0.5 * w); inten *= clamp(abs(w), 0.0, 1.0);
        } else if (uViewMode < 4.5) {
          float p = textureCube(uPress, dir).x * uPressScale;
          fc = cmapDiv(0.5 + 0.5 * p); inten *= clamp(abs(p), 0.0, 1.0);
        } else {
          fc = cmapTemp(climBaseline(dir));            // baked climatological baseline
        }
        // fluid is lit by the sun and fades out over the night side
        float lit = mix(1.0, clamp(sun, 0.0, 1.0), uDayNight);
        inten *= mix(1.0, day, uDayNight);
        result = mix(earthCol, fc * lit, inten);      // fluid as a translucent layer
      }
      return result;
    }
  `;
}

const FIELD_UNIFORMS = () => ({
  uSmoke: { value: dye.read.texture },
  uTemp: { value: temperature.read.texture },
  uVel: { value: velocity.read.texture },
  uCurl: { value: curlRT.texture },
  uPress: { value: pressure.read.texture },
  uBase: { value: blackTex },
  uNight: { value: blackTex },
  uGeo: { value: blackTex },
  uTopo: { value: blackTex },
  uUseBase: { value: 0 },
  uHasNight: { value: 0 },
  uHasGeo: { value: 0 },
  uHasTopo: { value: 0 },
  uViewMode: { value: params.viewMode },
  uOverlay: { value: params.overlay },
  uDayNight: { value: params.dayNight ? 1 : 0 },
  uSunObj: { value: new THREE.Vector3() },
  uSunWorld: { value: SUN_WORLD },
  uSpecStrength: { value: params.specular },
  uWindScale: { value: WIND_SCALE },
  uVortScale: { value: VORT_SCALE },
  uPressScale: { value: PRESS_SCALE },
});

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------
const sphereGeo = new THREE.SphereGeometry(1, 160, 96);
const sphereMat = new THREE.ShaderMaterial({
  uniforms: FIELD_UNIFORMS(),
  vertexShader: /* glsl */`
    varying vec3 vDir; varying vec3 vNormalW; varying vec3 vPosW; varying vec2 vUv;
    void main() {
      vDir = normalize(position);
      vUv = uv;
      vNormalW = normalize(mat3(modelMatrix) * normal);
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vPosW = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
  fragmentShader: /* glsl */`
    precision highp float;
    #define PI 3.141592653589793
    varying vec3 vDir; varying vec3 vNormalW; varying vec3 vPosW; varying vec2 vUv;
    uniform samplerCube uSmoke, uTemp, uVel, uCurl, uPress;
    uniform sampler2D uBase, uNight, uGeo, uTopo;
    uniform float uUseBase, uHasNight, uHasGeo, uHasTopo, uViewMode, uOverlay, uDayNight, uWindScale, uVortScale, uPressScale, uSpecStrength;
    uniform vec3 uSunObj, uSunWorld;
    ${CLIM}
    ${CMAP}
    ${SPEC}
    ${fieldBody(true)}
    void main() {
      vec3 dir = normalize(vDir);
      vec3 N = normalize(vNormalW);
      vec3 V = normalize(cameraPosition - vPosW);
      float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
      float sun = dot(dir, normalize(uSunObj));
      vec3 spec = sunSpecular(N, V, normalize(uSunWorld), vUv);
      vec3 col = fieldColor(dir, vUv, sun, fres, spec);
      col = col / (col + 1.0);
      col = pow(col, vec3(0.4545));
      gl_FragColor = vec4(col, 1.0);
    }
  `,
});
const sphere = new THREE.Mesh(sphereGeo, sphereMat);
scene.add(sphere);

// ---------------------------------------------------------------------------
// Atmosphere — physically-based single-scattering on a shell around the planet.
// For each pixel the view ray is marched through the atmosphere shell; at each
// sample the optical depth toward the sun is integrated, Rayleigh (wavelength
// dependent, ~1/λ^4 -> blue) and Mie (forward-scattering haze) in-scattering are
// accumulated with their phase functions, and the planet casts a shadow. This
// gives a blue limb that bleeds past the disk, a bright sunlit edge, forward
// glow toward the sun, and red sunset tints at the terminator — all emergent.
// ---------------------------------------------------------------------------
const RP = 1.0, RA = 1.10;
const atmosphereMat = new THREE.ShaderMaterial({
  uniforms: {
    uSunWorld: { value: SUN_WORLD },
    uStrength: { value: params.atmosphere },
  },
  transparent: true,
  blending: THREE.AdditiveBlending,
  side: THREE.BackSide,
  depthWrite: false,
  depthTest: false,
  vertexShader: /* glsl */`
    varying vec3 vWorldPos;
    void main() {
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
  fragmentShader: /* glsl */`
    precision highp float;
    #define PI 3.141592653589793
    varying vec3 vWorldPos;
    uniform vec3 uSunWorld;
    uniform float uStrength;

    const float RP = ${RP.toFixed(3)};
    const float RA = ${RA.toFixed(3)};
    const int VIEW_STEPS = 12;
    const int SUN_STEPS = 4;
    const float hR = 0.30;                          // Rayleigh scale height (shell fraction)
    const float hM = 0.10;                          // Mie scale height
    const vec3  betaR = vec3(5.8, 13.5, 33.1) * 0.05;   // ~1/λ^4 -> blue scatters most
    const float betaM = 1.1;
    const float gM = 0.76;                          // Mie forward-scatter anisotropy

    vec2 raySphere(vec3 ro, vec3 rd, float r) {
      float b = dot(ro, rd);
      float c = dot(ro, ro) - r * r;
      float d = b * b - c;
      if (d < 0.0) return vec2(1.0, -1.0);
      d = sqrt(d);
      return vec2(-b - d, -b + d);
    }

    void main() {
      vec3 ro = cameraPosition;
      vec3 rd = normalize(vWorldPos - cameraPosition);
      vec3 L = normalize(uSunWorld);

      vec2 atmo = raySphere(ro, rd, RA);
      if (atmo.y < 0.0) discard;
      float tNear = max(atmo.x, 0.0);
      float tFar = atmo.y;
      vec2 planet = raySphere(ro, rd, RP);
      if (planet.x < planet.y && planet.y > 0.0) tFar = min(tFar, max(planet.x, 0.0));
      if (tFar <= tNear) discard;

      float ds = (tFar - tNear) / float(VIEW_STEPS);
      float t = tNear + ds * 0.5;
      float odViewR = 0.0, odViewM = 0.0;
      vec3 inR = vec3(0.0);
      float inM = 0.0;

      for (int i = 0; i < VIEW_STEPS; i++) {
        vec3 p = ro + rd * t;
        float h = clamp((length(p) - RP) / (RA - RP), 0.0, 1.0);
        float dR = exp(-h / hR) * ds;
        float dM = exp(-h / hM) * ds;
        odViewR += dR; odViewM += dM;

        // planet blocks direct sunlight at this sample?
        vec2 shp = raySphere(p, L, RP);
        if (!(shp.x > 0.0 && shp.x < shp.y)) {
          vec2 sa = raySphere(p, L, RA);
          float dss = max(sa.y, 0.0) / float(SUN_STEPS);
          float ts = dss * 0.5;
          float odSunR = 0.0, odSunM = 0.0;
          for (int j = 0; j < SUN_STEPS; j++) {
            vec3 q = p + L * ts;
            float hq = clamp((length(q) - RP) / (RA - RP), 0.0, 1.0);
            odSunR += exp(-hq / hR) * dss;
            odSunM += exp(-hq / hM) * dss;
            ts += dss;
          }
          vec3 tau = betaR * (odViewR + odSunR) + betaM * (odViewM + odSunM);
          vec3 trans = exp(-tau);
          inR += trans * dR;
          inM += (trans.g) * dM;
        }
        t += ds;
      }

      float mu = dot(rd, L);
      float phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
      float g2 = gM * gM;
      float phaseM = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) /
                     ((2.0 + g2) * pow(1.0 + g2 - 2.0 * gM * mu, 1.5));

      vec3 col = uStrength * (betaR * phaseR * inR + betaM * phaseM * inM);
      col = 1.0 - exp(-col);                         // soft tonemap for additive blend
      gl_FragColor = vec4(col, 1.0);
    }
  `,
});
const atmosphere = new THREE.Mesh(new THREE.SphereGeometry(RA, 96, 64), atmosphereMat);
atmosphere.renderOrder = 2;
scene.add(atmosphere);

// ---------------------------------------------------------------------------
// Star skysphere — fully procedural, shader-rendered stars. No bitmap texture,
// so stars stay crisp at any viewport size and orbit zoom level.
// ---------------------------------------------------------------------------
const starSphere = new THREE.Mesh(
  new THREE.SphereGeometry(500, 128, 64),
  new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      precision highp float;
      #define PI 3.141592653589793
      varying vec3 vDir;
      uniform float uTime;

      float hash12(vec2 p) {
        vec3 p3 = fract(vec3(p.xyx) * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
      }
      vec2 hash22(vec2 p) {
        return vec2(hash12(p), hash12(p + 19.19));
      }
      vec3 starTemp(float t) {
        vec3 warm = vec3(1.00, 0.70, 0.42);
        vec3 white = vec3(1.00, 0.96, 0.82);
        vec3 blue = vec3(0.62, 0.76, 1.00);
        return t < 0.55 ? mix(warm, white, t / 0.55) : mix(white, blue, (t - 0.55) / 0.45);
      }

      void main() {
        vec3 dir = normalize(vDir);
        vec2 uv = vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5, asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5);

        float bandCenter = 0.5 + 0.08 * sin(uv.x * 13.0 + 0.8) + 0.035 * sin(uv.x * 31.0);
        float band = exp(-pow((uv.y - bandCenter) / 0.055, 2.0));
        vec3 col = vec3(0.00005, 0.00007, 0.00016) + band * vec3(0.0018, 0.0015, 0.0024);

        vec2 grid = vec2(620.0, 310.0);
        vec2 p = uv * grid;
        vec2 base = floor(p);

        for (int yy = -1; yy <= 1; yy++) {
          for (int xx = -1; xx <= 1; xx++) {
            vec2 cell = base + vec2(float(xx), float(yy));
            vec2 rnd = hash22(cell);
            float localBand = exp(-pow((fract((cell.y + rnd.y) / grid.y) - bandCenter) / 0.075, 2.0));
            float density = 0.010 + localBand * 0.034;
            if (hash12(cell + 7.17) < density) {
              vec2 center = cell + rnd;
              vec2 d = p - center;
              float bright = pow(hash12(cell + 41.7), 3.3);
              float radius = mix(0.007, 0.026, bright);
              float d2 = dot(d, d);
              float core = exp(-d2 / (radius * radius));
              float halo = exp(-d2 / (radius * radius * 9.0)) * 0.045;
              float twinkle = 0.88 + 0.12 * sin(uTime * (0.6 + bright * 3.0) + hash12(cell + 91.2) * 6.28318);
              vec3 temp = starTemp(hash12(cell + 23.8));
              col += temp * (core * 1.9 + halo) * (0.08 + bright * 1.45) * twinkle;
            }
          }
        }

        col = col / (col + vec3(1.0));
        gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
      }
    `,
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
  })
);
starSphere.renderOrder = -100;
scene.add(starSphere);

// ---------------------------------------------------------------------------
// The sun — physical Sun/Earth radius and 1 AU distance in scene units, fixed
// in world space along SUN_WORLD. Bloom now comes from the HDR photosphere, and
// lens artifacts are generated in a post-process shader rather than canvas art.
// ---------------------------------------------------------------------------
const sunPos = SUN_WORLD.clone().multiplyScalar(SUN_DISTANCE);
const sunCore = new THREE.Mesh(
  new THREE.SphereGeometry(SUN_RADIUS, 64, 32),
  new THREE.MeshBasicMaterial({
    color: new THREE.Color(18.0, 15.2, 8.8),
    depthWrite: false,
    toneMapped: false,
  })
);
sunCore.position.copy(sunPos);
scene.add(sunCore);
const sunLight = new THREE.DirectionalLight(0xfff4e0, 2.0);
sunLight.position.copy(sunPos);
scene.add(sunLight);

const _sunClip = new THREE.Vector3();
const _sunView = new THREE.Vector3();
const _sunRay = new THREE.Vector3();
const _camToSun = new THREE.Vector3();
const _camToEarth = new THREE.Vector3();
const _sunUv = new THREE.Vector2(0.5, 0.5);

function smoothstep(edge0, edge1, x) {
  const t = THREE.MathUtils.clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function circleOverlapArea(r0, r1, d) {
  if (d >= r0 + r1) return 0;
  if (d <= Math.abs(r1 - r0)) {
    const r = Math.min(r0, r1);
    return Math.PI * r * r;
  }
  const r02 = r0 * r0, r12 = r1 * r1;
  const a0 = Math.acos(THREE.MathUtils.clamp((d * d + r02 - r12) / (2 * d * r0), -1, 1));
  const a1 = Math.acos(THREE.MathUtils.clamp((d * d + r12 - r02) / (2 * d * r1), -1, 1));
  const lens = 0.5 * Math.sqrt(Math.max(0, (-d + r0 + r1) * (d + r0 - r1) * (d - r0 + r1) * (d + r0 + r1)));
  return r02 * a0 + r12 * a1 - lens;
}

function sunVisibility() {
  _camToSun.subVectors(sunPos, camera.position);
  const distSun = _camToSun.length();
  _sunRay.copy(_camToSun).multiplyScalar(1 / distSun);
  const distEarth = camera.position.length();
  _camToEarth.copy(camera.position).multiplyScalar(-1 / distEarth);

  const sunAng = Math.asin(THREE.MathUtils.clamp(SUN_RADIUS / distSun, 0, 0.999));
  const earthAng = Math.asin(THREE.MathUtils.clamp(1.12 / distEarth, 0, 0.999));
  const sep = _sunRay.angleTo(_camToEarth);
  const sunArea = Math.PI * sunAng * sunAng;
  const occulted = circleOverlapArea(sunAng, earthAng, sep) / Math.max(sunArea, 1e-8);
  const disk = THREE.MathUtils.clamp(1 - occulted, 0, 1);

  const limbDistance = Math.abs(sep - earthAng);
  const limbScatter = (1 - smoothstep(sunAng + 0.004, sunAng + 0.035, limbDistance)) *
    (1 - smoothstep(earthAng + sunAng + 0.02, earthAng + sunAng + 0.12, sep));
  return {
    disk,
    bloom: Math.max(disk, limbScatter * 0.18),
    distSun,
    sunAng,
  };
}

function sunScreenState() {
  _sunView.copy(sunPos).applyMatrix4(camera.matrixWorldInverse);
  if (_sunView.z >= -camera.near) {
    return { visible: false, uv: _sunUv.set(0.5, 0.5), bloom: 0, ghost: 0 };
  }

  _sunClip.copy(sunPos).project(camera);
  _sunUv.set(_sunClip.x * 0.5 + 0.5, _sunClip.y * 0.5 + 0.5);

  const edge = Math.max(Math.abs(_sunClip.x), Math.abs(_sunClip.y));
  const softOffscreen = 1 - smoothstep(1.02, 2.35, edge);
  const artifactFade = 1 - smoothstep(0.98, 1.42, edge);
  const visibility = sunVisibility();
  const distanceExposure = THREE.MathUtils.clamp(1.0 - camera.position.length() / (SUN_DISTANCE * 0.92), 0.35, 1.0);

  return {
    visible: softOffscreen > 0.001 && visibility.bloom > 0.001,
    uv: _sunUv,
    bloom: Math.pow(visibility.bloom * softOffscreen, 0.58) * distanceExposure,
    ghost: Math.pow(visibility.disk * artifactFade, 0.78) * distanceExposure,
  };
}

const LensOpticsShader = {
  uniforms: {
    tDiffuse: { value: null },
    uSunUv: { value: new THREE.Vector2(0.5, 0.5) },
    uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
    uBloomVisibility: { value: 0 },
    uGhostVisibility: { value: 0 },
    uTime: { value: 0 },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */`
    precision highp float;
    uniform sampler2D tDiffuse;
    uniform vec2 uSunUv;
    uniform vec2 uResolution;
    uniform float uBloomVisibility;
    uniform float uGhostVisibility;
    uniform float uTime;
    varying vec2 vUv;

    float gauss(vec2 p, float radius) {
      float r2 = max(radius * radius, 1e-6);
      return exp(-dot(p, p) / r2);
    }

    vec2 aspectVec(vec2 p, float aspect) {
      return vec2(p.x * aspect, p.y);
    }

    vec3 chromaGhost(vec2 uv, vec2 center, float radius, vec3 tint, float strength, vec2 axis, float aspect) {
      vec2 dir = normalize(aspectVec(axis, aspect) + vec2(1e-5, 0.0));
      vec2 uvDir = vec2(dir.x / aspect, dir.y);
      float r = gauss(aspectVec(uv - center - uvDir * radius * 0.25, aspect), radius * 1.12);
      float g = gauss(aspectVec(uv - center, aspect), radius);
      float b = gauss(aspectVec(uv - center + uvDir * radius * 0.22, aspect), radius * 1.18);
      float falloff = 1.0 - smoothstep(0.0, 1.1, length(aspectVec(center - vec2(0.5), aspect)));
      return vec3(r, g, b) * tint * strength * falloff;
    }

    void main() {
      vec3 color = texture2D(tDiffuse, vUv).rgb;
      float aspect = uResolution.x / max(uResolution.y, 1.0);
      vec2 axis = vec2(0.5) - uSunUv;
      vec2 p = aspectVec(vUv - uSunUv, aspect);
      float d = length(p);
      float angle = atan(p.y, p.x);

      float broadGlare = 0.090 / (1.0 + d * d * 120.0);
      broadGlare += gauss(p, 0.060) * 0.22;
      broadGlare += gauss(p, 0.190) * 0.035;
      float airyRing = exp(-abs(d - 0.052) * 72.0) * 0.010;
      float apertureMod = 0.94 + 0.06 * cos(6.0 * angle + 0.25 + uTime * 0.015);
      color += vec3(1.00, 0.82, 0.50) * (broadGlare * apertureMod + airyRing) * uBloomVisibility;

      float veil = gauss(aspectVec(vUv - vec2(0.5), aspect), 0.72) * 0.014 * uBloomVisibility;
      color += vec3(1.00, 0.78, 0.48) * veil;

      vec3 ghosts = vec3(0.0);
      ghosts += chromaGhost(vUv, uSunUv + axis * 0.30, 0.018, vec3(0.70, 1.00, 0.76), 0.32, axis, aspect);
      ghosts += chromaGhost(vUv, uSunUv + axis * 0.52, 0.026, vec3(0.56, 0.78, 1.00), 0.26, axis, aspect);
      ghosts += chromaGhost(vUv, uSunUv + axis * 0.82, 0.038, vec3(1.00, 0.72, 0.48), 0.18, axis, aspect);
      ghosts += chromaGhost(vUv, uSunUv + axis * 1.16, 0.055, vec3(0.88, 0.52, 1.00), 0.11, axis, aspect);
      color += ghosts * uGhostVisibility;

      gl_FragColor = vec4(color, 1.0);
    }
  `,
};

const composerTarget = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
  type: THREE.HalfFloatType,
  format: THREE.RGBAFormat,
  depthBuffer: true,
  stencilBuffer: false,
});
composerTarget.texture.name = 'main-hdr-postprocess';
const composer = new EffectComposer(renderer, composerTarget);
composer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
composer.setSize(window.innerWidth, window.innerHeight);
const renderPass = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.62, 0.72, 1.0);
bloomPass.threshold = 1.05;
bloomPass.strength = 0.62;
bloomPass.radius = 0.72;
const lensOpticsPass = new ShaderPass(LensOpticsShader);
composer.addPass(renderPass);
composer.addPass(bloomPass);
composer.addPass(lensOpticsPass);

function resizePostProcessing() {
  composer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  composer.setSize(window.innerWidth, window.innerHeight);
  lensOpticsPass.uniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
}

function updateLensOptics(now) {
  const state = sunScreenState();
  lensOpticsPass.uniforms.uSunUv.value.copy(state.uv);
  lensOpticsPass.uniforms.uBloomVisibility.value = state.visible ? state.bloom : 0;
  lensOpticsPass.uniforms.uGhostVisibility.value = state.visible ? state.ghost : 0;
  lensOpticsPass.uniforms.uTime.value = now * 0.001;
}

// ---------------------------------------------------------------------------
// Moon — real radius and orbit ratios, eccentric inclined orbit, synchronous
// rotation, and scene lighting from the same directional sun so phases are
// physically driven by the shared simulation clock.
// ---------------------------------------------------------------------------
function makeMoonTextures() {
  const size = 768;
  const albedo = document.createElement('canvas');
  const bump = document.createElement('canvas');
  albedo.width = albedo.height = bump.width = bump.height = size;
  const ga = albedo.getContext('2d');
  const gb = bump.getContext('2d');
  ga.fillStyle = '#777872';
  ga.fillRect(0, 0, size, size);
  gb.fillStyle = '#7f7f7f';
  gb.fillRect(0, 0, size, size);

  const image = ga.getImageData(0, 0, size, size);
  const data = image.data;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const latShade = 1.0 - Math.abs(y / size - 0.5) * 0.10;
      const grain = (Math.random() - 0.5) * 20;
      const v = THREE.MathUtils.clamp(116 * latShade + grain, 72, 165);
      data[i] = v * 1.02; data[i + 1] = v; data[i + 2] = v * 0.94; data[i + 3] = 255;
    }
  }
  ga.putImageData(image, 0, 0);

  function crater(ctx, x, y, r, dark, light) {
    const floor = ctx.createRadialGradient(x, y, r * 0.08, x, y, r);
    floor.addColorStop(0.0, dark);
    floor.addColorStop(0.55, dark);
    floor.addColorStop(0.72, light);
    floor.addColorStop(1.0, 'rgba(127,127,127,0)');
    ctx.fillStyle = floor;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  for (let i = 0; i < 170; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const large = Math.random() < 0.16;
    const r = large ? 20 + Math.random() * 58 : 4 + Math.random() * 16;
    const shade = large ? 0.22 + Math.random() * 0.18 : 0.12 + Math.random() * 0.16;
    crater(ga, x, y, r, `rgba(48,48,46,${shade})`, `rgba(205,204,190,${shade * 0.55})`);
    crater(gb, x, y, r, 'rgba(54,54,54,0.55)', 'rgba(232,232,232,0.45)');
  }

  const mare = [
    [0.58, 0.44, 96, 0.28],
    [0.42, 0.51, 68, 0.20],
    [0.68, 0.58, 58, 0.18],
    [0.52, 0.32, 44, 0.17],
  ];
  for (const [u, v, r, a] of mare) {
    const grd = ga.createRadialGradient(u * size, v * size, 0, u * size, v * size, r);
    grd.addColorStop(0.0, `rgba(42,44,45,${a})`);
    grd.addColorStop(0.82, `rgba(42,44,45,${a * 0.55})`);
    grd.addColorStop(1.0, 'rgba(42,44,45,0)');
    ga.fillStyle = grd;
    ga.beginPath();
    ga.arc(u * size, v * size, r, 0, Math.PI * 2);
    ga.fill();
  }

  const map = new THREE.CanvasTexture(albedo);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = THREE.RepeatWrapping;
  const bumpMap = new THREE.CanvasTexture(bump);
  bumpMap.wrapS = THREE.RepeatWrapping;
  return { map, bumpMap };
}

const moonTex = makeMoonTextures();
const moon = new THREE.Mesh(
  new THREE.SphereGeometry(MOON_RADIUS, 96, 48),
  new THREE.MeshStandardMaterial({
    map: moonTex.map,
    bumpMap: moonTex.bumpMap,
    bumpScale: 0.018,
    color: 0xd8d3c8,
    roughness: 0.95,
    metalness: 0.0,
  })
);
moon.renderOrder = 1;
scene.add(moon);

const moonOrbitPts = [];
const _orbitPos = new THREE.Vector3();
const _xAxis = new THREE.Vector3(1, 0, 0);
const _yAxis = new THREE.Vector3(0, 1, 0);
const MOON_NODE = THREE.MathUtils.degToRad(-28);

function moonPositionAtMeanAnomaly(meanAnomaly, out) {
  const M = THREE.MathUtils.euclideanModulo(meanAnomaly, Math.PI * 2);
  let E = M;
  for (let i = 0; i < 5; i++) E -= (E - MOON_ORBIT_E * Math.sin(E) - M) / (1 - MOON_ORBIT_E * Math.cos(E));
  const x = MOON_ORBIT_A * (Math.cos(E) - MOON_ORBIT_E);
  const z = MOON_ORBIT_A * Math.sqrt(1 - MOON_ORBIT_E * MOON_ORBIT_E) * Math.sin(E);
  return out.set(x, 0, z).applyAxisAngle(_xAxis, MOON_ORBIT_INCLINATION).applyAxisAngle(_yAxis, MOON_NODE);
}

for (let i = 0; i < 192; i++) {
  moonOrbitPts.push(moonPositionAtMeanAnomaly((i / 192) * Math.PI * 2, _orbitPos.clone()).clone());
}
const moonOrbitLine = new THREE.LineLoop(
  new THREE.BufferGeometry().setFromPoints(moonOrbitPts),
  new THREE.LineBasicMaterial({ color: 0x8793a8, transparent: true, opacity: 0.16 })
);
scene.add(moonOrbitLine);

function updateMoon(physicalSeconds) {
  const anomaly = MOON_INITIAL_ANOMALY + (physicalSeconds / MOON_SIDEREAL_ORBIT_SECONDS) * Math.PI * 2;
  moonPositionAtMeanAnomaly(anomaly, moon.position);
  moon.lookAt(0, 0, 0);       // keep the same hemisphere facing Earth
}
updateMoon(0);

// ---------------------------------------------------------------------------
// 2D unwrap
// ---------------------------------------------------------------------------
const unwrapScene = new THREE.Scene();
const unwrapCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const unwrapMat = new THREE.ShaderMaterial({
  uniforms: FIELD_UNIFORMS(),
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: /* glsl */`
    precision highp float;
    #define PI 3.141592653589793
    varying vec2 vUv;
    uniform samplerCube uSmoke, uTemp, uVel, uCurl, uPress;
    uniform sampler2D uBase, uNight, uGeo, uTopo;
    uniform float uUseBase, uHasNight, uHasGeo, uHasTopo, uViewMode, uOverlay, uDayNight, uWindScale, uVortScale, uPressScale;
    uniform vec3 uSunObj;
    ${CLIM}
    ${CMAP}
    ${fieldBody(false)}
    void main() {
      float phi = vUv.x * 2.0 * PI;
      float theta = (1.0 - vUv.y) * PI;
      float st = sin(theta);
      vec3 dir = vec3(-cos(phi) * st, cos(theta), sin(phi) * st);
      float sun = dot(dir, normalize(uSunObj));
      vec3 col = fieldColor(dir, vUv, sun, 0.0, vec3(0.0));
      col = col / (col + 1.0);
      col = pow(col, vec3(0.4545));
      gl_FragColor = vec4(col, 1.0);
    }
  `,
  depthTest: false, depthWrite: false,
});
unwrapScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), unwrapMat));
const unwrapFrame = document.getElementById('unwrap-frame');
const unwrapView = document.getElementById('unwrap-view');
let showUnwrap = false;

const textureDebug = {
  visible: false,
  el: null,
  cards: {},
  textures: {
    base: null,
    emissive: null,
    geo: null,
    topo: null,
  },
};

function makeTextureDebugPanel() {
  if (textureDebug.el) return textureDebug.el;

  const panel = document.createElement('div');
  panel.id = 'tex-debug';

  const addCard = (key, label) => {
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
    textureDebug.cards[key] = canvas;
  };

  addCard('base', 'uBase day albedo');
  addCard('emissive', 'uNight Black Marble');
  addCard('geo', 'uGeo climate geography');
  addCard('topo', 'uTopo elevation');

  document.body.appendChild(panel);
  textureDebug.el = panel;
  return panel;
}

function drawTexturePreview(canvas, tex) {
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = '#000';
  g.fillRect(0, 0, canvas.width, canvas.height);

  const img = tex && tex.image;
  if (!img) {
    g.fillStyle = '#657284';
    g.font = '12px ui-monospace, Menlo, Consolas, monospace';
    g.fillText('not bound', 14, 24);
    return;
  }

  try {
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = 'high';
    g.drawImage(img, 0, 0, canvas.width, canvas.height);
  } catch {
    g.fillStyle = '#657284';
    g.font = '12px ui-monospace, Menlo, Consolas, monospace';
    g.fillText('preview unavailable', 14, 24);
  }
}

function updateTextureDebugPanel() {
  if (!textureDebug.el) return;
  for (const [key, canvas] of Object.entries(textureDebug.cards)) {
    drawTexturePreview(canvas, textureDebug.textures[key]);
  }
}

function setTextureDebugVisible(show) {
  textureDebug.visible = show;
  const panel = makeTextureDebugPanel();
  panel.classList.toggle('show', textureDebug.visible);
  if (textureDebug.visible) updateTextureDebugPanel();
}

// ---------------------------------------------------------------------------
// Underlay textures
// ---------------------------------------------------------------------------
const loader = new THREE.TextureLoader();
loader.setCrossOrigin('anonymous');
const TEX_URLS = {
  earth: 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
  night: 'https://assets.science.nasa.gov/content/dam/science/esd/eo/images/imagerecords/144000/144897/BlackMarble_2016_01deg_gray.jpg',
  topo: 'https://unpkg.com/three-globe/example/img/earth-topology.png',
};
const texCache = {};
function makeCheckerTexture() {
  const c = document.createElement('canvas'); c.width = 1024; c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#0a2030'; g.fillRect(0, 0, 1024, 512);
  g.fillStyle = '#143a4f';
  const n = 24;
  for (let y = 0; y < n / 2; y++) for (let x = 0; x < n; x++)
    if ((x + y) % 2 === 0) g.fillRect(x * (1024 / n), y * (512 / (n / 2)), 1024 / n, 512 / (n / 2));
  g.strokeStyle = 'rgba(120,200,230,0.5)'; g.lineWidth = 1;
  for (let x = 0; x <= n; x++) { g.beginPath(); g.moveTo(x * (1024 / n), 0); g.lineTo(x * (1024 / n), 512); g.stroke(); }
  for (let y = 0; y <= n / 2; y++) { g.beginPath(); g.moveTo(0, y * (512 / (n / 2))); g.lineTo(1024, y * (512 / (n / 2))); g.stroke(); }
  return new THREE.CanvasTexture(c);
}

// Underlay is RENDERING ONLY (the shaded shell under the fluid). The climate
// physics uses uGeo/uTopo instead, loaded independently below.
function applyBase(day, night) {
  textureDebug.textures.base = day || null;
  textureDebug.textures.emissive = night || null;
  updateTextureDebugPanel();
  for (const m of [sphereMat, unwrapMat]) {
    m.uniforms.uBase.value = day; m.uniforms.uUseBase.value = 1;
    m.uniforms.uNight.value = night || blackTex; m.uniforms.uHasNight.value = night ? 1 : 0;
  }
}
function clearBase() {
  textureDebug.textures.base = null;
  textureDebug.textures.emissive = null;
  updateTextureDebugPanel();
  for (const m of [sphereMat, unwrapMat]) { m.uniforms.uUseBase.value = 0; m.uniforms.uHasNight.value = 0; }
}
function loadCached(key, cb) {
  if (texCache[key]) { cb(texCache[key]); return; }
  loader.load(TEX_URLS[key], (t) => { t.wrapS = THREE.RepeatWrapping; t.colorSpace = THREE.NoColorSpace; texCache[key] = t; cb(t); });
}
function setUnderlay(key) {
  if (key === 'none') { clearBase(); return; }
  if (key === 'grid') {
    if (!texCache.grid) { texCache.grid = makeCheckerTexture(); texCache.grid.wrapS = THREE.RepeatWrapping; }
    applyBase(texCache.grid, null);
    return;
  }
  // earth: day albedo + NASA Black Marble night-lights radiance map
  loadCached('earth', (day) => loadCached('night', (night) => applyBase(day, night)));
}

// Climate geography: blue-marble land/ocean (uGeo) + topography/elevation (uTopo),
// bound to every shader that evaluates climBaseline. Loaded once at startup,
// independent of the chosen underlay.
function reInitTemperature() {
  runFaces(tempInitMat, temperature.read);
  runFaces(tempInitMat, temperature.write);
}
function applyClimateTex(kind, tex) {
  textureDebug.textures[kind] = tex || null;
  updateTextureDebugPanel();
  for (const m of [heatMat, tempInitMat, sphereMat, unwrapMat]) {
    if (kind === 'geo') { m.uniforms.uGeo.value = tex; m.uniforms.uHasGeo.value = 1; }
    else { m.uniforms.uTopo.value = tex; m.uniforms.uHasTopo.value = 1; }
  }
  reInitTemperature();   // re-seed the field so it reflects the real geography
}
function setupClimate() {
  loadCached('earth', (geo) => applyClimateTex('geo', geo));
  loadCached('topo', (topo) => applyClimateTex('topo', topo));
}

// ---------------------------------------------------------------------------
// Pointer interaction — paint a wind gust + warm air + smoke
// ---------------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let painting = false, prevDir = null;
const palette = [[0.18,0.55,0.95],[0.95,0.35,0.55],[0.55,0.9,0.55],[0.95,0.7,0.25],[0.7,0.45,0.95],[0.25,0.85,0.85]];
let colorIdx = Math.floor(Math.random() * palette.length);
const _v = new THREE.Vector3();

function dirAtPointer(e) {
  ndc.x = (e.clientX / window.innerWidth) * 2 - 1;
  ndc.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObject(sphere, false);
  return hits.length ? sphere.worldToLocal(hits[0].point.clone()).normalize() : null;
}
function paintAt(dir) {
  const radiusAng = params.brush * 0.06;
  const radius = radiusAng * radiusAng + 0.0004;
  if (prevDir) {
    _v.subVectors(dir, prevDir).multiplyScalar(36.0);
    splat(velocity, dir, _v, radius, true);
  }
  const c = palette[colorIdx];
  splat(dye, dir, _v.set(c[0], c[1], c[2]), radius, false);
  splat(temperature, dir, _v.set(18.0, 0, 0), radius, false);   // warm bump (+18 K)
  prevDir = dir.clone();
}
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  const dir = dirAtPointer(e);
  if (dir) { painting = true; prevDir = null; colorIdx = (colorIdx + 1) % palette.length; paintAt(dir); }
});
renderer.domElement.addEventListener('pointermove', (e) => { if (painting) { const d = dirAtPointer(e); if (d) paintAt(d); else prevDir = null; } });
window.addEventListener('pointerup', () => { painting = false; prevDir = null; });

// ---------------------------------------------------------------------------
// Auto emitters (dye plumes, off by default)
// ---------------------------------------------------------------------------
function tangents(n) {
  const up = Math.abs(n.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const t1 = new THREE.Vector3().crossVectors(up, n).normalize();
  return [t1, new THREE.Vector3().crossVectors(n, t1)];
}
const emitters = [
  { dir: new THREE.Vector3(0.35, 0.15, 0.92).normalize(), col: [0.95, 0.45, 0.2] },
  { dir: new THREE.Vector3(-0.45, -0.1, 0.88).normalize(), col: [0.2, 0.6, 0.95] },
].map((e) => ({ ...e, t: tangents(e.dir) }));
let emitT = 0;
const _vel = new THREE.Vector3(), _col = new THREE.Vector3();
function runEmitters(physicalDays) {
  if (!params.emitters) return;
  emitT += physicalDays;
  for (let i = 0; i < emitters.length; i++) {
    const em = emitters[i];
    _vel.copy(em.t[0]).multiplyScalar(0.6 * Math.sin(emitT * 0.9 + i * 2.1)).addScaledVector(em.t[1], 0.8);
    splat(velocity, em.dir, _vel, 0.0035, true);
    const f = 2.6 + 0.6 * Math.sin(emitT * 2.0 + i);
    _col.set(em.col[0] * f, em.col[1] * f, em.col[2] * f);
    splat(dye, em.dir, _col, 0.0040, false);
  }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
function bindSlider(id, valId, key, fmt) {
  const s = document.getElementById(id), v = document.getElementById(valId);
  const update = () => { params[key] = parseFloat(s.value); v.textContent = fmt ? fmt(params[key]) : s.value; };
  s.addEventListener('input', update); update();
}
function formatTimeScale(x) {
  const minutesPerSecond = x * PHYSICAL_SECONDS_PER_REAL_SECOND / 60;
  if (minutesPerSecond < 60) return `${minutesPerSecond.toFixed(0)}m/s`;
  const hoursPerSecond = minutesPerSecond / 60;
  return `${hoursPerSecond % 1 === 0 ? hoursPerSecond.toFixed(0) : hoursPerSecond.toFixed(1)}h/s`;
}
bindSlider('s-speed', 'v-speed', 'speed', formatTimeScale);
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

const selView = document.getElementById('sel-view');
selView.addEventListener('change', () => { params.viewMode = VIEW[selView.value]; });
params.viewMode = VIEW[selView.value];

const selTex = document.getElementById('sel-tex');
selTex.addEventListener('change', () => setUnderlay(selTex.value));

const bEmit = document.getElementById('b-emit');
const bDay = document.getElementById('b-daynight');
const bUnwrap = document.getElementById('b-unwrap');
const bTexDebug = document.getElementById('b-texdebug');
bEmit.addEventListener('click', () => { params.emitters = !params.emitters; bEmit.classList.toggle('active', params.emitters); });
bDay.addEventListener('click', () => { params.dayNight = !params.dayNight; bDay.classList.toggle('active', params.dayNight); });
bUnwrap.addEventListener('click', () => { showUnwrap = !showUnwrap; bUnwrap.classList.toggle('active', showUnwrap); unwrapFrame.classList.toggle('show', showUnwrap); });
bTexDebug.addEventListener('click', () => { setTextureDebugVisible(!textureDebug.visible); bTexDebug.classList.toggle('active', textureDebug.visible); });
document.getElementById('b-reset').addEventListener('click', reset);

// ---------------------------------------------------------------------------
// Init + loop
// ---------------------------------------------------------------------------
reset();
setUnderlay(selTex.value);
setupClimate();

addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  resizePostProcessing();
});

function syncFieldUniforms(u) {
  u.uSmoke.value = dye.read.texture;
  u.uTemp.value = temperature.read.texture;
  u.uVel.value = velocity.read.texture;
  u.uCurl.value = curlRT.texture;
  u.uPress.value = pressure.read.texture;
  u.uViewMode.value = params.viewMode;
  u.uOverlay.value = params.overlay;
  u.uDayNight.value = params.dayNight ? 1 : 0;
  u.uSpecStrength.value = params.specular;
  u.uSunObj.value.copy(sunObject());
}

function advectionStepFromPhysicalSeconds(physicalSeconds) {
  return physicalSeconds * MODEL_VELOCITY_UNIT_MPS / EARTH_RADIUS_M;
}

function thermalStepFromPhysicalSeconds(physicalSeconds) {
  return physicalSeconds / EARTH_SOLAR_DAY_SECONDS;
}

const simClock = {
  realSeconds: 0,
  physicalSeconds: 0,
  deltaRealSeconds: 0,
  deltaPhysicalSeconds: 0,
  earthRotationRadians: 0,
};

function advanceGlobalClock(elapsed) {
  simClock.deltaRealSeconds = elapsed;
  simClock.deltaPhysicalSeconds = elapsed * PHYSICAL_SECONDS_PER_REAL_SECOND * params.speed;
  simClock.realSeconds += elapsed;
  simClock.physicalSeconds += simClock.deltaPhysicalSeconds;
  simClock.earthRotationRadians += (Math.PI * 2) * (simClock.deltaPhysicalSeconds / EARTH_SOLAR_DAY_SECONDS) * params.spin;
  return simClock;
}

// One simulation step + render per animation frame. The visual clock is
// accelerated, but all moving systems now derive from physical seconds:
// day/night from the solar day, Coriolis from Earth's sidereal rotation, fluid
// transport from Earth radius + wind speed, and lunar motion from the sidereal
// month. Slow frames are clamped so the solver never receives a catch-up spike.
let lastTime = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  if (now === undefined) now = performance.now();
  let elapsed = (now - lastTime) / 1000;
  lastTime = now;
  elapsed = Math.min(Math.max(elapsed, 1 / 240), 1 / 20);   // clamp 4–50 ms
  const clock = advanceGlobalClock(elapsed);
  const advectDt = advectionStepFromPhysicalSeconds(clock.deltaPhysicalSeconds);
  const thermalDt = thermalStepFromPhysicalSeconds(clock.deltaPhysicalSeconds);

  runEmitters(clock.deltaPhysicalSeconds / EARTH_SOLAR_DAY_SECONDS);
  step(advectDt, thermalDt);
  sphere.rotation.y = clock.earthRotationRadians;
  updateMoon(clock.physicalSeconds);
  syncFieldUniforms(sphereMat.uniforms);
  syncFieldUniforms(unwrapMat.uniforms);
  atmosphereMat.uniforms.uStrength.value = params.atmosphere;
  controls.update();
  starSphere.position.copy(camera.position);
  starSphere.material.uniforms.uTime.value = now * 0.001;
  updateLensOptics(now);

  renderer.setRenderTarget(null);
  renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
  renderer.setScissorTest(false);
  composer.render();

  if (showUnwrap) {
    const r = unwrapView.getBoundingClientRect();
    if (r.width > 1 && r.height > 1) {
      const x = r.left, y = window.innerHeight - r.bottom;
      renderer.setViewport(x, y, r.width, r.height);
      renderer.setScissor(x, y, r.width, r.height);
      renderer.setScissorTest(true);
      renderer.render(unwrapScene, unwrapCamera);
      renderer.setScissorTest(false);
    }
  }
}
frame();
