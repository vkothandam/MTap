// Intraday → Space: the 3-D pocket space, a Streamlit custom component (st.components.v2).
//
// Python (space_page.py) sends the pocket maps as `data`; this draws each as a landscape:
// x = time ahead (log), depth = price vs now (σ, higher farther back), height = the share of the
// later volume in the cell. Cells are coloured by pocket (50 / 80 / 95% / outside), blue above
// the price now and red below. The views share one camera: rotating one turns them all.
// Hover shows a cell; a click sends it back with setStateValue("cell", {i, j}).
//
// Streamlit calls the default export again whenever `data` changes. The renderer, camera and
// controls are kept per mount (keyed by its parentElement), so the camera stays where it was.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const W = 2.0, D = 1.8, H = 0.8;          // scene size: time ahead, price, height
const ROW_H = 320;                         // px per view
const COLS = 2;
const REGIONS = ["inside 50%", "inside 80%", "inside 95%", "outside"];
// OKLCH-matched steps (50% pocket strongest), checked for colour-vision deficiency
// against the light and the dark surface: blue above now, red below, grey outside.
const PALETTE = {
  light: { up: ["#015bb5", "#3280de", "#62a6fe"], down: ["#b0081d", "#db4141", "#fe726c"],
           out: "#d6d5d1" },
  dark: { up: ["#92c1fe", "#509cfd", "#2c79d8"], down: ["#fea199", "#fb605c", "#d43a3b"],
          out: "#3a3a37" },
};
const START = new THREE.Vector3(-2.5, 1.9, 2.0);   // from the near-time side: both sides of now show
const TARGET = new THREE.Vector3(0, 0.15, 0);

const mounts = new WeakMap();

const CSS = `
.s3 { font: 13px var(--st-font, sans-serif); color: var(--st-text-color); }
.s3 .bar { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin: 2px 0 6px; }
.s3 button { font: inherit; color: inherit; background: var(--st-secondary-background-color);
  border: 1px solid var(--st-border-color); border-radius: 6px; padding: 2px 9px; cursor: pointer; }
.s3 button.on { border-color: var(--st-primary-color); color: var(--st-primary-color); }
.s3 .seg button:first-child { border-radius: 6px 0 0 6px; }
.s3 .seg button:last-child { border-radius: 0 6px 6px 0; margin-left: -1px; }
.s3 .legend { display: flex; gap: 12px; align-items: center; opacity: .85; }
.s3 .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 1px;
  vertical-align: -2px; }
.s3 .stage { position: relative; }
.s3 canvas { display: block; touch-action: none; }
.s3 .cells { position: absolute; inset: 0; display: grid; pointer-events: none;
  grid-template-columns: repeat(${COLS}, 1fr); gap: 0; }
.s3 .cell { border: 1px solid var(--st-border-color); border-radius: 8px; margin: 2px;
  padding: 6px 10px; font-weight: 600; }
.s3 .tip { position: absolute; pointer-events: none; background: var(--st-background-color);
  border: 1px solid var(--st-border-color); border-radius: 6px; padding: 6px 9px; line-height: 1.45;
  white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,.18); display: none; z-index: 2; }
.s3 .tip b { font-weight: 600; }
.s3 .hint { opacity: .7; margin-top: 4px; }
`;

// ---- small helpers ---------------------------------------------------------------------------

const span = (m) => (m < 10 ? `${m.toFixed(1)}m` : m < 90 ? `${Math.round(m)}m` : `${(m / 60).toFixed(1)}h`);
const usd = (x) => `$${x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const sigma = (s) => `${s >= 0 ? "+" : "−"}${Math.abs(s).toFixed(2)}σ`;

function theme(el) {
  const cs = getComputedStyle(el);
  const v = (k, d) => cs.getPropertyValue(k).trim() || d;
  const bg = new THREE.Color(v("--st-background-color", "#ffffff"));
  return { dark: bg.getHSL({}).l < 0.5, text: v("--st-text-color", "#31333f"),
           font: v("--st-font", "sans-serif") };
}

/** Per cell corner, the mean of the cells around it: a surface whose quads are the cells. */
function corners(Q, nx, ny) {
  const F = new Float32Array((nx + 1) * (ny + 1));
  for (let a = 0; a <= nx; a++) {
    for (let b = 0; b <= ny; b++) {
      let s = 0, n = 0;
      for (const i of [a - 1, a]) {
        for (const j of [b - 1, b]) {
          if (i >= 0 && i < nx && j >= 0 && j < ny) { s += Q[i * ny + j]; n++; }
        }
      }
      F[a * (ny + 1) + b] = s / n;
    }
  }
  return F;
}

/** Marching squares on the corner field: segments (a0, b0, a1, b1, ...) in cell-edge units. */
function contour(F, nx, ny, t) {
  const v = (a, b) => F[a * (ny + 1) + b];
  const out = [];
  for (let a = 0; a < nx; a++) {
    for (let b = 0; b < ny; b++) {
      const P = [[a, b], [a + 1, b], [a + 1, b + 1], [a, b + 1]];
      const c = P.map(([x, y]) => v(x, y));
      const pts = [];
      for (let k = 0; k < 4; k++) {
        const u = c[k], w = c[(k + 1) % 4];
        if ((u >= t) !== (w >= t)) {
          const f = (t - u) / (w - u), [a0, b0] = P[k], [a1, b1] = P[(k + 1) % 4];
          pts.push(a0 + f * (a1 - a0), b0 + f * (b1 - b0));
        }
      }
      if (pts.length >= 4) out.push(...pts.slice(0, 4));
      if (pts.length === 8) out.push(...pts.slice(4, 8));
    }
  }
  return out;
}

function label(text, color, font, { px = 30, k = 0.0015, anchor = [0.5, 0.5] } = {}) {
  const c = document.createElement("canvas");
  const g = c.getContext("2d");
  g.font = `${px}px ${font}`;
  c.width = Math.ceil(g.measureText(text).width) + 8;
  c.height = px + 10;
  g.font = `${px}px ${font}`;
  g.fillStyle = color;
  g.textBaseline = "middle";
  g.fillText(text, 4, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearFilter;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, opacity: 0.8,
                                                        transparent: true }));
  s.scale.set(c.width * k, c.height * k, 1);
  s.center.set(...anchor);
  return s;
}

function lines(pts, color, opacity, y) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts.map((p, n) => (n % 3 === 1 ? p + y : p)), 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, opacity, transparent: true }));
}

function dispose(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) {
      o.material.map?.dispose();
      o.material.dispose();
    }
  });
}

// ---- one view: a scene with the surface, rings, floor, now plane and labels -----------------

function buildView(v, d, m) {
  const { nx, ny } = d;
  const pal = m.theme.dark ? PALETTE.dark : PALETTE.light;
  const X = (a) => -W / 2 + (a / nx) * W;
  const Z = (b) => D / 2 - (b / ny) * D;
  const peak = Math.max(...v.q) || 1;
  const hgt = (q) => H * (m.sqrt ? Math.sqrt(Math.max(q, 0) / peak) : q / peak);
  const F = corners(v.q, nx, ny);
  const Y = (a, b) => hgt(F[a * (ny + 1) + b]);

  // the surface: one quad per cell (non-indexed so each cell has one colour), smooth normals
  const pos = [], col = [], nor = [];
  const normal = (a, b) => {
    const dx = (Y(Math.min(a + 1, nx), b) - Y(Math.max(a - 1, 0), b)) / (2 * W / nx);
    const dz = (Y(a, Math.min(b + 1, ny)) - Y(a, Math.max(b - 1, 0))) / (2 * D / ny);
    return new THREE.Vector3(-dx, 1, dz).normalize();
  };
  const c = new THREE.Color();
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const r = v.region[i * ny + j];
      c.set(r >= 3 ? pal.out : (j >= ny / 2 ? pal.up : pal.down)[r]);
      for (const [a, b] of [[i, j], [i + 1, j], [i + 1, j + 1], [i, j], [i + 1, j + 1], [i, j + 1]]) {
        pos.push(X(a), Y(a, b), Z(b));
        const n = normal(a, b);
        nor.push(n.x, n.y, n.z);
        col.push(c.r, c.g, c.b);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const surface = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true,
                                                                    side: THREE.DoubleSide }));
  const scene = new THREE.Scene();
  scene.add(surface);
  scene.add(new THREE.AmbientLight(0xffffff, m.theme.dark ? 1.1 : 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(0.8, 2.0, 1.2);
  scene.add(sun);

  const ink = m.theme.text;
  // the rings at the 50 / 80 / 95% levels on the surface, and their shadows on the floor
  if (m.rings) {
    v.levels.forEach((t, k) => {
      const seg = contour(F, nx, ny, t);
      const pts = [];
      for (let n = 0; n < seg.length; n += 2) pts.push(X(seg[n]), hgt(t), Z(seg[n + 1]));
      scene.add(lines(pts, ink, [0.95, 0.7, 0.45][k], 0.004));
      scene.add(lines(pts.map((p, n) => (n % 3 === 1 ? 0 : p)), ink, 0.25, 0.002));
    });
  }
  // the floor frame, tick lines and the price-now line; a translucent "now" wall
  const fr = [];
  const edge = (x0, z0, x1, z1) => fr.push(x0, 0, z0, x1, 0, z1);
  edge(X(0), Z(0), X(nx), Z(0)); edge(X(nx), Z(0), X(nx), Z(ny));
  edge(X(nx), Z(ny), X(0), Z(ny)); edge(X(0), Z(ny), X(0), Z(0));
  scene.add(lines(fr, ink, 0.5, 0));
  const tk = [];
  for (const [a] of d.xt) tk.push(X(a), 0, Z(0), X(a), 0, Z(ny));
  for (const [b] of d.yt) tk.push(X(0), 0, Z(b), X(nx), 0, Z(b));
  scene.add(lines(tk, ink, 0.12, 0));
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(W, H),
                              new THREE.MeshBasicMaterial({ color: ink, opacity: 0.07, transparent: true,
                                                            depthWrite: false, side: THREE.DoubleSide }));
  wall.position.set(0, H / 2, Z(ny / 2));
  scene.add(wall);
  scene.add(lines([X(0), 0, Z(ny / 2), X(nx), 0, Z(ny / 2)], ink, 0.8, 0.003));

  // axis labels
  for (const [a, t] of d.xt) {
    const s = label(t, ink, m.theme.font);
    s.position.set(X(a), 0, Z(0) + 0.09);
    scene.add(s);
  }
  for (const [b, t] of d.yt) {
    const s = label(t, ink, m.theme.font, { anchor: [1, 0.5] });
    s.position.set(X(0) - 0.05, 0, Z(b));
    scene.add(s);
  }
  const tx = label("time ahead →", ink, m.theme.font, { px: 32 });
  tx.position.set(0, 0, Z(0) + 0.3);
  scene.add(tx);
  const ty = label("price vs now ↑", ink, m.theme.font, { px: 32, anchor: [1, 0.5] });
  ty.position.set(X(0) - 0.05, 0, Z(ny) - 0.12);
  scene.add(ty);

  // the picked cell: a frame on the surface and a post down to the floor
  if (d.cell) {
    const [i, j] = d.cell;
    const p = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b]) => [X(a), Y(a, b), Z(b)]);
    const box = [];
    for (let k = 0; k < 4; k++) box.push(...p[k], ...p[(k + 1) % 4]);
    const cx = X(i + 0.5), cz = Z(j + 0.5), top = Math.max(...p.map((q) => q[1]));
    box.push(cx, 0, cz, cx, top, cz);
    scene.add(lines(box, ink, 1, 0.008));
  }
  return { scene, surface, title: v.title, name: v.title.split(" · ")[0], q: v.q,
           region: v.region };
}

// ---- the mount: renderer, camera, controls, overlay; updated in place on new data ----------

function create(parent, setStateValue) {
  const style = document.createElement("style");
  style.textContent = CSS;
  const root = document.createElement("div");
  root.className = "s3";
  root.innerHTML = `
    <div class="bar">
      <span class="seg"><button data-h="lin">linear height</button><button data-h="sqrt">√ height</button></span>
      <button data-rings>rings</button>
      <button data-reset>reset view</button>
      <span class="legend"></span>
    </div>
    <div class="stage"><div class="cells"></div><div class="tip"></div></div>
    <div class="hint">Drag to turn (all views turn together), scroll to zoom, right-drag to pan.
      Hover a cell for its numbers; click it to pick it (Compare tab → Pick a cell).</div>`;
  parent.appendChild(style);
  parent.appendChild(root);
  const stage = root.querySelector(".stage");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.autoClear = false;
  stage.prepend(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(34, 1, 0.05, 50);
  camera.position.copy(START);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(TARGET);
  controls.maxPolarAngle = Math.PI / 2 - 0.03;
  controls.minDistance = 1.0;
  controls.maxDistance = 9;
  controls.update();

  const m = { parent, root, stage, style, renderer, camera, controls, views: [], data: null,
              sqrt: true, rings: true, w: 0, rows: 1, setStateValue };
  controls.addEventListener("change", () => draw(m));
  root.querySelectorAll("[data-h]").forEach((b) => b.addEventListener("click", () => {
    m.sqrt = b.dataset.h === "sqrt";
    rebuild(m);
  }));
  root.querySelector("[data-rings]").addEventListener("click", () => {
    m.rings = !m.rings;
    rebuild(m);
  });
  root.querySelector("[data-reset]").addEventListener("click", () => {
    camera.position.copy(START);
    controls.target.copy(TARGET);
    controls.update();
  });
  new ResizeObserver(() => resize(m)).observe(root);
  pointer(m);
  return m;
}

function resize(m) {
  const w = Math.floor(m.root.clientWidth);
  if (!w) return;
  m.w = w;
  m.rows = Math.max(1, Math.ceil(m.views.length / COLS));
  m.renderer.setSize(w, m.rows * ROW_H);
  m.stage.style.height = `${m.rows * ROW_H}px`;
  draw(m);
}

function rebuild(m) {
  m.views.forEach((v) => dispose(v.scene));
  m.theme = theme(m.root);
  m.views = m.data.views.map((v) => buildView(v, m.data, m));
  const pal = m.theme.dark ? PALETTE.dark : PALETTE.light;
  const sw = (cs) => cs.map((c) => `<span class="sw" style="background:${c}"></span>`).join("");
  m.root.querySelector(".legend").innerHTML =
    `<span>${sw(pal.up)} above now</span><span>${sw(pal.down)} below now</span>` +
    `<span>${sw([pal.out])} outside 95%</span><span>dark → light: 50 / 80 / 95% pocket</span>`;
  m.root.querySelectorAll("[data-h]").forEach((b) => b.classList.toggle("on", (b.dataset.h === "sqrt") === m.sqrt));
  m.root.querySelector("[data-rings]").classList.toggle("on", m.rings);
  m.root.querySelector(".cells").innerHTML = m.views.map((v) => `<div class="cell">${v.title}</div>`).join("");
  m.root.querySelector(".cells").style.gridTemplateRows = `repeat(${Math.ceil(m.views.length / COLS)}, ${ROW_H}px)`;
  resize(m);
}

function draw(m) {
  const { renderer, camera } = m;
  if (!m.w) return;
  const vw = Math.floor(m.w / COLS);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.setScissorTest(true);
  camera.aspect = vw / ROW_H;
  camera.updateProjectionMatrix();
  m.views.forEach((v, k) => {
    const x = (k % COLS) * vw, y = (m.rows - 1 - Math.floor(k / COLS)) * ROW_H;
    renderer.setViewport(x, y, vw, ROW_H);
    renderer.setScissor(x, y, vw, ROW_H);
    renderer.render(v.scene, camera);
  });
}

/** The cell under the pointer: {view, i, j} or null. */
function pick(m, e) {
  const rect = m.renderer.domElement.getBoundingClientRect();
  const px = e.clientX - rect.left, py = e.clientY - rect.top;
  const vw = rect.width / COLS;
  const col = Math.floor(px / vw), row = Math.floor(py / ROW_H);
  const k = row * COLS + col;
  const v = m.views[k];
  if (!v || col >= COLS) return null;
  const ndc = new THREE.Vector2(((px - col * vw) / vw) * 2 - 1, -((py - row * ROW_H) / ROW_H) * 2 + 1);
  m.ray = m.ray || new THREE.Raycaster();
  m.ray.setFromCamera(ndc, m.camera);
  const hit = m.ray.intersectObject(v.surface)[0];
  if (!hit) return null;
  const cell = Math.floor(hit.faceIndex / 2);
  return { v, i: Math.floor(cell / m.data.ny), j: cell % m.data.ny, px, py };
}

function pointer(m) {
  const el = m.renderer.domElement;
  const tip = m.root.querySelector(".tip");
  let down = null;
  el.addEventListener("pointermove", (e) => {
    const h = m.data && !e.buttons ? pick(m, e) : null;
    if (!h) { tip.style.display = "none"; return; }
    const { xe, ye, now, sig, ny } = m.data;
    const q = h.v.q[h.i * ny + h.j], r = h.v.region[h.i * ny + h.j];
    const lo = now * Math.exp(ye[h.j] * sig), hi = now * Math.exp(ye[h.j + 1] * sig);
    tip.innerHTML = `<b>${h.v.name}</b><br>${span(xe[h.i])}–${span(xe[h.i + 1])} ahead<br>` +
      `${sigma(ye[h.j])} to ${sigma(ye[h.j + 1])} (${usd(lo)}–${usd(hi)})<br>` +
      `${(q * 100).toFixed(2)}% of the later volume<br>${REGIONS[r]}, ${h.j >= ny / 2 ? "above" : "below"} now`;
    tip.style.display = "block";
    const right = h.px > m.w - 260;
    tip.style.left = `${right ? h.px - tip.offsetWidth - 14 : h.px + 14}px`;
    tip.style.top = `${Math.max(0, h.py - 10)}px`;
  });
  el.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  el.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    down = null;
    const h = pick(m, e);
    if (h) m.setStateValue("cell", { i: h.i, j: h.j, at: Date.now() });  // at: re-picking a cell still counts
  });
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let m = mounts.get(parentElement);
  if (!m || !m.root.isConnected) {
    m = create(parentElement, setStateValue);
    mounts.set(parentElement, m);
  }
  m.setStateValue = setStateValue;
  m.data = data;
  rebuild(m);
  return () => {
    // a rerun with new data calls this too: only free the GPU once the element is gone
    setTimeout(() => {
      if (m.root.isConnected) return;
      m.views.forEach((v) => dispose(v.scene));
      m.controls.dispose();
      m.renderer.dispose();
      m.renderer.forceContextLoss();
      mounts.delete(parentElement);
    }, 2000);
  };
}
