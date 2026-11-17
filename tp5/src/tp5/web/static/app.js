// Latent space explorer. Dataset-agnostic: samples are flattened images (height, width, channels),
// channels-last, values in [0, 1]; every decode goes through the Python model (/api/decode).

import { $, api, binarized, css, drawImage, drawImageAt, escapeHtml, fmt, imageError, latestOnly } from "./common.js";

// ---------------------------------------------------------------- state

const state = {
  data: null,
  z: [],
  axisX: 0,
  axisY: 1,
  selected: null,
  grid: false,
  gridSize: 12,
  gridImages: null,
  gridKey: null,
  dragging: false,
  hover: null,
  sort: { column: 0, ascending: true },
  binarize: false,
  lastOutput: null,
  sigma: false,
  interpolation: { from: 0, to: 1, t: 0.5 },
};

// Error between a sample and its reconstruction: wrong pixels (binary) or mean absolute error
function reconstructionError(i) {
  const { images, reconstructions, binary } = state.data;
  return imageError(images[i], reconstructions[i], binary);
}

// ---------------------------------------------------------------- latent map

const map = $("map");
const margin = 34;

function mapSize() {
  const dpr = window.devicePixelRatio || 1;
  const size = map.clientWidth;
  if (map.width !== Math.round(size * dpr)) {
    map.width = map.height = Math.round(size * dpr);
  }
  return { size, dpr };
}

// Latent value ↔ CSS pixel on the map, for the selected axes
function toPixel(value, axis, size, vertical) {
  const [low, high] = state.data.bounds[axis];
  const t = (value - low) / (high - low);
  return vertical ? size - margin - t * (size - 2 * margin) : margin + t * (size - 2 * margin);
}
function toValue(pixel, axis, size, vertical) {
  const [low, high] = state.data.bounds[axis];
  const t = vertical ? (size - margin - pixel) / (size - 2 * margin) : (pixel - margin) / (size - 2 * margin);
  return low + Math.min(1, Math.max(0, t)) * (high - low);
}

function oneDimensional() {
  return state.data.latent_dim === 1;
}

function drawMap() {
  const { size, dpr } = mapSize();
  const ctx = map.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = css("--panel");
  ctx.fillRect(0, 0, size, size);

  const { latent, labels, image_shape, ink } = state.data;
  const ax = state.axisX, ay = state.axisY;
  const yOf = (code) => (oneDimensional() ? size / 2 : toPixel(code[ay], ay, size, true));
  const inner = size - 2 * margin;

  // Decoded grid behind the points
  if (state.grid && state.gridImages) {
    const n = state.gridSize;
    const cell = inner / n;
    const rows = oneDimensional() ? 1 : n;
    const top = oneDimensional() ? size / 2 - cell / 2 : margin;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < n; c++) {
        const image = state.gridImages[r * n + c];
        // Fit the image in its cell keeping the aspect ratio
        const scale = (cell * 0.92) / Math.max(image_shape.width, image_shape.height);
        const w = image_shape.width * scale, h = image_shape.height * scale;
        drawImageAt(ctx, shown(image), image_shape, ink, margin + c * cell + (cell - w) / 2, top + r * cell + (cell - h) / 2, w, h);
      }
    }
  } else {
    // Light grid lines
    ctx.strokeStyle = css("--grid-line");
    ctx.lineWidth = 1;
    for (let k = 0; k <= 10; k++) {
      const p = margin + (k / 10) * inner;
      ctx.beginPath(); ctx.moveTo(p, margin); ctx.lineTo(p, size - margin); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(margin, p); ctx.lineTo(size - margin, p); ctx.stroke();
    }
  }

  // Frame and axis labels
  ctx.strokeStyle = css("--border");
  ctx.strokeRect(margin, margin, inner, inner);
  ctx.fillStyle = css("--muted");
  ctx.font = "11px system-ui, sans-serif";
  const [xl, xh] = state.data.bounds[ax];
  ctx.textAlign = "left"; ctx.fillText(fmt(xl), margin, size - margin + 14);
  ctx.textAlign = "right"; ctx.fillText(fmt(xh), size - margin, size - margin + 14);
  ctx.textAlign = "center"; ctx.fillText(`z${ax}`, size / 2, size - 8);
  if (!oneDimensional()) {
    const [yl, yh] = state.data.bounds[ay];
    ctx.save();
    ctx.translate(12, size / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(`z${ay}`, 0, 0);
    ctx.restore();
    ctx.textAlign = "left";
    ctx.fillText(fmt(yh), 2, margin + 4);
    ctx.fillText(fmt(yl), 2, size - margin);
  }

  // Training samples
  const showLabels = latent.length <= 80;
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "left";
  latent.forEach((code, i) => {
    const x = toPixel(code[ax], ax, size, false), y = yOf(code);
    const selected = i === state.selected;
    ctx.beginPath();
    ctx.arc(x, y, selected ? 6 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = selected ? css("--selected") : css("--point");
    ctx.fill();
    if (state.grid) { ctx.strokeStyle = css("--panel"); ctx.lineWidth = 1.5; ctx.stroke(); }
    if (showLabels || selected) {
      // Halo so labels stay readable over the decoded grid
      ctx.strokeStyle = css("--panel");
      ctx.lineWidth = 3;
      ctx.strokeText(labels[i], x + 6, y - 6);
      ctx.fillStyle = selected ? css("--selected") : css("--text");
      ctx.fillText(labels[i], x + 6, y - 6);
    }
  });

  // σ(x) of each code (variational): ellipse of one standard deviation along the axes shown
  if (state.sigma && state.data.latent_sigma) {
    ctx.strokeStyle = css("--point");
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1;
    latent.forEach((code, i) => {
      const sigma = state.data.latent_sigma[i];
      const x = toPixel(code[ax], ax, size, false), y = yOf(code);
      const rx = Math.abs(toPixel(code[ax] + sigma[ax], ax, size, false) - x);
      const ry = oneDimensional() ? 4 : Math.abs(toPixel(code[ay] + sigma[ay], ay, size, true) - y);
      ctx.beginPath();
      ctx.ellipse(x, y, Math.max(rx, 1), Math.max(ry, 1), 0, 0, 2 * Math.PI);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  // Current z (crosshair)
  const cx = toPixel(state.z[ax], ax, size, false), cy = yOf(state.z);
  ctx.strokeStyle = css("--selected");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy);
  ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, 7, 0, 2 * Math.PI); ctx.stroke();
}

// Sample under the pointer (within 8 px), or null
function hit(event) {
  const rect = map.getBoundingClientRect();
  const size = rect.width;
  const px = event.clientX - rect.left, py = event.clientY - rect.top;
  let best = null, bestDistance = 8;
  state.data.latent.forEach((code, i) => {
    const x = toPixel(code[state.axisX], state.axisX, size, false);
    const y = oneDimensional() ? size / 2 : toPixel(code[state.axisY], state.axisY, size, true);
    const d = Math.hypot(x - px, y - py);
    if (d < bestDistance) { best = i; bestDistance = d; }
  });
  return { index: best, px, py, size };
}

function moveTo(event) {
  const { px, py, size } = hit(event);
  state.z[state.axisX] = toValue(px, state.axisX, size, false);
  if (!oneDimensional()) state.z[state.axisY] = toValue(py, state.axisY, size, true);
  update();
}

map.addEventListener("pointerdown", (event) => {
  const { index } = hit(event);
  if (index !== null) {
    selectSample(index);
    return;
  }
  state.dragging = true;
  map.setPointerCapture(event.pointerId);
  moveTo(event);
});

map.addEventListener("pointermove", (event) => {
  if (state.dragging) { moveTo(event); return; }
  const { index, px, py } = hit(event);
  showTooltip(index, px, py);
});

map.addEventListener("pointerup", () => { state.dragging = false; });
map.addEventListener("pointerleave", () => showTooltip(null));

function showTooltip(index, px, py) {
  const tooltip = $("tooltip");
  if (index === null) { tooltip.hidden = true; return; }
  const { labels, latent, images, image_shape, ink } = state.data;
  tooltip.innerHTML = "";
  const canvas = document.createElement("canvas");
  canvas.className = "image";
  canvas.style.width = "36px";
  drawImage(canvas, images[index], image_shape, ink);
  const text = document.createElement("div");
  text.innerHTML = `<strong>${escapeHtml(labels[index])}</strong><br><span class="muted">(${latent[index].map(fmt).join(", ")})</span>`;
  tooltip.append(canvas, text);
  tooltip.hidden = false;
  tooltip.style.left = `${Math.min(px + 14, map.clientWidth - 200)}px`;
  tooltip.style.top = `${py + 14}px`;
}

// ---------------------------------------------------------------- decoding

// Binary datasets can show the decoded image as the model outputs it or thresholded like the metric
function shown(pixels) {
  return state.binarize ? binarized(pixels) : pixels;
}

const decodeCurrent = latestOnly(async (z) => {
  const { images, nearest, distance } = await api("/api/decode", { z: [z] });
  state.lastOutput = images[0];
  const { image_shape, ink, labels, binary, tolerance } = state.data;
  drawImage($("output"), shown(images[0]), image_shape, ink);
  drawImage($("nearest"), state.data.images[nearest[0]], image_shape, ink);

  const d = distance[0];
  $("nearest-caption").textContent = `nearest: ${labels[nearest[0]]} (${binary ? `${d} px` : `MAE ${fmt(d)}`})`;
  $("novelty").innerHTML = binary && d > tolerance
    ? `<span class="badge">new</span> differs from every training sample by more than ${tolerance} px`
    : "";
});

// Grid over the two map axes, the other coordinates fixed at the current z
function gridPoints() {
  const n = state.gridSize;
  const [xl, xh] = state.data.bounds[state.axisX];
  const [yl, yh] = state.data.bounds[state.axisY];
  const rows = oneDimensional() ? 1 : n;
  const points = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < n; c++) {
      const z = state.z.slice();
      z[state.axisX] = xl + ((c + 0.5) / n) * (xh - xl);
      if (!oneDimensional()) z[state.axisY] = yh - ((r + 0.5) / n) * (yh - yl);
      points.push(z);
    }
  }
  return points;
}

const decodeGrid = latestOnly(async (key, points) => {
  const { images } = await api("/api/decode", { z: points });
  state.gridImages = images;
  state.gridKey = key;
  drawMap();
});

function refreshGrid() {
  if (!state.grid) return;
  // The grid only depends on the axes shown, its size and the coordinates NOT shown
  const fixed = state.z.map((v, k) => (k === state.axisX || k === state.axisY ? null : v));
  const key = JSON.stringify([state.axisX, state.axisY, state.gridSize, fixed]);
  if (key !== state.gridKey) decodeGrid(key, gridPoints());
}

// ---------------------------------------------------------------- sliders

function buildSliders() {
  const container = $("sliders");
  container.innerHTML = "";
  state.data.bounds.forEach(([low, high], k) => {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.dataset.k = k;
    const step = (high - low) / 1000;
    row.innerHTML = `
      <label for="z-${k}">z${k}</label>
      <input type="range" id="z-${k}" min="${low}" max="${high}" step="${step}">
      <input type="number" step="${step}" aria-label="z${k} value">`;
    const [range, number] = row.querySelectorAll("input");
    range.addEventListener("input", () => { state.z[k] = Number(range.value); update(range); });
    number.addEventListener("change", () => {
      const value = Number(number.value);
      if (Number.isFinite(value)) { state.z[k] = value; update(number); }
    });
    container.append(row);
  });
}

function syncSliders(source) {
  // Only the latent sliders (the interpolation `t` row shares the class but has no data-k)
  $("sliders").querySelectorAll(".slider-row").forEach((row) => {
    const k = Number(row.dataset.k);
    const [range, number] = row.querySelectorAll("input");
    if (range !== source) range.value = state.z[k];
    if (number !== source) number.value = Number(state.z[k].toFixed(4));
    row.classList.toggle("on-map", k === state.axisX || (k === state.axisY && !oneDimensional()));
  });
  $("z-text").textContent = `[${state.z.map(fmt).join(", ")}]`;
}

// ---------------------------------------------------------------- table

function buildTable() {
  const { latent_dim, binary } = state.data;
  const head = ["#", "image", "label", binary ? "recon. px" : "recon. MAE"];
  for (let k = 0; k < latent_dim; k++) head.push(`z${k}`);
  const tr = document.createElement("tr");
  head.forEach((name, column) => {
    const th = document.createElement("th");
    th.textContent = name;
    if (column !== 1) th.addEventListener("click", () => sortBy(column));
    tr.append(th);
  });
  $("codes").tHead.replaceChildren(tr);
  renderTable();
}

function sortBy(column) {
  state.sort = { column, ascending: state.sort.column === column ? !state.sort.ascending : true };
  renderTable();
}

function renderTable() {
  const { labels, latent, images, image_shape, ink, binary, tolerance } = state.data;
  const errors = labels.map((_, i) => reconstructionError(i));
  const key = (i) => {
    const c = state.sort.column;
    if (c === 0) return i;
    if (c === 2) return labels[i];
    if (c === 3) return errors[i];
    return latent[i][c - 4];
  };
  const order = labels.map((_, i) => i).sort((a, b) => {
    const ka = key(a), kb = key(b);
    const cmp = typeof ka === "string" ? ka.localeCompare(kb) : ka - kb;
    return state.sort.ascending ? cmp : -cmp;
  });

  const body = $("codes").tBodies[0];
  body.replaceChildren(...order.map((i) => {
    const tr = document.createElement("tr");
    tr.dataset.index = i;
    tr.classList.toggle("selected", i === state.selected);
    const canvas = document.createElement("canvas");
    canvas.className = "image";
    drawImage(canvas, images[i], image_shape, ink);
    const bad = binary && errors[i] > tolerance;
    const cells = [String(i), canvas, labels[i], binary ? String(errors[i]) : fmt(errors[i]), ...latent[i].map(fmt)];
    cells.forEach((content, column) => {
      const td = document.createElement("td");
      if (content instanceof HTMLElement) td.append(content); else td.textContent = content;
      if (column === 3 && bad) td.className = "bad";
      tr.append(td);
    });
    tr.addEventListener("click", () => selectSample(i));
    return tr;
  }));

  $("codes").tHead.querySelectorAll("th").forEach((th, column) => {
    th.classList.toggle("sorted", column === state.sort.column);
    th.classList.toggle("asc", column === state.sort.column && state.sort.ascending);
  });
}

function markSelectedRow() {
  $("codes").tBodies[0].querySelectorAll("tr").forEach((tr) => {
    const selected = Number(tr.dataset.index) === state.selected;
    tr.classList.toggle("selected", selected);
    // Scrolls only the table box (scrollIntoView would also scroll the page away from the map)
    if (selected) {
      const box = tr.closest(".table-wrap");
      const header = $("codes").tHead.offsetHeight;
      if (tr.offsetTop - header < box.scrollTop) box.scrollTop = tr.offsetTop - header;
      else if (tr.offsetTop + tr.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = tr.offsetTop + tr.offsetHeight - box.clientHeight;
    }
  });
}

// ---------------------------------------------------------------- actions

function selectSample(i) {
  state.selected = i;
  state.z = state.data.latent[i].slice();
  $("reset").disabled = false;
  markSelectedRow();
  update();
}

function update(source) {
  syncSliders(source);
  drawMap();
  decodeCurrent(state.z.slice());
  refreshGrid();
}

function buildAxes() {
  const n = state.data.latent_dim;
  for (const [select, axis] of [[$("axis-x"), "axisX"], [$("axis-y"), "axisY"]]) {
    select.innerHTML = "";
    for (let k = 0; k < n; k++) select.add(new Option(`z${k}`, k));
    select.value = state[axis];
    select.disabled = n === 1 && axis === "axisY";
    select.addEventListener("change", () => {
      state[axis] = Number(select.value);
      update();
    });
  }
}

$("grid-toggle").addEventListener("change", (event) => {
  state.grid = event.target.checked;
  state.gridKey = null;
  if (!state.grid) state.gridImages = null;
  update();
});

$("grid-size").addEventListener("change", (event) => {
  state.gridSize = Math.min(40, Math.max(3, Number(event.target.value) || 12));
  event.target.value = state.gridSize;
  state.gridImages = null;
  update();
});

$("binarize").addEventListener("change", (event) => {
  state.binarize = event.target.checked;
  if (state.lastOutput) drawImage($("output"), shown(state.lastOutput), state.data.image_shape, state.data.ink);
  drawMap();
  refreshStrip();
});

// Saves the decoder output for the current z on the server (generated_dir): normal + binarized + JSON
// The file name is always typed by the user; existing files are never overwritten
async function save() {
  const name = $("save-name").value.trim();
  if (!name) {
    $("saved").textContent = "Type a file name first.";
    $("save-name").focus();
    return;
  }
  const button = $("save");
  button.disabled = true;
  $("saved").textContent = "Saving…";
  try {
    const { files } = await api("/api/generate", { z: state.z, name });
    $("save-name").value = "";
    $("saved").textContent = `Saved: ${files.join(", ")}`;
  } catch (error) {
    $("saved").textContent = `Could not save: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

$("save").addEventListener("click", save);
$("save-name").addEventListener("keydown", (event) => {
  if (event.key === "Enter") save();
});

// ---------------------------------------------------------------- variational

// New sample of the variational autoencoder: z ~ N(0, I) drawn by the server
$("prior").addEventListener("click", async () => {
  try {
    const { z } = await api("/api/sample", { count: 1 });
    state.z = z[0];
    update();
  } catch (error) {
    console.error(error);
  }
});

$("sigma-toggle").addEventListener("change", (event) => {
  state.sigma = event.target.checked;
  drawMap();
});

const INTERPOLATION_STEPS = 9;

function interpolationPoint(t) {
  const a = state.data.latent[state.interpolation.from], b = state.data.latent[state.interpolation.to];
  return a.map((v, k) => (1 - t) * v + t * b[k]);
}

const decodeStrip = latestOnly(async (points) => {
  const { images } = await api("/api/decode", { z: points });
  const { image_shape, ink } = state.data;
  $("interp-strip").replaceChildren(...images.map((image, i) => {
    const canvas = document.createElement("canvas");
    canvas.className = "image";
    const t = i / (INTERPOLATION_STEPS - 1);
    canvas.title = `t = ${t.toFixed(3)}`;
    drawImage(canvas, shown(image), image_shape, ink);
    canvas.addEventListener("click", () => setInterpolation(t));
    return canvas;
  }));
});

function refreshStrip() {
  const points = Array.from({ length: INTERPOLATION_STEPS }, (_, i) => interpolationPoint(i / (INTERPOLATION_STEPS - 1)));
  decodeStrip(points);
}

// Moves z to the point t of the line between both codes
function setInterpolation(t) {
  state.interpolation.t = t;
  $("interp-t").value = t;
  $("interp-t-text").textContent = t.toFixed(2);
  state.z = interpolationPoint(t);
  update();
}

function buildInterpolation() {
  const { labels } = state.data;
  for (const [id, key] of [["interp-from", "from"], ["interp-to", "to"]]) {
    const select = $(id);
    select.replaceChildren(...labels.map((label, i) => new Option(`${label} (#${i})`, i)));
    select.value = state.interpolation[key];
    select.addEventListener("change", () => {
      state.interpolation[key] = Number(select.value);
      refreshStrip();
      setInterpolation(state.interpolation.t);
    });
  }
  $("interp-t").addEventListener("input", () => setInterpolation(Number($("interp-t").value)));
  $("interp-t-text").textContent = state.interpolation.t.toFixed(2);
  refreshStrip();
}

$("random").addEventListener("click", () => {
  state.z = state.data.bounds.map(([low, high]) => low + Math.random() * (high - low));
  update();
});

$("mean").addEventListener("click", () => {
  const { latent, latent_dim } = state.data;
  state.z = Array.from({ length: latent_dim }, (_, k) => latent.reduce((s, code) => s + code[k], 0) / latent.length);
  update();
});

$("reset").addEventListener("click", () => {
  if (state.selected !== null) selectSample(state.selected);
});

window.addEventListener("resize", () => drawMap());
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => drawMap());

// ---------------------------------------------------------------- start

async function start() {
  try {
    state.data = await api("/api/dataset");
  } catch (error) {
    $("summary").textContent = `Could not load the dataset: ${error.message}`;
    return;
  }
  const { labels, latent_dim, image_shape } = state.data;
  state.axisY = latent_dim > 1 ? 1 : 0;
  $("binarize-label").hidden = !state.data.binary;

  // Optional initial view from the URL: ?x=0&y=2&grid=1&size=15&binarize=1&sample=4
  const params = new URLSearchParams(location.search);
  const axis = (name, fallback) => {
    const k = Number(params.get(name));
    return params.has(name) && Number.isInteger(k) && k >= 0 && k < latent_dim ? k : fallback;
  };
  state.axisX = axis("x", state.axisX);
  state.axisY = axis("y", state.axisY);
  state.grid = params.get("grid") === "1";
  state.gridSize = Math.min(40, Math.max(3, Number(params.get("size")) || state.gridSize));
  state.binarize = state.data.binary && params.get("binarize") === "1";
  // Variational extras: sample from the prior and σ(x) ellipses (?sigma=1)
  $("prior").hidden = !state.data.variational;
  $("sigma-label").hidden = !state.data.latent_sigma;
  state.sigma = Boolean(state.data.latent_sigma) && params.get("sigma") === "1";
  $("sigma-toggle").checked = state.sigma;
  // Interpolation between two samples (?from=3&to=7)
  const index = (name, fallback) => {
    const i = Number(params.get(name));
    return params.has(name) && Number.isInteger(i) && i >= 0 && i < labels.length ? i : fallback;
  };
  state.interpolation.from = index("from", 0);
  state.interpolation.to = index("to", Math.min(1, labels.length - 1));
  $("grid-toggle").checked = state.grid;
  $("grid-size").value = state.gridSize;
  $("binarize").checked = state.binarize;
  const sample = Number(params.get("sample"));
  $("summary").textContent = `${labels.length} samples · images ${image_shape.height}x${image_shape.width}x${image_shape.channels} · latent space of ${latent_dim} dimension${latent_dim === 1 ? "" : "s"}`;

  buildAxes();
  buildSliders();
  buildTable();
  buildInterpolation();
  selectSample(Number.isInteger(sample) && sample >= 0 && sample < labels.length ? sample : 0);
}

start();
