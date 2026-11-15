// Denoising page: corrupt a training sample (same noise functions as training, /api/noise), load or save
// noisy pattern files (/api/noisy) or edit pixels by hand, and see what the trained autoencoder outputs
// (/api/reconstruct). Dataset-agnostic like the latent explorer (flattened (h, w, c) images in [0, 1]).

import { $, api, binarized, drawImage, escapeHtml, fmt, imageError, latestOnly } from "./common.js";

const state = {
  data: null,
  index: 0,
  noisy: null,
  // Where the noisy image comes from: generated here, or a noisy file ({ name, content, variant })
  file: null,
  edited: false,
  binarize: false,
  output: null,
  all: false,
};

const NOISE_HINTS = {
  flip: "flip: each pixel is inverted (x → 1 − x) with probability = level.",
  salt_pepper: "salt_pepper: each pixel is set to 0 or 1 at random with probability = level.",
  gaussian: "gaussian: x + N(0, level²), clipped to [0, 1].",
};

// ---------------------------------------------------------------- helpers

const size = () => state.data.images[0].length;
const error = (a, b) => imageError(a, b, state.data.binary);
const errorText = (value) => (state.data.binary ? `${value} px` : `MAE ${fmt(value)}`);
const shown = (pixels) => (state.binarize ? binarized(pixels) : pixels);

function status(id, text, isError = false) {
  $(id).textContent = text;
  $(id).classList.toggle("error", isError);
}

function noiseParameters() {
  const seedText = $("noise-seed").value.trim();
  return {
    type: $("noise-type").value,
    level: Number($("noise-level").value),
    seed: seedText === "" ? null : Number(seedText),
  };
}

// Description of the current noisy image, stored with saved files and figures
function currentNoise() {
  let noise;
  if (state.file) {
    const { content, name, variant } = state.file;
    noise = { ...(content.noise || { type: "manual" }), file: name };
    if (content.noisy.length > 1) noise.variant = variant + 1;
  } else {
    noise = noiseParameters();
  }
  if (state.edited) noise.edited = true;
  return noise;
}

// ---------------------------------------------------------------- drawing

function buildSamples() {
  const { labels, images, image_shape, ink } = state.data;
  $("samples").replaceChildren(...labels.map((label, i) => {
    const button = document.createElement("button");
    button.title = `'${label}' (#${i})`;
    const canvas = document.createElement("canvas");
    canvas.className = "image";
    drawImage(canvas, images[i], image_shape, ink);
    button.append(canvas);
    button.addEventListener("click", () => selectSample(i));
    return button;
  }));
}

function markSample() {
  [...$("samples").children].forEach((button, i) => button.classList.toggle("selected", i === state.index));
}

function drawInput() {
  const { images, image_shape, ink, labels } = state.data;
  const original = images[state.index];
  drawImage($("original"), original, image_shape, ink);
  drawImage($("noisy"), state.noisy, image_shape, ink);
  $("noisy-caption").innerHTML = `noisy input · <strong>${errorText(error(state.noisy, original))}</strong>`;

  const noise = currentNoise();
  const parts = [`sample '${escapeHtml(labels[state.index])}' (#${state.index})`];
  if (state.file) parts.push(`file ${escapeHtml(state.file.name)}${noise.variant ? `, variant ${noise.variant}/${state.file.content.noisy.length}` : ""}`);
  if (noise.type) parts.push(`noise ${escapeHtml(noise.type)}${noise.level != null ? ` ${noise.level}` : ""}${noise.seed != null ? `, seed ${noise.seed}` : ""}`);
  if (state.edited) parts.push("edited by hand");
  $("source").innerHTML = parts.join(" · ");
  markSample();
}

function drawOutput() {
  if (!state.output) return;
  const { images, image_shape, ink, labels, binary, tolerance } = state.data;
  const { output, latent, nearest, distance } = state.output;
  const outputError = error(output, images[state.index]);
  drawImage($("output"), shown(output), image_shape, ink);
  drawImage($("nearest"), images[nearest], image_shape, ink);
  const good = binary && outputError <= tolerance;
  $("output-caption").innerHTML = `output · <strong class="${good ? "good" : ""}">${errorText(outputError)}</strong>`;
  $("nearest-caption").textContent = `nearest: ${labels[nearest]} (${errorText(distance)})`;
  $("z-text").textContent = `[${latent.map(fmt).join(", ")}]`;
}

// ---------------------------------------------------------------- model calls

const reconstruct = latestOnly(async (noisy) => {
  const result = await api("/api/reconstruct", { images: [noisy] });
  state.output = { output: result.outputs[0], latent: result.latent[0], nearest: result.nearest[0], distance: result.distance[0] };
  drawOutput();
});

function setNoisy(noisy) {
  state.noisy = noisy;
  drawInput();
  reconstruct(noisy.slice());
}

// New noisy version of the selected sample with the parameters of the controls
const generate = latestOnly(async (keepStatus = false) => {
  const parameters = noiseParameters();
  try {
    const { images, seed } = await api("/api/noise", { samples: [state.index], ...parameters });
    $("noise-seed").value = seed;
    state.file = null;
    state.edited = false;
    $("variant-controls").hidden = true;
    if (!keepStatus) status("input-status", "");
    setNoisy(images[0]);
    refreshAll();
  } catch (err) {
    status("input-status", err.message, true);
  }
});

function selectSample(i) {
  state.index = i;
  generate();
}

// ---------------------------------------------------------------- manual editing

$("noisy").addEventListener("click", (event) => {
  if (!state.noisy) return;
  const { width, height, channels } = state.data.image_shape;
  const rect = event.target.getBoundingClientRect();
  const column = Math.floor(((event.clientX - rect.left) / rect.width) * width);
  const row = Math.floor(((event.clientY - rect.top) / rect.height) * height);
  if (column < 0 || column >= width || row < 0 || row >= height) return;
  const noisy = state.noisy.slice();
  const first = (row * width + column) * channels;
  for (let c = 0; c < channels; c++) noisy[first + c] = 1 - noisy[first + c];
  state.edited = true;
  setNoisy(noisy);
});

// ---------------------------------------------------------------- noisy files

async function refreshFiles(select) {
  try {
    const { files } = await api("/api/noisy");
    const list = $("noisy-files");
    list.replaceChildren(...(files.length ? files.map((name) => new Option(name, name)) : [new Option("(no files)", "")]));
    list.disabled = $("load-file").disabled = files.length === 0;
    if (select && files.includes(select)) list.value = select;
  } catch (err) {
    status("input-status", `Could not list the noisy files: ${err.message}`, true);
  }
}

// Checks a noisy file against the served dataset and shows its first variant
function applyFile(content, name) {
  const { image_shape, labels } = state.data;
  if (!content || typeof content !== "object" || !Array.isArray(content.noisy) || content.noisy.length === 0) {
    throw new Error("not a noisy pattern file (missing 'noisy').");
  }
  const shape = content.image_shape || {};
  if (shape.height !== image_shape.height || shape.width !== image_shape.width || (shape.channels ?? 1) !== image_shape.channels) {
    throw new Error(`its images are ${shape.height}×${shape.width}×${shape.channels ?? 1}, the model uses ${image_shape.height}×${image_shape.width}×${image_shape.channels}.`);
  }
  // The label must be a sample of the dataset (the index disambiguates repeated labels)
  let index = Number.isInteger(content.index) && labels[content.index] === content.label ? content.index : labels.indexOf(content.label);
  if (index < 0) throw new Error(`sample '${content.label}' is not in the dataset of this model.`);
  const valid = content.noisy.every((image) => Array.isArray(image) && image.length === size() && image.every(Number.isFinite));
  if (!valid) throw new Error(`every noisy image must have ${size()} numeric values.`);

  state.index = index;
  state.file = { name, content, variant: 0 };
  // Controls show the file's noise, so "New noise" continues from it
  const noise = content.noise || {};
  if (state.data.noise_types.includes(noise.type)) $("noise-type").value = noise.type;
  if (typeof noise.level === "number") setLevel(noise.level);
  $("noise-seed").value = Number.isInteger(noise.seed) ? noise.seed : "";
  status("input-status", `Loaded ${name}: '${content.label}', ${content.noisy.length} variant${content.noisy.length === 1 ? "" : "s"}.`);
  showVariant(0);
  refreshAll();
}

function showVariant(k) {
  const count = state.file.content.noisy.length;
  state.file.variant = (k + count) % count;
  state.edited = false;
  $("variant-controls").hidden = count === 1;
  $("variant-text").textContent = `variant ${state.file.variant + 1} / ${count}`;
  setNoisy(state.file.content.noisy[state.file.variant].map((v) => Math.min(1, Math.max(0, v))));
}

async function loadServerFile(name) {
  if (!name) return false;
  try {
    applyFile(await api(`/api/noisy/${encodeURIComponent(name)}`), name);
    return true;
  } catch (err) {
    status("input-status", `Cannot load ${name}: ${err.message}`, true);
    return false;
  }
}

$("load-file").addEventListener("click", () => loadServerFile($("noisy-files").value));
$("refresh-files").addEventListener("click", () => refreshFiles($("noisy-files").value));
$("prev-variant").addEventListener("click", () => showVariant(state.file.variant - 1));
$("next-variant").addEventListener("click", () => showVariant(state.file.variant + 1));

$("local-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    applyFile(JSON.parse(await file.text()), file.name);
  } catch (err) {
    status("input-status", `Cannot load ${file.name}: ${err.message}`, true);
  }
  event.target.value = "";
});

$("save-noisy").addEventListener("click", async () => {
  try {
    const { file } = await api("/api/noisy", { sample: state.index, noise: currentNoise(), noisy: [state.noisy] });
    status("input-status", `Saved ${file}`);
    refreshFiles(file.split("/").pop());
  } catch (err) {
    status("input-status", `Could not save: ${err.message}`, true);
  }
});

// ---------------------------------------------------------------- report figure

async function save() {
  const name = $("save-name").value.trim();
  if (!name) {
    status("saved", "Type a file name first.", true);
    $("save-name").focus();
    return;
  }
  $("save").disabled = true;
  status("saved", "Saving…");
  try {
    const { files } = await api("/api/save-denoising", { sample: state.index, noisy: state.noisy, noise: currentNoise(), name });
    $("save-name").value = "";
    status("saved", `Saved: ${files.join(", ")}`);
  } catch (err) {
    status("saved", `Could not save: ${err.message}`, true);
  } finally {
    $("save").disabled = false;
  }
}

$("save").addEventListener("click", save);
$("save-name").addEventListener("keydown", (event) => { if (event.key === "Enter") save(); });

// ---------------------------------------------------------------- all samples

const runAll = latestOnly(async (parameters) => {
  const { labels, images, image_shape, ink, binary, tolerance } = state.data;
  const indices = labels.map((_, i) => i);
  const { images: noisy } = await api("/api/noise", { samples: indices, ...parameters });
  const { outputs, nearest } = await api("/api/reconstruct", { images: noisy });

  const noisyErrors = indices.map((i) => error(noisy[i], images[i]));
  const outputErrors = indices.map((i) => error(outputs[i], images[i]));
  const thumbnail = (pixels) => {
    const canvas = document.createElement("canvas");
    canvas.className = "image";
    drawImage(canvas, pixels, image_shape, ink);
    return canvas;
  };

  const head = ["sample", "original", "noisy", "output", "nearest", "noisy error", "output error"];
  const tr = document.createElement("tr");
  head.forEach((name) => { const th = document.createElement("th"); th.textContent = name; tr.append(th); });
  $("all-table").tHead.replaceChildren(tr);
  $("all-table").tBodies[0].replaceChildren(...indices.map((i) => {
    const row = document.createElement("tr");
    const cells = [labels[i], thumbnail(images[i]), thumbnail(noisy[i]), thumbnail(shown(outputs[i])), labels[nearest[i]], errorText(noisyErrors[i]), errorText(outputErrors[i])];
    cells.forEach((content, column) => {
      const td = document.createElement("td");
      if (content instanceof HTMLElement) td.append(content); else td.textContent = content;
      if (column === 4 && nearest[i] !== i) td.className = "bad";
      if (column === 6 && binary) td.className = outputErrors[i] <= tolerance ? "good" : "bad";
      row.append(td);
    });
    row.addEventListener("click", () => selectSample(i));
    return row;
  }));

  const mean = (values) => values.reduce((s, v) => s + v, 0) / values.length;
  const recognized = indices.filter((i) => nearest[i] === i).length;
  let summary = `${parameters.type} ${parameters.level}, seed ${parameters.seed} · mean noisy error ${errorText(Number(mean(noisyErrors).toFixed(binary ? 2 : 4)))} → mean output error ${errorText(Number(mean(outputErrors).toFixed(binary ? 2 : 4)))} · output nearest to its own sample: ${recognized}/${labels.length}`;
  if (binary) summary += ` · recovered (≤ ${tolerance} px): ${outputErrors.filter((e) => e <= tolerance).length}/${labels.length}`;
  $("all-summary").textContent = summary;
});

function refreshAll() {
  $("all-wrap").hidden = !state.all;
  if (!state.all) { $("all-summary").textContent = ""; return; }
  const parameters = noiseParameters();
  if (parameters.seed === null) return;
  runAll(parameters);
}

$("all-toggle").addEventListener("change", (event) => { state.all = event.target.checked; refreshAll(); });

// ---------------------------------------------------------------- controls

function setLevel(level) {
  $("noise-level").value = level;
  $("noise-level-value").value = level;
}

$("noise-type").addEventListener("change", () => {
  $("noise-hint").textContent = NOISE_HINTS[$("noise-type").value] || "";
  generate();
});
// Level and seed changes keep the seed: the same random draw at another level
$("noise-level").addEventListener("input", () => { $("noise-level-value").value = $("noise-level").value; generate(); });
$("noise-level-value").addEventListener("change", () => {
  const level = Math.min(1, Math.max(0, Number($("noise-level-value").value) || 0));
  setLevel(level);
  generate();
});
$("noise-seed").addEventListener("change", () => generate());
$("new-noise").addEventListener("click", () => { $("noise-seed").value = ""; generate(); });

$("binarize").addEventListener("change", (event) => {
  state.binarize = event.target.checked;
  drawOutput();
  refreshAll();
});

// ---------------------------------------------------------------- start

async function start() {
  try {
    state.data = await api("/api/dataset");
  } catch (err) {
    $("summary").textContent = `Could not load the dataset: ${err.message}`;
    return;
  }
  const { labels, image_shape, noise, noise_types, binary } = state.data;
  $("summary").textContent = `${labels.length} samples · images ${image_shape.height}×${image_shape.width}×${image_shape.channels} · `
    + (noise ? `trained with ${noise.type} noise ${noise.level} (${noise.copies} copies/epoch)` : "trained without noise (basic autoencoder: baseline to compare)");
  $("binarize-label").hidden = !binary;
  $("noise-type").replaceChildren(...noise_types.map((type) => new Option(type, type)));

  // Defaults: training noise; optional URL ?sample=a|3&type=flip&level=0.2&seed=7&all=1&binarize=1&file=name.json
  const params = new URLSearchParams(location.search);
  const type = params.get("type") ?? noise?.type ?? noise_types[0];
  $("noise-type").value = noise_types.includes(type) ? type : noise_types[0];
  $("noise-hint").textContent = NOISE_HINTS[$("noise-type").value] || "";
  const level = Number(params.get("level") ?? noise?.level ?? 0.1);
  setLevel(Number.isFinite(level) ? Math.min(1, Math.max(0, level)) : 0.1);
  const seed = Number(params.get("seed"));
  if (params.has("seed") && Number.isInteger(seed) && seed >= 0) $("noise-seed").value = seed;
  state.binarize = binary && params.get("binarize") === "1";
  $("binarize").checked = state.binarize;
  state.all = params.get("all") === "1";
  $("all-toggle").checked = state.all;

  const sample = params.get("sample");
  if (sample !== null) {
    const byIndex = Number(sample);
    const index = Number.isInteger(byIndex) && byIndex >= 0 && byIndex < labels.length ? byIndex : labels.indexOf(sample);
    if (index >= 0) state.index = index;
  }

  buildSamples();
  await refreshFiles(params.get("file"));
  // A file that cannot be loaded leaves its error visible over the generated noise
  if (!(await loadServerFile(params.get("file")))) generate(true);
}

start();
