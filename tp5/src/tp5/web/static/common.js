// Helpers shared by the pages (latent explorer, denoising). Images are flattened (height, width, channels),
// channels-last, values in [0, 1].

// ---------------------------------------------------------------- images

// Paints a flattened image into an ImageData (grayscale with ink = 1 black, or RGB)
export function toImageData(pixels, shape, ink) {
  const { height, width, channels } = shape;
  const data = new ImageData(width, height);
  for (let i = 0; i < height * width; i++) {
    let r, g, b;
    if (channels === 1) {
      const v = Math.round(255 * (ink ? 1 - pixels[i] : pixels[i]));
      r = g = b = v;
    } else {
      r = Math.round(255 * pixels[i * channels]);
      g = Math.round(255 * pixels[i * channels + 1]);
      b = Math.round(255 * pixels[i * channels + 2]);
    }
    data.data.set([r, g, b, 255], i * 4);
  }
  return data;
}

// Draws an image at native resolution on its own canvas (CSS scales it, pixelated)
export function drawImage(canvas, pixels, shape, ink) {
  canvas.width = shape.width;
  canvas.height = shape.height;
  canvas.getContext("2d").putImageData(toImageData(pixels, shape, ink), 0, 0);
}

// Draws an image scaled into a rectangle of another canvas
const scratch = document.createElement("canvas");
export function drawImageAt(ctx, pixels, shape, ink, x, y, w, h) {
  scratch.width = shape.width;
  scratch.height = shape.height;
  scratch.getContext("2d").putImageData(toImageData(pixels, shape, ink), 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, x, y, w, h);
}

// Thresholded copy (binary datasets show outputs as is or binarized like the pixel metric)
export const binarized = (pixels) => pixels.map((v) => (v >= 0.5 ? 1 : 0));

// Error between two images: wrong pixels after binarizing (binary datasets) or mean absolute error
export function imageError(a, b, binary) {
  let total = 0;
  for (let k = 0; k < a.length; k++) {
    total += binary ? ((a[k] >= 0.5) !== (b[k] >= 0.5) ? 1 : 0) : Math.abs(a[k] - b[k]);
  }
  return binary ? total : total / a.length;
}

// ---------------------------------------------------------------- api

export async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

// Runs the latest requested job only, one at a time (slider drags request faster than the server answers)
export function latestOnly(job) {
  let pending = null;
  let running = false;
  async function flush() {
    if (running || pending === null) return;
    const args = pending;
    pending = null;
    running = true;
    try {
      await job(...args);
    } catch (error) {
      console.error(error);
    } finally {
      running = false;
      flush();
    }
  }
  return (...args) => { pending = args; flush(); };
}

// ---------------------------------------------------------------- text / dom

export const $ = (id) => document.getElementById(id);
export const fmt = (v) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : v.toFixed(3));
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
