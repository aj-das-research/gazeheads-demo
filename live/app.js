// Gaze-head steering of Qwen3-VL-2B, in the browser.
//
// The model runs on WebGPU through transformers.js. Its ONNX decoder carries
// two extra graph inputs -- gaze_head_mask (layers x heads) and gaze_sign
// (one value per KV position) -- whose product is added to the attention
// scores before softmax. Steering is therefore a matter of feeding those two
// tensors on every decode step; this file computes them from the cursor.
//
// Vision features are precomputed per case (see tools/build_webdemo.py in the
// research repo): the browser never runs a vision encoder. We hand them to the
// library by replacing model.encode_image, and everything else -- chat
// template, image placeholders, rotary positions, KV cache, streaming -- is the
// library's normal path.

import {
  AutoProcessor, AutoModelForImageTextToText, Tensor, TextStreamer, RawImage,
  InterruptableStoppingCriteria,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

const MODEL_ID = 'baulab/Qwen3-VL-2B-Instruct-GazeHeads-AllLayers-ONNX';
const N_LAYERS = 28, N_HEADS = 16, HIDDEN = 2048;
const MAX_NEW_TOKENS = 64;
const asset = (p) => new URL(p, import.meta.url).href;
const $ = (id) => document.getElementById(id);

// float32 -> IEEE 754 half, as the uint16 bit patterns ONNX Runtime expects.
// Round-to-nearest-even; the steering values (0, 1, +/-delta <= 16) are all
// exactly representable, so precision is not a concern here.
const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
function f16bits(v) {
  _f32[0] = v;
  const x = _u32[0], sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff, mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // inf / nan
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;                           // overflow
  if (exp <= 0) {                                                  // subnormal / zero
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >>> (1 - exp);
    return sign | ((mant + 0x1000) >>> 13);
  }
  return sign | (exp << 10) | ((mant + 0x1000) >>> 13);
}
function toF16(arr) {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = f16bits(arr[i]);
  return out;
}

// ---------------------------------------------------------------- steering --
// Owns the two steering tensors and the decoder-session patch that feeds them.
class Steering {
  constructor() {
    this.headMask = new Float32Array(N_LAYERS * N_HEADS);
    this.boost = new Set();       // absolute KV positions to push attention toward
    this.suppress = new Set();    // ... and away from (the other image tokens)
    this.delta = 4;
    this.active = false;
    this.calls = 0;
  }
  setHeads(ranking, topK) {
    this.headMask.fill(0);
    for (const e of ranking.slice(0, topK)) this.headMask[e.layer * N_HEADS + e.head] = 1;
  }
  setRegion(boostPositions, allImagePositions) {
    this.boost = new Set(boostPositions);
    this.suppress = new Set(allImagePositions.filter((p) => !this.boost.has(p)));
  }
  // transformers.js hides unknown graph inputs from the session's inputNames
  // list; expose the two we need and append them to every run() feed. Steering
  // is decode-only: the prefill (q_len > 1) builds the KV cache unmodified.
  attach(session) {
    const names = [...session.inputNames];
    // Hide the two steering inputs from the list transformers.js validates feeds
    // against, then add them ourselves in run(); otherwise it throws
    // "Missing the following inputs" before run() is ever reached.
    const hidden = ['gaze_head_mask', 'gaze_sign'].filter((n) => names.includes(n));
    Object.defineProperty(session, 'inputNames', {
      get: () => names.filter((n) => !hidden.includes(n)), configurable: true,
    });
    const run = session.run.bind(session);
    const self = this;
    session.run = async function (feeds, ...rest) {
      const seqLen = feeds.inputs_embeds ? Number(feeds.inputs_embeds.dims[1]) : 1;
      const totalLen = feeds.attention_mask ? Number(feeds.attention_mask.dims[1]) : seqLen;
      const steer = self.active && seqLen === 1;
      self.calls += 1;
      const sign = new Float32Array(totalLen);
      const mask = steer ? self.headMask : new Float32Array(N_LAYERS * N_HEADS);
      if (steer) {
        for (const p of self.boost) if (p < totalLen) sign[p] = +self.delta;
        for (const p of self.suppress) if (p < totalLen) sign[p] = -self.delta;
      }
      // ORT float16 tensors carry IEEE half bit patterns in a Uint16Array.
      feeds.gaze_head_mask = new Tensor('float16', toF16(mask), [N_LAYERS, N_HEADS]);
      feeds.gaze_sign = new Tensor('float16', toF16(sign), [totalLen]);
      return run(feeds, ...rest);
    };
  }
}

// ------------------------------------------------------------------- state --
const S = {
  processor: null, model: null, config: null, ranking: null,
  steer: new Steering(), cases: [], meta: null, embeds: null, image: null,
  inputs: null, imagePositions: [], grid: { h: 0, w: 0 },
  generating: false, stopper: null, hover: null, spotlightCells: 2.5,
};

const status = (t, warn = false) => { const el = $('status'); el.textContent = t; el.classList.toggle('warn', warn); };

// -------------------------------------------------------------- geometry --
// Cursor -> image-token positions. The image tokens are a row-major h x w grid
// after the processor's 2x2 merge; token k sits at (row k / w, col k % w).
function cellOf(x, y) {
  const r = Math.floor((y / S.image.height) * S.grid.h);
  const c = Math.floor((x / S.image.width) * S.grid.w);
  return [Math.min(Math.max(r, 0), S.grid.h - 1), Math.min(Math.max(c, 0), S.grid.w - 1)];
}
function cellsWithin(rc, radius) {
  const out = [];
  for (let r = 0; r < S.grid.h; r++)
    for (let c = 0; c < S.grid.w; c++)
      if (Math.hypot(r + 0.5 - (rc[0] + 0.5), c + 0.5 - (rc[1] + 0.5)) <= radius) out.push(r * S.grid.w + c);
  return out;
}
function cellsInBox(b) {
  const [x0, y0, x1, y1] = b;
  const out = [];
  for (let r = 0; r < S.grid.h; r++)
    for (let c = 0; c < S.grid.w; c++) {
      const cx = ((c + 0.5) / S.grid.w) * S.image.width, cy = ((r + 0.5) / S.grid.h) * S.image.height;
      if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) out.push(r * S.grid.w + c);
    }
  return out;
}
function regionUnder(x, y) {
  let best = null;
  for (const reg of S.meta.regions) {
    const [x0, y0, x1, y1] = reg.bbox;
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
      const area = (x1 - x0) * (y1 - y0);
      if (!best || area < best.area) best = { ...reg, area };   // innermost box wins
    }
  }
  return best;
}
// Which tokens the cursor selects, and a label for the readout.
function targetFor(x, y) {
  const snap = $('snap').checked ? regionUnder(x, y) : null;
  const cells = snap ? cellsInBox(snap.bbox) : cellsWithin(cellOf(x, y), S.spotlightCells);
  return { cells, label: snap ? snap.label : `spotlight r=${S.spotlightCells.toFixed(1)} cells` };
}

// --------------------------------------------------------------- drawing --
function draw(target) {
  const cv = $('overlay'), img = $('img');
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  const sx = cv.width / S.image.width, sy = cv.height / S.image.height;
  const css = getComputedStyle(document.documentElement);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = css.getPropertyValue('--radiologist').trim();
  ctx.font = '11px "IBM Plex Mono", monospace';
  for (const reg of S.meta.regions) {
    const [x0, y0, x1, y1] = reg.bbox;
    ctx.globalAlpha = 0.55;
    ctx.strokeRect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy);
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(reg.label, x0 * sx + 3, Math.max(10, y0 * sy - 3));
  }
  if (target) {
    ctx.fillStyle = css.getPropertyValue('--steered').trim();
    ctx.globalAlpha = 0.32;
    const cw = cv.width / S.grid.w, ch = cv.height / S.grid.h;
    for (const k of target.cells) ctx.fillRect((k % S.grid.w) * cw, Math.floor(k / S.grid.w) * ch, cw, ch);
  }
  ctx.globalAlpha = 1;
}

// ----------------------------------------------------------------- model --
async function loadModel() {
  if (!navigator.gpu) { status('This browser has no WebGPU; use recent Chrome or Edge on a desktop.', true); return false; }
  // The only published build of this model is q4f16, whose embedding kernel
  // needs half-precision shaders. Check before the 1.2 GB download, not after:
  // a GPU without shader-f16 fails at the first token with an ONNX Runtime
  // error the visitor cannot act on.
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { status('WebGPU is present but no GPU adapter was offered; try Chrome with hardware acceleration on.', true); return false; }
  const gpuName = `${adapter.info?.vendor || ''} ${adapter.info?.architecture || ''}`.trim() || 'unknown GPU';
  if (!adapter.features.has('shader-f16')) {
    status(`Your GPU adapter (${gpuName}) does not expose shader-f16, which this model build requires. ` +
           'Try Chrome/Edge on a laptop or desktop with a recent NVIDIA, AMD, Apple or Intel Arc GPU; software rendering will not work.', true);
    return false;
  }
  status(`GPU: ${gpuName} · shader-f16 OK`);
  $('start').disabled = true;
  $('dl').classList.remove('hidden');
  status('Downloading the model — one time, cached by the browser afterwards');
  const files = new Map();
  const bar = () => {
    let l = 0, t = 0; for (const f of files.values()) { l += f.loaded ?? 0; t += f.total ?? 0; }
    if (!t) return; $('prog').value = (l / t) * 100; $('dlLabel').textContent = `${(l / 1e9).toFixed(2)} / ${(t / 1e9).toFixed(2)} GB`;
  };
  S.processor = await AutoProcessor.from_pretrained(MODEL_ID);
  S.config = await (await fetch(`https://huggingface.co/${MODEL_ID}/resolve/main/config.json`)).json();
  S.model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
    dtype: { embed_tokens: 'q4f16', vision_encoder: 'q4f16', decoder_model_merged: 'q4f16' },
    device: 'webgpu',
    progress_callback: (p) => { if (p.status === 'progress' && p.file && p.total) { files.set(p.file, p); bar(); } },
  });
  $('dl').classList.add('hidden');
  const decoder = Object.values(S.model.sessions).find((s) => s.inputNames?.includes?.('gaze_sign'));
  if (!decoder) { status('This model build has no steering inputs.', true); return false; }
  S.steer.attach(decoder);
  // The vision encoder is never run: hand the library our precomputed features.
  S.model.encode_image = async () => new Tensor('float32', S.embeds, [S.embeds.length / HIDDEN, HIDDEN]);
  S.ranking = await (await fetch(asset('gaze_head_ranking.json'))).json();
  S.steer.setHeads(S.ranking, +$('topk').value);
  return true;
}

async function loadCase(name) {
  S.meta = await (await fetch(asset(`cases/${name}.json`))).json();
  S.embeds = new Float32Array(await (await fetch(asset(`cases/${S.meta.embeds.file}`))).arrayBuffer());
  S.grid = { h: S.meta.grid_thw[1], w: S.meta.grid_thw[2] };
  S.image = { width: S.meta.width, height: S.meta.height };
  $('img').src = asset(`cases/${name}.jpg`);
  $('caseMeta').textContent = `${S.meta.modality} · ${name} · ${S.grid.h}×${S.grid.w} image tokens`;
  $('out').textContent = ''; $('base').textContent = ''; $('outMeta').textContent = '';
  await prepareInputs();
  draw(null);
}

// Tokenise the chat prompt with the image placeholders the processor expects,
// then locate the image tokens so the spotlight can address them.
async function prepareInputs() {
  const image = await RawImage.fromURL(asset(`cases/${S.meta.name}.jpg`));
  const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: $('prompt').value }] }];
  const text = S.processor.apply_chat_template(messages, { add_generation_prompt: true });
  S.inputs = await S.processor(text, image);
  const ids = S.inputs.input_ids.data, imageId = BigInt(S.config.image_token_id);
  S.imagePositions = [];
  for (let i = 0; i < ids.length; i++) if (ids[i] === imageId) S.imagePositions.push(i);
  if (S.imagePositions.length !== S.embeds.length / HIDDEN) {
    status(`token mismatch: prompt has ${S.imagePositions.length} image tokens, features cover ${S.embeds.length / HIDDEN}`, true);
  }
}

// One generation. Steering state is read on every decode step by the patched
// session, so moving the cursor re-steers the sentence already in flight.
async function generate(target, into, tint) {
  if (S.generating) { S.stopper?.interrupt(); await new Promise((r) => setTimeout(r, 30)); }
  S.generating = true;
  S.stopper = new InterruptableStoppingCriteria();
  S.steer.active = !!target;
  if (target) S.steer.setRegion(target.cells.map((k) => S.imagePositions[k]), S.imagePositions);
  S.steer.delta = +$('bias').value;
  into.textContent = '';
  const t0 = performance.now(); let n = 0;
  const streamer = new TextStreamer(S.processor.tokenizer, {
    skip_prompt: true, skip_special_tokens: true,
    callback_function: (piece) => {
      n += 1;
      const span = document.createElement('span');
      span.textContent = piece;
      if (tint && S.hover) span.style.background = `color-mix(in srgb, var(--steered) ${tint}%, transparent)`;
      into.appendChild(span);
    },
  });
  try {
    await S.model.generate({
      ...S.inputs, max_new_tokens: MAX_NEW_TOKENS, do_sample: false, repetition_penalty: 1.1,
      streamer, stopping_criteria: S.stopper,
    });
    const dt = (performance.now() - t0) / 1000;
    $('outMeta').textContent = `${n} tokens · ${dt.toFixed(1)}s · ${(n / dt).toFixed(1)} tok/s · ` +
      (target ? `${target.label} · ${target.cells.length} of ${S.imagePositions.length} image tokens · δ=${S.steer.delta}` : 'unsteered');
  } catch (e) {
    if (!/interrupt/i.test(String(e))) { status(String(e), true); console.error(e); }
  } finally {
    S.generating = false; S.steer.active = false;
  }
}

// ------------------------------------------------------------------- ui --
function wireUI() {
  const scan = $('scan');
  const toImage = (ev) => {
    const r = $('img').getBoundingClientRect();
    return [((ev.clientX - r.left) / r.width) * S.image.width, ((ev.clientY - r.top) / r.height) * S.image.height];
  };
  let lastKey = '';
  scan.addEventListener('pointermove', (ev) => {
    if (!S.meta || !S.model) return;
    const [x, y] = toImage(ev);
    const target = targetFor(x, y);
    S.hover = target;
    draw(target);
    // Update the live steering target; start a generation if none is running.
    if (S.generating) S.steer.setRegion(target.cells.map((k) => S.imagePositions[k]), S.imagePositions);
    const key = target.cells.join(',');
    if (!S.generating && key !== lastKey) generate(target, $('out'), 22);
    lastKey = key;
  });
  scan.addEventListener('pointerleave', () => { S.hover = null; draw(null); S.stopper?.interrupt(); lastKey = ''; });
  scan.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    S.spotlightCells = Math.min(8, Math.max(1, S.spotlightCells + (ev.deltaY < 0 ? 0.5 : -0.5)));
    const [x, y] = toImage(ev); const target = targetFor(x, y); S.hover = target; draw(target);
    if (S.generating) S.steer.setRegion(target.cells.map((k) => S.imagePositions[k]), S.imagePositions);
  }, { passive: false });
  $('bias').oninput = (e) => { $('biasV').textContent = e.target.value; S.steer.delta = +e.target.value; };
  $('topk').oninput = (e) => { $('topkV').textContent = e.target.value; if (S.ranking) S.steer.setHeads(S.ranking, +e.target.value); };
  $('case').onchange = async (e) => { S.stopper?.interrupt(); await loadCase(e.target.value); };
  $('prompt').onchange = async () => { if (S.meta) await prepareInputs(); };
  $('baselineBtn').onclick = () => generate(null, $('base'), 0);
  window.addEventListener('resize', () => S.meta && draw(S.hover));
}

async function main() {
  wireUI();
  S.cases = await (await fetch(asset('cases/cases.json'))).json();
  for (const c of S.cases) { const o = document.createElement('option'); o.value = c.name; o.textContent = c.label; $('case').append(o); }
  $('start').onclick = async () => {
    try {
      if (!(await loadModel())) return;
      status('Loading a case…');
      await loadCase(S.cases[0].name);
      for (const id of ['case', 'bias', 'topk', 'baselineBtn']) $(id).disabled = false;
      status('ready — hover the scan and the model starts writing');
    } catch (e) { status(String(e), true); console.error(e); $('start').disabled = false; }
  };
  if (!navigator.gpu) status('needs a WebGPU browser (recent Chrome or Edge on a laptop/desktop)', true);
  else status('ready to start');
}
main();
