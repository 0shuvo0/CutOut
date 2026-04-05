// ─── NOTE ─────────────────────────────────────────────────────────────────────
// This app must be opened as a local file in Chrome/Edge/Firefox.
// It will NOT work inside Claude's artifact viewer, iframes, or sandboxed pages.
// On first run it downloads ~40 MB model from huggingface.co (then caches it).
// ─────────────────────────────────────────────────────────────────────────────

let pipelineModule = null;
let envModule = null;

async function importTransformers() {
  // Try multiple CDN sources for resilience
  const cdns = [
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js',
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0/dist/transformers.min.js',
    'https://unpkg.com/@huggingface/transformers@3.5.2/dist/transformers.min.js',
  ];
  let lastErr;
  for (const url of cdns) {
    try {
      const mod = await import(url);
      return mod;
    } catch(e) { lastErr = e; }
  }
  throw lastErr;
}

const RMBG_MODEL = 'briaai/RMBG-1.4';

let segmenter = null;
const processedImages = [];
let imageCounter = 0;

const loadingScreen  = document.getElementById('loading-screen');
const mainApp        = document.getElementById('main-app');
const progFill       = document.getElementById('prog-fill');
const progLabel      = document.getElementById('prog-label');
const progPct        = document.getElementById('prog-pct');
const stepWasm       = document.getElementById('step-wasm');
const stepModel      = document.getElementById('step-model');
const stepWarmup     = document.getElementById('step-warmup');
const errorBox       = document.getElementById('error-box');
const errorMsg       = document.getElementById('error-msg');
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const resultsSection = document.getElementById('results-section');
const resultsGrid    = document.getElementById('results-grid');
const resultCount    = document.getElementById('result-count');
const loadingSteps   = document.querySelector('.loading-steps');

function setProgress(pct, label) {
  progFill.style.width  = pct + '%';
  progPct.textContent   = Math.round(pct) + '%';
  progLabel.textContent = label;
}

function setStep(el, state) {
  el.className = 'step ' + state;
}

function showError(msg) {
  progFill.style.background = '#ff6b6b';
  setProgress(progFill.style.width || '10%', 'Error — see below');
  loadingSteps.style.display = 'none';
  errorBox.style.display = 'block';
  // Friendly messages for common errors
  if (msg.includes('fetch') || msg.includes('network') || msg.includes('Failed to fetch')) {
    errorMsg.innerHTML = `<strong>Network error:</strong> Could not reach Hugging Face to download the model.<br><br>
    This usually means the page is running in a <strong>restricted environment</strong> (like Claude's artifact viewer) that blocks external requests.<br><br>
    Please download and open the HTML file directly in your browser.`;
  } else if (msg.includes('SharedArrayBuffer') || msg.includes('COOP')) {
    errorMsg.innerHTML = `<strong>Security header error:</strong> ONNX Runtime needs SharedArrayBuffer, which requires specific HTTP headers.<br><br>
    Open the file via a local server: <code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;">npx serve .</code>`;
  } else {
    errorMsg.textContent = msg;
  }
}

async function loadModel() {
  errorBox.style.display = 'none';
  loadingSteps.style.display = 'flex';
  progFill.style.background = '';
  setProgress(0, 'Initializing…');
  setStep(stepWasm,   'active');
  setStep(stepModel,  '');
  setStep(stepWarmup, '');

  try {
    setProgress(3, 'Loading Transformers.js…');
    const mod = await importTransformers();
    const { pipeline, env } = mod;

    env.allowLocalModels = false;
    // Allow cross-origin model loading
    env.useBrowserCache = true;

    setProgress(8, 'ONNX Runtime ready');
    await new Promise(r => setTimeout(r, 300));
    setStep(stepWasm, 'done');

    setStep(stepModel, 'active');
    setProgress(12, 'Connecting to model server…');

    const progressCb = (p) => {
      if (p.status === 'progress' && p.total) {
        const pct = 12 + ((p.loaded / p.total) * 72);
        const mb  = (p.loaded  / 1048576).toFixed(1);
        const tot = (p.total   / 1048576).toFixed(1);
        setProgress(pct, `Downloading model… ${mb} / ${tot} MB`);
      } else if (p.status === 'initiate') {
        setProgress(12, `Fetching: ${p.file || 'model files'}…`);
      } else if (p.status === 'done') {
        setProgress(85, 'Model cached ✓');
      }
    };

    // Try WebGPU first for speed, fall back to WASM
    try {
      segmenter = await pipeline('image-segmentation', RMBG_MODEL, {
        device: 'webgpu',
        progress_callback: progressCb,
      });
      setProgress(86, 'WebGPU acceleration active ⚡');
    } catch (gpuErr) {
      console.warn('WebGPU unavailable, falling back to WASM:', gpuErr.message);
      setProgress(12, 'WebGPU unavailable — using WebAssembly…');
      segmenter = await pipeline('image-segmentation', RMBG_MODEL, {
        progress_callback: progressCb,
      });
    }

    setStep(stepModel, 'done');
    setStep(stepWarmup, 'active');
    setProgress(90, 'Warming up inference engine…');

    // Warmup with a tiny canvas
    const wc = document.createElement('canvas');
    wc.width = wc.height = 64;
    const wctx = wc.getContext('2d');
    wctx.fillStyle = '#888'; wctx.fillRect(0,0,64,64);
    await segmenter(wc.toDataURL());

    setStep(stepWarmup, 'done');
    setProgress(100, 'Ready!');
    await new Promise(r => setTimeout(r, 700));

    loadingScreen.style.opacity = '0';
    loadingScreen.style.transition = 'opacity 0.4s ease';
    await new Promise(r => setTimeout(r, 400));
    loadingScreen.style.display = 'none';
    mainApp.classList.add('visible');

  } catch(err) {
    console.error('Model load error:', err);
    showError(err.message || String(err));
  }
}

window.retryLoad = () => loadModel();

// ─── DRAG & DROP ──────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => { handleFiles([...fileInput.files]); fileInput.value = ''; });

function handleFiles(files) {
  const imgs = files.filter(f => f.type.startsWith('image/'));
  if (!imgs.length) return;
  imgs.forEach(processFile);
}

// ─── PROCESS ──────────────────────────────────────────────────────────────────
async function processFile(file) {
  const id  = ++imageCounter;
  const url = URL.createObjectURL(file);
  resultsSection.style.display = 'flex';
  resultsGrid.prepend(createCard(id, file.name));
  updateCount();

  try {
    const result = await segmenter(url, { threshold: 0.5 });
    const mask   = result[0]?.mask;
    if (!mask) throw new Error('No mask returned from model');
    const outURL = await applyMask(url, mask);
    finalizeCard(id, url, outURL, file.name);
    processedImages.push({ id, name: file.name, dataURL: outURL });
    updateCount();
  } catch(err) {
    console.error(err);
    errorCard(id, err.message);
  }
}

async function applyMask(imageURL, mask) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth, h = img.naturalHeight;
      const canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, w, h);
      const mW = mask.width, mH = mask.height;
      for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
          const mx = Math.round((px / w) * mW);
          const my = Math.round((py / h) * mH);
          imgData.data[(py * w + px) * 4 + 3] = mask.data[my * mW + mx];
        }
      }
      ctx.putImageData(imgData, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = reject;
    img.src = imageURL;
  });
}

// ─── CARDS ────────────────────────────────────────────────────────────────────
function createCard(id, filename) {
  const card = document.createElement('div');
  card.className = 'img-card';
  card.id = 'card-' + id;
  card.innerHTML = `
    <div class="card-preview" id="preview-${id}">
      <div class="processing-overlay">
        <div class="spinner"></div>
        <div class="mini-progress"><div class="mini-fill"></div></div>
        <div class="processing-text">Removing background…</div>
      </div>
    </div>
    <div class="card-info">
      <div class="card-filename" title="${escHtml(filename)}">${escHtml(filename)}</div>
      <div class="card-actions">
        <div class="icon-btn danger" title="Remove" onclick="removeCard(${id})">✕</div>
      </div>
    </div>`;
  return card;
}

function finalizeCard(id, origURL, outURL, filename) {
  const preview = document.getElementById('preview-' + id);
  if (!preview) return;
  preview.innerHTML = `
    <div class="card-after-checker" style="position:absolute;inset:0;background-image:repeating-conic-gradient(#2a2a3a 0% 25%,#1a1a28 0% 50%);background-size:16px 16px;"></div>
    <img class="card-after-img" src="${outURL}" alt="Background removed" draggable="false" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;display:block;">
    <div class="card-before-wrap" id="awrap-${id}" style="position:absolute;inset:0;clip-path:inset(0 50% 0 0);transition:clip-path 0.05s;">
      <div style="position:absolute;inset:0;background-image:url('${origURL}');background-size:cover;background-position:center;"></div>
    </div>
    <div class="card-labels">
      <span class="card-label label-before">Before</span>
      <span class="card-label label-after">After</span>
    </div>
    <div class="slider-line"   id="sline-${id}"></div>
    <div class="slider-handle" id="shandle-${id}">
      <svg viewBox="0 0 24 24" fill="none" stroke="#666" stroke-width="2.5">
        <path d="M8 9l-4 3 4 3M16 9l4 3-4 3"/>
      </svg>
    </div>
    <div class="done-badge">✓</div>`;
  document.querySelector('#card-' + id + ' .card-actions').innerHTML = `
    <div class="icon-btn" title="Download PNG" onclick="downloadSingle(${id})">⬇</div>
    <div class="icon-btn danger" title="Remove" onclick="removeCard(${id})">✕</div>`;
  preview.dataset.output = outURL;
  preview.dataset.name   = filename;
  initSlider(id, preview);
}

function errorCard(id, msg) {
  const p = document.getElementById('preview-' + id);
  if (p) p.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ff6b6b;font-size:12px;padding:20px;text-align:center;">⚠ ${escHtml(msg)}</div>`;
}

// ─── SLIDER ───────────────────────────────────────────────────────────────────
function initSlider(id, preview) {
  let dragging = false;
  const move = (clientX) => {
    const r   = preview.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100));
    const wrap = document.getElementById('awrap-' + id);
    // Clip the "before" overlay from the right — left of slider shows before, right shows after
    if (wrap) wrap.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    document.getElementById('sline-'   + id).style.left = pct + '%';
    document.getElementById('shandle-' + id).style.left = pct + '%';
  };
  preview.addEventListener('mousedown',  e => { dragging = true; move(e.clientX); });
  preview.addEventListener('mousemove',  e => { if (dragging) move(e.clientX); });
  window .addEventListener('mouseup',    () => { dragging = false; });
  preview.addEventListener('touchstart', e => { dragging = true; move(e.touches[0].clientX); }, {passive:true});
  preview.addEventListener('touchmove',  e => { if (dragging) move(e.touches[0].clientX); }, {passive:true});
  window .addEventListener('touchend',   () => { dragging = false; });
}

// ─── DOWNLOADS ────────────────────────────────────────────────────────────────
window.downloadSingle = (id) => {
  const p = document.getElementById('preview-' + id);
  if (!p?.dataset.output) return;
  const a = document.createElement('a');
  a.href = p.dataset.output;
  a.download = (p.dataset.name || 'image').replace(/\.[^.]+$/, '') + '_cutout.png';
  a.click();
};

window.downloadAll = async () => {
  const previews = [...document.querySelectorAll('[id^="preview-"][data-output]')];
  if (!previews.length) return;
  const btn = document.getElementById('download-all-btn');
  btn.textContent = '⏳ Zipping…'; btn.disabled = true;
  try {
    const { default: JSZip } = await import('https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm');
    const zip = new JSZip();
    previews.forEach(p => {
      const name = (p.dataset.name || 'image').replace(/\.[^.]+$/, '') + '_cutout.png';
      zip.file(name, p.dataset.output.split(',')[1], { base64: true });
    });
    const blob = await zip.generateAsync({ type: 'blob' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'cutouts.zip'; a.click();
    URL.revokeObjectURL(a.href);
  } catch {
    previews.forEach(p => {
      const a = document.createElement('a');
      a.href = p.dataset.output;
      a.download = (p.dataset.name || 'image').replace(/\.[^.]+$/, '') + '_cutout.png';
      a.click();
    });
  }
  btn.textContent = '⬇ Download all'; btn.disabled = false;
};

window.removeCard = (id) => {
  const c = document.getElementById('card-' + id);
  if (c) { c.style.transition = '0.2s'; c.style.opacity = '0'; c.style.transform = 'scale(0.95)'; setTimeout(() => c.remove(), 200); }
  setTimeout(updateCount, 250);
};

window.clearAll = () => {
  resultsGrid.innerHTML = '';
  processedImages.length = 0;
  resultsSection.style.display = 'none';
  imageCounter = 0;
  updateCount();
};

function updateCount() {
  const n = resultsGrid.querySelectorAll('.img-card').length;
  resultCount.textContent = n;
  if (n === 0) resultsSection.style.display = 'none';
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadModel();