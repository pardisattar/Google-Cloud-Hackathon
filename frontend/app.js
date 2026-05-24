/* =============================================================================
   Fashion Finder — app.js
   Vanilla JS: upload, filters, search, render
============================================================================= */

'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// ── State ────────────────────────────────────────────────────────────────────
let topKValue        = 10;
let selectedViewpoint = '';
let currentFile      = null;

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const uploadZone         = document.getElementById('uploadZone');
const fileInput          = document.getElementById('fileInput');
const uploadPlaceholder  = document.getElementById('uploadPlaceholder');
const previewWrap        = document.getElementById('previewWrap');
const imagePreview       = document.getElementById('imagePreview');
const clearBtn           = document.getElementById('clearBtn');

const categorySelect     = document.getElementById('categorySelect');
const seasonSelect       = document.getElementById('seasonSelect');
const formalityMin       = document.getElementById('formalityMin');
const formalityMax       = document.getElementById('formalityMax');
const formalityMinDisplay= document.getElementById('formalityMinDisplay');
const formalityMaxDisplay= document.getElementById('formalityMaxDisplay');
const dualSlider         = document.getElementById('dualSlider');
const viewpointToggle    = document.getElementById('viewpointToggle');
const topKDown           = document.getElementById('topKDown');
const topKUp             = document.getElementById('topKUp');
const topKDisplay        = document.getElementById('topKDisplay');

const searchBtn          = document.getElementById('searchBtn');
const searchBtnText      = document.getElementById('searchBtnText');
const searchBtnSpinner   = document.getElementById('searchBtnSpinner');
const searchBtnSpinnerText = document.getElementById('searchBtnSpinnerText');
const resetBtn           = document.getElementById('resetBtn');
const retryBtn           = document.getElementById('retryBtn');

const statusBar          = document.getElementById('statusBar');
const statusText         = document.getElementById('statusText');
const emptyState         = document.getElementById('emptyState');
const loadingState       = document.getElementById('loadingState');
const errorState         = document.getElementById('errorState');
const errorMessage       = document.getElementById('errorMessage');
const resultsGrid        = document.getElementById('resultsGrid');

// ── Initialise ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  updateSliderTrack();

  // Upload zone
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
  });

  // Drag & Drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImage();
  });

  // Formality sliders
  formalityMin.addEventListener('input', () => {
    if (parseFloat(formalityMin.value) > parseFloat(formalityMax.value)) {
      formalityMin.value = formalityMax.value;
    }
    formalityMinDisplay.textContent = parseFloat(formalityMin.value).toFixed(2);
    updateSliderTrack();
  });

  formalityMax.addEventListener('input', () => {
    if (parseFloat(formalityMax.value) < parseFloat(formalityMin.value)) {
      formalityMax.value = formalityMin.value;
    }
    formalityMaxDisplay.textContent = parseFloat(formalityMax.value).toFixed(2);
    updateSliderTrack();
  });

  // Viewpoint toggles
  viewpointToggle.querySelectorAll('.toggle-btn').forEach((btn) => {
    btn.addEventListener('click', () => selectViewpoint(btn.dataset.value, btn));
  });

  // Stepper
  topKDown.addEventListener('click', () => changeTopK(-1));
  topKUp.addEventListener('click',   () => changeTopK(+1));

  // Actions
  searchBtn.addEventListener('click', runSearch);
  resetBtn.addEventListener('click',  resetFilters);
  retryBtn.addEventListener('click',  runSearch);
});

// ── Health Check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById('healthDot');
  const text = document.getElementById('healthText');

  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();

    if (data.status === 'ok' && data.pinecone_connected) {
      dot.style.background = 'var(--success)';
      dot.style.boxShadow  = '0 0 6px var(--success)';
      text.textContent     = `${data.index_vector_count.toLocaleString()} items indexed`;
    } else {
      dot.style.background = 'var(--error)';
      dot.style.boxShadow  = '0 0 6px var(--error)';
      text.textContent     = 'backend error';
    }
  } catch {
    dot.style.background = 'var(--error)';
    dot.style.boxShadow  = '0 0 6px var(--error)';
    text.textContent     = 'offline';
  }
}

// ── File Handling ─────────────────────────────────────────────────────────────
function handleFileSelect(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file (JPG, PNG, or WEBP).');
    return;
  }

  currentFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    uploadPlaceholder.hidden = true;
    previewWrap.hidden       = false;
    searchBtn.disabled       = false;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentFile       = null;
  fileInput.value   = '';
  imagePreview.src  = '';
  previewWrap.hidden       = true;
  uploadPlaceholder.hidden = false;
  searchBtn.disabled       = true;
  showState('empty');
  statusBar.hidden = true;
}

// ── Slider Track ─────────────────────────────────────────────────────────────
function updateSliderTrack() {
  const min  = parseFloat(formalityMin.value);
  const max  = parseFloat(formalityMax.value);
  const pMin = min * 100;
  const pMax = max * 100;

  dualSlider.style.background = `linear-gradient(to right,
    var(--border)        ${pMin}%,
    var(--accent)        ${pMin}%,
    var(--accent)        ${pMax}%,
    var(--border)        ${pMax}%)`;
}

// ── Viewpoint Toggle ──────────────────────────────────────────────────────────
function selectViewpoint(value, clickedBtn) {
  viewpointToggle.querySelectorAll('.toggle-btn').forEach((b) => {
    b.classList.remove('active');
  });
  clickedBtn.classList.add('active');
  selectedViewpoint = value;
}

// ── Stepper ───────────────────────────────────────────────────────────────────
function changeTopK(delta) {
  topKValue = Math.max(1, Math.min(50, topKValue + delta));
  topKDisplay.textContent = topKValue;
}

// ── Build Filters ─────────────────────────────────────────────────────────────
function buildFilters() {
  const filters = { top_k: topKValue };

  const cat = categorySelect.value;
  if (cat) filters.category_name = cat;

  const season = seasonSelect.value;
  if (season) filters.season = season;

  const fMin = parseFloat(formalityMin.value);
  const fMax = parseFloat(formalityMax.value);
  if (fMin > 0)   filters.min_formality = fMin;
  if (fMax < 1)   filters.max_formality = fMax;

  if (selectedViewpoint !== '') {
    filters.viewpoint = parseInt(selectedViewpoint, 10);
  }

  return filters;
}

// ── Run Search ────────────────────────────────────────────────────────────────
async function runSearch() {
  if (!currentFile) return;

  showState('loading');
  setSearchLoading(true);

  const startTime = performance.now();

  try {
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('filters', JSON.stringify(buildFilters()));

    const res = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let detail = `Server error ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) detail = errJson.detail;
      } catch { /* ignore parse error */ }
      throw new Error(detail);
    }

    const data    = await res.json();
    const elapsed = (performance.now() - startTime).toFixed(0);
    renderResults(data, elapsed);

  } catch (err) {
    showError(err.message || 'An unknown error occurred.');
  } finally {
    setSearchLoading(false);
  }
}

function setSearchLoading(isLoading) {
  searchBtn.disabled              = isLoading || !currentFile;
  searchBtnText.hidden            = isLoading;
  searchBtnSpinner.hidden         = !isLoading;
  searchBtnSpinnerText.hidden     = !isLoading;
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data, elapsedMs) {
  statusBar.hidden = false;
  statusText.textContent =
    `${data.total} result${data.total !== 1 ? 's' : ''} · ${data.query_time_ms.toFixed(0)}ms server · ${elapsedMs}ms total`;

  if (data.results.length === 0) {
    showState('empty');
    emptyState.querySelector('.state-sub').textContent =
      'No matching items found. Try removing some filters.';
    return;
  }

  resultsGrid.innerHTML = '';
  showState('results');

  data.results.forEach((item, index) => {
    const card = buildCard(item, index);
    resultsGrid.appendChild(card);
  });
}

// ── Build Card ────────────────────────────────────────────────────────────────
function buildCard(item, index) {
  const hexColor      = labToHex(item.color_lab);
  const scorePercent  = (item.score * 100).toFixed(1) + '%';
  const formalityLbl  = formatFormality(item.formality);
  const imageSrc      = `${API_BASE}/image?path=${encodeURIComponent(item.image_path)}`;

  const card = document.createElement('div');
  card.className = 'result-card';
  card.style.animationDelay = `${index * 40}ms`;
  card.setAttribute('role', 'article');
  card.setAttribute('aria-label', `${item.category_name}, score ${scorePercent}`);

  card.innerHTML = `
    <div class="card-image-wrap">
      <img
        src="${imageSrc}"
        alt="${item.category_name}"
        class="card-image"
        loading="lazy"
        onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"
      >
      <div class="card-image-fallback">👗</div>
      <span class="score-badge">${scorePercent}</span>
      <span class="rank-badge">#${index + 1}</span>
    </div>
    <div class="card-info">
      <div class="card-category">${item.category_name}</div>
      <div class="card-meta">
        <span>${capitalize(item.season)}</span>
        <span>·</span>
        <span>${formalityLbl}</span>
      </div>
      <div class="card-color">
        <span class="color-dot" style="background:${hexColor}"></span>
        <span class="color-label">${hexColor}</span>
      </div>
    </div>
  `;

  return card;
}

// ── State Manager ─────────────────────────────────────────────────────────────
function showState(state) {
  emptyState.hidden   = state !== 'empty';
  loadingState.hidden = state !== 'loading';
  errorState.hidden   = state !== 'error';
  resultsGrid.hidden  = state !== 'results';
}

function showError(message) {
  showState('error');
  errorMessage.textContent = message;
}

// ── Reset Filters ─────────────────────────────────────────────────────────────
function resetFilters() {
  categorySelect.value   = '';
  seasonSelect.value     = '';

  formalityMin.value     = '0';
  formalityMax.value     = '1';
  formalityMinDisplay.textContent = '0.00';
  formalityMaxDisplay.textContent = '1.00';
  updateSliderTrack();

  selectedViewpoint = '';
  viewpointToggle.querySelectorAll('.toggle-btn').forEach((b) => {
    b.classList.remove('active');
  });
  document.getElementById('vpAny').classList.add('active');

  topKValue = 10;
  topKDisplay.textContent = '10';
}

// ── Color Conversion: LAB → Hex ───────────────────────────────────────────────
function labToHex([L, a, b]) {
  // Step 1: LAB → XYZ (illuminant D65)
  let fy = (L + 16) / 116;
  let fx = a / 500 + fy;
  let fz = fy - b / 200;

  const cube = (v) => v ** 3;
  const inv  = (v) => (cube(v) > 0.008856 ? cube(v) : (v - 16 / 116) / 7.787);

  let X = 0.95047 * inv(fx);
  let Y = 1.00000 * inv(fy);
  let Z = 1.08883 * inv(fz);

  // Step 2: XYZ → linear sRGB
  let r  =  X *  3.2406 + Y * -1.5372 + Z * -0.4986;
  let g  =  X * -0.9689 + Y *  1.8758 + Z *  0.0415;
  let bC =  X *  0.0557 + Y * -0.2040 + Z *  1.0570;

  // Step 3: gamma correction (sRGB)
  const gamma = (v) => (v > 0.0031308 ? 1.055 * Math.pow(Math.abs(v), 1 / 2.4) - 0.055 : 12.92 * v);

  r  = Math.round(Math.min(255, Math.max(0, gamma(r)  * 255)));
  g  = Math.round(Math.min(255, Math.max(0, gamma(g)  * 255)));
  bC = Math.round(Math.min(255, Math.max(0, gamma(bC) * 255)));

  if (isNaN(r) || isNaN(g) || isNaN(bC)) return '#888888';

  return '#' + [r, g, bC].map((v) => v.toString(16).padStart(2, '0')).join('');
}

// ── Formality Label ───────────────────────────────────────────────────────────
function formatFormality(score) {
  if (score < 0.2) return 'Very casual';
  if (score < 0.4) return 'Casual';
  if (score < 0.6) return 'Smart casual';
  if (score < 0.8) return 'Semi-formal';
  return 'Formal';
}

// ── Capitalize ────────────────────────────────────────────────────────────────
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}
