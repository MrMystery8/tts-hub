/* TTS Hub - Enhanced Application JavaScript */

const $ = (id) => document.getElementById(id);

// ============================================
// Session Persistence (localStorage + IndexedDB)
// ============================================
const SessionStore = {
  DB_NAME: 'tts-hub-session',
  STORE_NAME: 'audio-blobs',
  db: null,

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, 1);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => { this.db = request.result; resolve(); };
      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(this.STORE_NAME)) {
          db.createObjectStore(this.STORE_NAME);
        }
      };
    });
  },

  save(key, value) {
    try { localStorage.setItem(`tts-hub:${key}`, JSON.stringify(value)); }
    catch (e) { console.warn('localStorage save failed:', e); }
  },

  load(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(`tts-hub:${key}`);
      return item ? JSON.parse(item) : defaultValue;
    } catch { return defaultValue; }
  },

  async saveAudio(key, blob) {
    if (!this.db || !blob) return;
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORE_NAME, 'readwrite');
      const store = tx.objectStore(this.STORE_NAME);
      store.put(blob, key);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  },

  async loadAudio(key) {
    if (!this.db) return null;
    return new Promise((resolve) => {
      const tx = this.db.transaction(this.STORE_NAME, 'readonly');
      const store = tx.objectStore(this.STORE_NAME);
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => resolve(null);
    });
  },

  async clearAll() {
    localStorage.clear();
    if (this.db) {
      const tx = this.db.transaction(this.STORE_NAME, 'readwrite');
      tx.objectStore(this.STORE_NAME).clear();
    }
  }
};

// ============================================
// DOM Elements
// ============================================
const el = {
  // Sidebar
  modelNav: $('modelNav'),
  ffmpegDot: $('ffmpegDot'),
  ffmpegStatus: $('ffmpegStatus'),
  
  // Top bar
  selectedModelName: $('selectedModelName'),
  modelStatusBadges: $('modelStatusBadges'),
  clearSession: $('clearSession'),
  unloadModel: $('unloadModel'),
  
  // Voice
  promptFile: $('promptFile'),
  promptPreview: $('promptPreview'),
  promptInfo: $('promptInfo'),
  voicePreviewContainer: $('voicePreviewContainer'),
  clearVoice: $('clearVoice'),
  recToggle: $('recToggle'),
  recordingIndicator: $('recordingIndicator'),
  recordingTime: $('recordingTime'),
  waveformCanvas: $('waveformCanvas'),
  waveformTime: $('waveformTime'),
  transcriptSection: $('transcriptSection'),
  emotionSection: $('emotionSection'),
  promptText: $('promptText'),
  emoFile: $('emoFile'),
  
  // Text & Generate
  text: $('text'),
  outputFormat: $('outputFormat'),
  generate: $('generate'),
  generateText: $('generateText'),
  reset: $('reset'),
  statusBar: $('statusBar'),
  statusIcon: $('statusIcon'),
  statusText: $('statusText'),
  
  // Output
  output: $('output'),
  outputPlaceholder: $('outputPlaceholder'),
  outputActions: $('outputActions'),
  download: $('download'),
  outputInfo: $('outputInfo'),
  
  // Settings
  modelDescription: $('modelDescription'),
  noModelSelected: $('noModelSelected'),
  
  // History
  historyList: $('historyList'),
  historyEmpty: $('historyEmpty'),
  clearHistory: $('clearHistory'),

  // Model-specific (IndexTTS2)
  settingsIndex: $('settingsIndex'),
  indexEmoMode: $('indexEmoMode'),
  indexEmoAlpha: $('indexEmoAlpha'),
  indexEmoAlphaVal: $('indexEmoAlphaVal'),
  indexUseRandom: $('indexUseRandom'),
  indexEmoVectorRow: $('indexEmoVectorRow'),
  indexEmoVector: $('indexEmoVector'),
  indexEmoTextRow: $('indexEmoTextRow'),
  indexEmoText: $('indexEmoText'),
  indexMaxTextTokens: $('indexMaxTextTokens'),
  indexMaxMelTokens: $('indexMaxMelTokens'),
  indexFastMode: $('indexFastMode'),
  indexDoSample: $('indexDoSample'),
  indexTemperature: $('indexTemperature'),
  indexTemperatureVal: $('indexTemperatureVal'),
  indexTopP: $('indexTopP'),
  indexTopPVal: $('indexTopPVal'),
  indexTopK: $('indexTopK'),
  indexNumBeams: $('indexNumBeams'),
  indexRepetitionPenalty: $('indexRepetitionPenalty'),
  indexLengthPenalty: $('indexLengthPenalty'),

  // Chatterbox
  settingsChatterbox: $('settingsChatterbox'),
  cbLanguage: $('cbLanguage'),
  cbUsePrompt: $('cbUsePrompt'),
  cbCfgWeight: $('cbCfgWeight'),
  cbCfgWeightVal: $('cbCfgWeightVal'),
  cbTemperature: $('cbTemperature'),
  cbTemperatureVal: $('cbTemperatureVal'),
  cbExaggeration: $('cbExaggeration'),
  cbExaggerationVal: $('cbExaggerationVal'),
  cbFastMode: $('cbFastMode'),
  cbChunking: $('cbChunking'),
  cbMaxChunkChars: $('cbMaxChunkChars'),
  cbCrossfade: $('cbCrossfade'),
  cbEnableDf: $('cbEnableDf'),
  cbEnableNovasr: $('cbEnableNovasr'),

  // F5
  settingsF5: $('settingsF5'),
  f5RomanMode: $('f5RomanMode'),
  f5OverridesEnabled: $('f5OverridesEnabled'),
  f5OverridesText: $('f5OverridesText'),
  f5CrossFade: $('f5CrossFade'),
  f5NfeStep: $('f5NfeStep'),
  f5Speed: $('f5Speed'),
  f5RemoveSilence: $('f5RemoveSilence'),
  f5Seed: $('f5Seed'),

  // CosyVoice
  settingsCosy: $('settingsCosy'),
  cosyModel: $('cosyModel'),
  cosyMode: $('cosyMode'),
  cosyLang: $('cosyLang'),
  cosySpeed: $('cosySpeed'),
  cosyInstructRow: $('cosyInstructRow'),
  cosyInstructText: $('cosyInstructText'),

  // Pocket
  settingsPocket: $('settingsPocket'),
  pocketVoice: $('pocketVoice'),
  pocketTemp: $('pocketTemp'),
  pocketLsd: $('pocketLsd'),
  pocketEos: $('pocketEos'),
  pocketNoiseClamp: $('pocketNoiseClamp'),
  pocketTruncate: $('pocketTruncate'),

  // VoxCPM
  settingsVoxcpm: $('settingsVoxcpm'),
  voxcpmVoice: $('voxcpmVoice'),
  voxcpmCfg: $('voxcpmCfg'),
  voxcpmSteps: $('voxcpmSteps'),
  voxcpmMaxLen: $('voxcpmMaxLen'),
};

// ============================================
// State
// ============================================
let models = [];
let currentModelId = null;
let promptBlob = null;
let promptUploadedFile = null;
let isRecording = false;
let recordingStartTime = null;
let recordingTimer = null;

const recorder = {
  stream: null,
  audioContext: null,
  source: null,
  processor: null,
  buffers: [],
  sampleRate: 48000,
};

const history = [];

// Model icons
const MODEL_ICONS = {
  'index-tts2': '🎭',
  'chatterbox-multilingual': '🌍',
  'f5-hindi-urdu': '🇮🇳',
  'cosyvoice3-mlx': '🍎',
  'pocket-tts': '⚡',
  'voxcpm-ane': '🧠',
};

// ============================================
// Utilities
// ============================================
function setStatus(message, kind = 'info') {
  const icons = { info: 'ℹ️', error: '❌', ok: '✅', loading: '⏳' };
  el.statusIcon.textContent = icons[kind] || icons.info;
  el.statusText.textContent = message;
  el.statusBar.className = `status-bar ${kind === 'error' ? 'error' : kind === 'ok' ? 'success' : kind === 'loading' ? 'loading' : ''}`;
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes, unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) { size /= 1024; unitIndex++; }
  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// ============================================
// Collapsible Sections
// ============================================
document.querySelectorAll('.collapsible-header').forEach(header => {
  header.addEventListener('click', () => {
    const collapsible = header.closest('.collapsible');
    collapsible.classList.toggle('open');
  });
});

// ============================================
// Slider Bindings
// ============================================
function bindRange(rangeEl, valEl, formatter = x => String(x)) {
  if (!rangeEl || !valEl) return;
  const update = () => { valEl.textContent = formatter(parseFloat(rangeEl.value)); };
  rangeEl.addEventListener('input', update);
  update();
}

// ============================================
// Model Loading
// ============================================
async function fetchModels() {
  const res = await fetch('/api/models');
  if (!res.ok) throw new Error(`Failed to load models: HTTP ${res.status}`);
  const data = await res.json();
  models = data.models || [];
  renderModelNav();
}

function renderModelNav() {
  el.modelNav.innerHTML = '';
  models.forEach(m => {
    const card = document.createElement('div');
    card.className = `model-card${m.id === currentModelId ? ' active' : ''}`;
    card.dataset.modelId = m.id;
    card.innerHTML = `
      <div class="model-icon">${MODEL_ICONS[m.id] || '🔊'}</div>
      <div class="model-info">
        <div class="model-name">${m.name.split(' (')[0]}</div>
        <div class="model-desc">${m.description || ''}</div>
        <div class="model-badges">
          <span class="badge badge-neutral" id="badge-${m.id}">Ready</span>
        </div>
      </div>
    `;
    card.addEventListener('click', () => selectModel(m.id));
    el.modelNav.appendChild(card);
  });
}

function selectModel(modelId) {
  currentModelId = modelId;
  SessionStore.save('selectedModel', modelId);
  
  // Update sidebar
  document.querySelectorAll('.model-card').forEach(card => {
    card.classList.toggle('active', card.dataset.modelId === modelId);
  });
  
  // Update header
  const model = models.find(m => m.id === modelId);
  if (model) {
    el.selectedModelName.textContent = model.name.split(' (')[0];
    el.modelDescription.textContent = model.description || 'Configure model options';
  }
  
  // Show/hide settings panels
  document.querySelectorAll('.model-settings').forEach(panel => {
    panel.classList.toggle('hidden', panel.dataset.model !== modelId);
  });
  el.noModelSelected.classList.toggle('hidden', !!modelId);
  
  updateVisibility();
}

// ============================================
// Visibility Updates
// ============================================
function updateVisibility() {
  const modelId = currentModelId;
  
  // Transcript section
  const needsTranscript = ['f5-hindi-urdu', 'cosyvoice3-mlx', 'voxcpm-ane'].includes(modelId);
  el.transcriptSection?.classList.toggle('hidden', !needsTranscript);
  
  // Emotion section (IndexTTS2)
  if (modelId !== 'index-tts2') {
    el.emotionSection?.classList.add('hidden');
  } else {
    const emoMode = el.indexEmoMode?.value || 'speaker';
    el.emotionSection?.classList.toggle('hidden', emoMode !== 'emo_ref');
    el.indexEmoVectorRow?.classList.toggle('hidden', emoMode !== 'emo_vector');
    el.indexEmoTextRow?.classList.toggle('hidden', emoMode !== 'emo_text');
  }
  
  // CosyVoice instruct
  if (modelId === 'cosyvoice3-mlx') {
    el.cosyInstructRow?.classList.toggle('hidden', el.cosyMode?.value !== 'instruct');
  }
}

// ============================================
// Voice Recording
// ============================================
function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function encodeWav(buffers, sampleRate) {
  let length = 0;
  for (const b of buffers) length += b.length;
  const samples = new Float32Array(length);
  let offset = 0;
  for (const b of buffers) { samples.set(b, offset); offset += b.length; }

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPCM(view, 44, samples);

  return new Blob([buffer], { type: 'audio/wav' });
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Microphone access not supported');
  }
  recorder.buffers = [];
  recorder.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  recorder.sampleRate = recorder.audioContext.sampleRate;
  recorder.source = recorder.audioContext.createMediaStreamSource(recorder.stream);
  recorder.processor = recorder.audioContext.createScriptProcessor(4096, 1, 1);
  recorder.processor.onaudioprocess = (e) => {
    recorder.buffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  recorder.source.connect(recorder.processor);
  recorder.processor.connect(recorder.audioContext.destination);
  
  isRecording = true;
  recordingStartTime = Date.now();
  el.recToggle.classList.add('recording');
  el.recToggle.textContent = '■';
  el.recordingIndicator.classList.remove('hidden');
  
  recordingTimer = setInterval(() => {
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    el.recordingTime.textContent = formatTime(elapsed);
  }, 100);
}

async function stopRecording() {
  if (!recorder.stream) return null;
  
  clearInterval(recordingTimer);
  recorder.stream.getTracks().forEach(t => t.stop());
  recorder.processor?.disconnect();
  recorder.source?.disconnect();
  if (recorder.audioContext) await recorder.audioContext.close();
  
  const blob = encodeWav(recorder.buffers, recorder.sampleRate);
  recorder.buffers = [];
  recorder.audioContext = null;
  recorder.processor = null;
  recorder.source = null;
  recorder.stream = null;
  
  isRecording = false;
  el.recToggle.classList.remove('recording');
  el.recToggle.textContent = '●';
  el.recordingIndicator.classList.add('hidden');
  
  return blob;
}

// ============================================
// Voice Preview
// ============================================
function setVoicePreview(blob, label) {
  promptBlob = blob;
  promptUploadedFile = null;
  const url = URL.createObjectURL(blob);
  el.promptPreview.src = url;
  el.promptInfo.textContent = `${label} • WAV • ${formatBytes(blob.size)}`;
  el.voicePreviewContainer.classList.remove('hidden');
  el.clearVoice.disabled = false;
  SessionStore.saveAudio('promptAudio', blob);
  drawWaveform(blob);
}

function setVoicePreviewFromFile(file) {
  promptUploadedFile = file;
  promptBlob = null;
  const url = URL.createObjectURL(file);
  el.promptPreview.src = url;
  el.promptInfo.textContent = `${file.name} • ${file.type || 'audio'} • ${formatBytes(file.size)}`;
  el.voicePreviewContainer.classList.remove('hidden');
  el.clearVoice.disabled = false;
  
  file.arrayBuffer().then(buf => {
    const blob = new Blob([buf], { type: file.type });
    SessionStore.saveAudio('promptAudio', blob);
    drawWaveform(blob);
  });
}

function clearVoice() {
  promptBlob = null;
  promptUploadedFile = null;
  el.promptFile.value = '';
  el.promptPreview.removeAttribute('src');
  el.promptPreview.load();
  el.promptInfo.textContent = '';
  el.voicePreviewContainer.classList.add('hidden');
  el.clearVoice.disabled = true;
  SessionStore.saveAudio('promptAudio', null);
}

async function drawWaveform(blob) {
  try {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const data = audioBuffer.getChannelData(0);
    
    const canvas = el.waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.offsetWidth * 2;
    const height = canvas.offsetHeight * 2;
    canvas.width = width;
    canvas.height = height;
    
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(99, 102, 241, 0.6)';
    
    const step = Math.ceil(data.length / width);
    const amp = height / 2;
    for (let i = 0; i < width; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[(i * step) + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.fillRect(i, (1 + min) * amp, 1, Math.max(1, (max - min) * amp));
    }
    
    el.waveformTime.textContent = formatTime(audioBuffer.duration);
    audioContext.close();
  } catch (e) {
    console.warn('Waveform draw failed:', e);
  }
}

// ============================================
// Generation
// ============================================
async function generate() {
  const modelId = currentModelId;
  const text = el.text.value.trim();
  
  if (!modelId) { setStatus('Please select a model.', 'error'); return; }
  if (!text) { setStatus('Please enter some text.', 'error'); return; }
  
  const prompt = promptBlob || promptUploadedFile;
  const promptText = el.promptText?.value.trim() || '';
  
  // Validation
  if (['index-tts2', 'f5-hindi-urdu', 'cosyvoice3-mlx'].includes(modelId) && !prompt) {
    setStatus('This model requires a reference audio.', 'error'); return;
  }
  if (modelId === 'cosyvoice3-mlx' && el.cosyMode.value === 'zero_shot' && !promptText) {
    setStatus('CosyVoice3 zero_shot requires a reference transcript.', 'error'); return;
  }
  if (modelId === 'cosyvoice3-mlx' && el.cosyMode.value === 'instruct' && !el.cosyInstructText.value.trim()) {
    setStatus('CosyVoice3 instruct mode requires instruction text.', 'error'); return;
  }
  if (modelId === 'voxcpm-ane') {
    const cachedVoice = el.voxcpmVoice.value.trim();
    if (!cachedVoice && !prompt) { setStatus('VoxCPM-ANE requires a voice.', 'error'); return; }
    if (!cachedVoice && !promptText) { setStatus('VoxCPM-ANE requires a transcript.', 'error'); return; }
  }

  const form = new FormData();
  form.append('model_id', modelId);
  form.append('text', text);
  form.append('output_format', el.outputFormat.value);
  if (promptText) form.append('prompt_text', promptText);
  if (promptBlob) form.append('prompt_audio', promptBlob, 'prompt.wav');
  else if (promptUploadedFile) form.append('prompt_audio', promptUploadedFile);

  // Model-specific params
  appendModelParams(form, modelId);

  setGenerating(true);
  setStatus(`Generating with ${modelId}...`, 'loading');
  el.outputPlaceholder.classList.remove('hidden');
  el.output.classList.add('hidden');
  el.outputActions.classList.add('hidden');

  try {
    const res = await fetch('/api/generate', { method: 'POST', body: form });
    if (!res.ok) {
      let details = '';
      try { const data = await res.json(); details = data.error || JSON.stringify(data); }
      catch { details = await res.text(); }
      throw new Error(details || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    el.output.src = url;
    el.output.classList.remove('hidden');
    el.outputPlaceholder.classList.add('hidden');
    el.download.href = url;
    el.download.download = `${modelId}.${el.outputFormat.value}`;
    el.outputActions.classList.remove('hidden');
    
    addToHistory({ modelId, timestamp: Date.now(), url, format: el.outputFormat.value });
    setStatus('Generation complete!', 'ok');
  } catch (e) {
    setStatus(e.message || String(e), 'error');
  } finally {
    setGenerating(false);
  }
}

function appendModelParams(form, modelId) {
  if (modelId === 'index-tts2') {
    form.append('emo_mode', el.indexEmoMode.value);
    form.append('emo_alpha', el.indexEmoAlpha.value);
    form.append('use_random', el.indexUseRandom.checked);
    form.append('max_text_tokens_per_segment', el.indexMaxTextTokens.value);
    form.append('max_mel_tokens', el.indexMaxMelTokens.value);
    form.append('fast_mode', el.indexFastMode.checked);
    if (el.indexEmoMode.value === 'emo_vector') form.append('emo_vector', el.indexEmoVector.value.trim());
    if (el.indexEmoMode.value === 'emo_text') form.append('emo_text', el.indexEmoText.value.trim());
    form.append('do_sample', el.indexDoSample.checked);
    form.append('temperature', el.indexTemperature.value);
    form.append('top_p', el.indexTopP.value);
    form.append('top_k', el.indexTopK.value);
    form.append('num_beams', el.indexNumBeams.value);
    form.append('repetition_penalty', el.indexRepetitionPenalty.value);
    form.append('length_penalty', el.indexLengthPenalty.value);
    if (el.indexEmoMode.value === 'emo_ref') {
      const emo = el.emoFile.files?.[0];
      if (emo) form.append('emo_audio', emo);
    }
  }
  if (modelId === 'chatterbox-multilingual') {
    form.append('language_id', el.cbLanguage.value);
    form.append('use_prompt_audio', el.cbUsePrompt.checked);
    form.append('cfg_weight', el.cbCfgWeight.value);
    form.append('temperature', el.cbTemperature.value);
    form.append('exaggeration', el.cbExaggeration.value);
    form.append('fast_mode', el.cbFastMode.checked);
    form.append('enable_chunking', el.cbChunking.checked);
    form.append('max_chunk_chars', el.cbMaxChunkChars.value);
    form.append('crossfade_ms', el.cbCrossfade.value);
    form.append('enable_df', el.cbEnableDf.checked);
    form.append('enable_novasr', el.cbEnableNovasr.checked);
  }
  if (modelId === 'f5-hindi-urdu') {
    form.append('roman_mode', el.f5RomanMode.checked);
    form.append('overrides_enabled', el.f5OverridesEnabled.checked);
    form.append('overrides_text', el.f5OverridesText.value || '');
    form.append('cross_fade_duration', el.f5CrossFade.value);
    form.append('nfe_step', el.f5NfeStep.value);
    form.append('speed', el.f5Speed.value);
    form.append('remove_silence', el.f5RemoveSilence.checked);
    form.append('seed', el.f5Seed.value);
  }
  if (modelId === 'cosyvoice3-mlx') {
    form.append('cosy_model', el.cosyModel.value);
    form.append('mode', el.cosyMode.value);
    form.append('language', el.cosyLang.value.trim() || 'auto');
    form.append('speed', el.cosySpeed.value);
    if (el.cosyMode.value === 'instruct') form.append('instruct_text', el.cosyInstructText.value.trim());
  }
  if (modelId === 'pocket-tts') {
    form.append('voice', el.pocketVoice.value.trim());
    form.append('temperature', el.pocketTemp.value);
    form.append('lsd_decode_steps', el.pocketLsd.value);
    form.append('eos_threshold', el.pocketEos.value);
    if (el.pocketNoiseClamp.value.trim()) form.append('noise_clamp', el.pocketNoiseClamp.value);
    form.append('truncate_prompt', el.pocketTruncate.checked);
  }
  if (modelId === 'voxcpm-ane') {
    if (el.voxcpmVoice.value.trim()) form.append('voice', el.voxcpmVoice.value.trim());
    form.append('cfg_value', el.voxcpmCfg.value);
    form.append('inference_timesteps', el.voxcpmSteps.value);
    form.append('max_length', el.voxcpmMaxLen.value);
  }
}

function setGenerating(isGenerating) {
  el.generate.disabled = isGenerating;
  el.unloadModel.disabled = isGenerating;
  el.generateText.textContent = isGenerating ? '⏳ Generating...' : '✨ Generate';
}

// ============================================
// History
// ============================================
function addToHistory(item) {
  history.unshift(item);
  if (history.length > 20) history.pop();
  renderHistory();
  SessionStore.save('history', history.map(h => ({ ...h, url: null })));
}

function renderHistory() {
  if (history.length === 0) {
    el.historyEmpty.classList.remove('hidden');
    return;
  }
  el.historyEmpty.classList.add('hidden');
  el.historyList.innerHTML = history.map((h, i) => `
    <div class="history-item">
      <div class="history-icon">${MODEL_ICONS[h.modelId] || '🔊'}</div>
      <div class="history-info">
        <div class="history-title">${h.modelId}</div>
        <div class="history-meta">${new Date(h.timestamp).toLocaleTimeString()} • ${h.format.toUpperCase()}</div>
      </div>
      <div class="history-actions">
        ${h.url ? `<a class="btn btn-secondary" href="${h.url}" download="${h.modelId}.${h.format}">⬇️</a>` : ''}
      </div>
    </div>
  `).join('');
}

// ============================================
// Unload & Reset
// ============================================
async function unloadModel() {
  if (!currentModelId) return;
  const form = new FormData();
  form.append('model_id', currentModelId);
  setStatus(`Unloading ${currentModelId}...`, 'loading');
  try {
    const res = await fetch('/api/unload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    setStatus(`Unloaded ${currentModelId}`, 'ok');
    const badge = document.getElementById(`badge-${currentModelId}`);
    if (badge) { badge.textContent = 'Ready'; badge.className = 'badge badge-neutral'; }
  } catch (e) {
    setStatus(e.message || String(e), 'error');
  }
}

function resetAll() {
  clearVoice();
  el.promptText.value = '';
  el.emoFile.value = '';
  el.text.value = '';
  el.outputFormat.value = 'wav';
  el.output.classList.add('hidden');
  el.outputPlaceholder.classList.remove('hidden');
  el.outputActions.classList.add('hidden');
  setStatus('Reset complete.', 'ok');
}

// ============================================
// System Info
// ============================================
async function fetchMeta() {
  try {
    const res = await fetch('/api/info');
    if (!res.ok) return;
    const data = await res.json();
    const ok = data.ffmpeg?.available;
    el.ffmpegDot.className = `status-dot ${ok ? '' : 'error'}`;
    el.ffmpegStatus.textContent = ok ? 'ffmpeg: OK' : 'ffmpeg: Missing';
  } catch { }
}

// ============================================
// Session Restore
// ============================================
async function restoreSession() {
  // Restore model
  const savedModel = SessionStore.load('selectedModel');
  if (savedModel && models.find(m => m.id === savedModel)) {
    selectModel(savedModel);
  } else if (models.length > 0) {
    selectModel(models[0].id);
  }

  // Restore text
  const savedText = SessionStore.load('text');
  if (savedText) el.text.value = savedText;
  
  const savedPromptText = SessionStore.load('promptText');
  if (savedPromptText) el.promptText.value = savedPromptText;

  // Restore audio
  const savedAudio = await SessionStore.loadAudio('promptAudio');
  if (savedAudio) {
    setVoicePreview(savedAudio, 'Restored');
  }
}

// ============================================
// Event Listeners
// ============================================
el.promptFile.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) setVoicePreviewFromFile(file);
});

el.recToggle.addEventListener('click', async () => {
  if (isRecording) {
    const blob = await stopRecording();
    if (blob) setVoicePreview(blob, 'Recorded');
    setStatus('Recording saved.', 'ok');
  } else {
    try {
      await startRecording();
      setStatus('Recording...', 'loading');
    } catch (e) {
      setStatus(e.message, 'error');
    }
  }
});

el.clearVoice.addEventListener('click', clearVoice);
el.generate.addEventListener('click', generate);
el.reset.addEventListener('click', resetAll);
el.unloadModel.addEventListener('click', unloadModel);
el.clearHistory.addEventListener('click', () => { history.length = 0; renderHistory(); });
el.clearSession.addEventListener('click', async () => {
  await SessionStore.clearAll();
  resetAll();
  setStatus('Session cleared.', 'ok');
});

el.text.addEventListener('input', () => SessionStore.save('text', el.text.value));
el.promptText.addEventListener('input', () => SessionStore.save('promptText', el.promptText.value));

el.indexEmoMode?.addEventListener('change', updateVisibility);
el.cosyMode?.addEventListener('change', updateVisibility);

// Slider bindings
bindRange(el.indexEmoAlpha, el.indexEmoAlphaVal, x => x.toFixed(2));
bindRange(el.indexTemperature, el.indexTemperatureVal, x => x.toFixed(1));
bindRange(el.indexTopP, el.indexTopPVal, x => x.toFixed(2));
bindRange(el.cbCfgWeight, el.cbCfgWeightVal, x => x.toFixed(2));
bindRange(el.cbTemperature, el.cbTemperatureVal, x => x.toFixed(1));
bindRange(el.cbExaggeration, el.cbExaggerationVal, x => x.toFixed(2));

// ============================================
// Initialization
// ============================================
window.addEventListener('load', async () => {
  try {
    await SessionStore.init();
    await fetchModels();
    await fetchMeta();
    await restoreSession();
    renderHistory();
    updateVisibility();
    setStatus('Ready.');
  } catch (e) {
    setStatus(e.message || String(e), 'error');
  }
});
