/* TTS Hub mobile client wired to the FastAPI backend. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const app = $("app");

  const ACTIVE_STATES = ["queued", "preparing", "generating", "watermarking", "converting"];
  const NEEDS_REF = new Set(["index-tts2", "f5-hindi-urdu", "cosyvoice3-mlx", "qwen3-tts-mlx"]);
  const NEEDS_TRANSCRIPT = new Set(["f5-hindi-urdu", "voxcpm-ane"]);
  const WM_SUPPORTED = new Set(["index-tts2", "qwen3-tts-mlx", "chatterbox-multilingual"]);
  const SHOWS_REF = (id) => NEEDS_REF.has(id) || id === "voxcpm-ane" || id === "chatterbox-multilingual" || id === "pocket-tts";
  const FALLBACK_SETTINGS = {
    index: { emoMode: "speaker", emoAlpha: 0.65, useRandom: false, emoVector: "[0,0,0,0,0,0,0.45,0]", emoText: "", maxTextTokens: 120, maxMelTokens: 1500, fastMode: false, doSample: true, temperature: 0.8, topP: 0.8, topK: 30, numBeams: 3, repetitionPenalty: 10, lengthPenalty: 0 },
    chatterbox: { language: "hi", usePrompt: true, cfgWeight: 0.5, temperature: 0.8, exaggeration: 0.5, fastMode: false, enableChunking: true, maxChunkChars: 150, crossfadeMs: 50, enableDf: false, enableNovasr: false },
    f5: { romanMode: true, overridesEnabled: true, overridesText: "", crossFade: 0.15, nfeStep: 32, speed: 1, removeSilence: false, seed: -1 },
    cosy: { model: "8bit", mode: "zero_shot", language: "auto", speed: 1, instructText: "" },
    qwen: { model: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", autoTranscribe: true, language: "auto", speed: 1, temperature: 0.7, maxTokens: 1200 },
    pocket: { voice: "hf://kyutai/tts-voices/alba-mackenna/casual.wav", temperature: 0.8, lsdDecodeSteps: 8, eosThreshold: 0.4, noiseClamp: "", truncatePrompt: false },
    voxcpm: { voice: "", cfgValue: 2, inferenceTimesteps: 10, maxLength: 2048 },
  };
  const MODEL_GROUP = { "index-tts2": "index", "chatterbox-multilingual": "chatterbox", "f5-hindi-urdu": "f5", "cosyvoice3-mlx": "cosy", "qwen3-tts-mlx": "qwen", "pocket-tts": "pocket", "voxcpm-ane": "voxcpm" };
  const clone = (x) => JSON.parse(JSON.stringify(x));

  const state = {
    surface: "generate",
    sheet: null,
    models: [],
    statusMap: {},
    voices: [],
    jobs: [],
    selectedModelId: null,
    selectedVoiceId: "",
    text: "",
    outputFormat: "wav",
    watermarkEnabled: false,
    watermarkRun: null,
    settings: clone(FALLBACK_SETTINGS),
    refMode: "saved",
    promptText: "",
    instructText: "",
    refFile: null,
    refFileName: "",
    refPreviewUrl: "",
    refRecording: false,
    emoFile: null,
    emoFileName: "",
    emoPreviewUrl: "",
    settingsOpen: { basics: true, advanced: false },
    activeJobId: null,
    outputJobId: null,
    playingVoiceId: null,
    jobFilter: "all",
    jobDetailId: null,
    confirmAction: null,
    addName: "",
    addText: "",
    addFile: null,       // File or Blob staged for the new voice
    addFileName: "",
    addRecording: false,
    editVoiceId: null,
    editOriginal: null,
    editName: "",
    editText: "",
    editFile: null,
    editFileName: "",
    editPreviewUrl: "",
    editRecording: false,
    verifyMode: "upload",
    verifyFile: null,
    verifyFileName: "",
    verifyPreviewUrl: "",
    verifyRecording: false,
    verifyRuns: [],
    verifyDefaultRunId: "",
    verifyRunId: "",
    verifyThreshold: 0.35,
    verifyAdvancedOpen: false,
    verifyLoading: false,
    verifyResult: null,
    verifyError: "",
    connected: false,
  };

  /* ================= helpers ================= */

  async function api(path, options) {
    const res = await fetch(path, options);
    const isJson = (res.headers.get("content-type") || "").includes("json");
    const body = isJson ? await res.json() : null;
    if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
    return body;
  }

  let toastTimer = null;
  function flash(msg, kind = "success") {
    const el = $("toast");
    el.className = `toast ${kind}`;
    $("toast-msg").textContent = msg;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 3000);
  }

  function fmtAgo(epochS) {
    if (!epochS) return "—";
    const d = Math.max(0, Math.floor(Date.now() / 1000 - epochS));
    if (d < 60) return d + "s ago";
    if (d < 3600) return Math.floor(d / 60) + "m ago";
    if (d < 86400) return Math.floor(d / 3600) + "h ago";
    return Math.floor(d / 86400) + "d ago";
  }
  function fmtDur(ms) { return ms == null ? "—" : (ms / 1000).toFixed(1) + "s"; }
  function fmtSec(s) {
    if (s == null) return "—";
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m + ":" + String(sec).padStart(2, "0");
  }
  function hash(str) {
    let h = 0; const s = String(str || "x");
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    return h;
  }
  function peaks(key, n) {
    const seed = hash(key), out = [];
    for (let i = 0; i < n; i++) {
      const v = Math.abs(Math.sin(i * 0.55 + seed * 0.013) * Math.cos(i * 0.17 + seed * 0.0007));
      const env = 0.55 + 0.45 * Math.sin((i / n) * Math.PI);
      out.push(Math.max(0.1, v * env));
    }
    return out;
  }
  const waveformCache = new Map();
  const waveformLoading = new Map();
  async function audioPeaks(url, n) {
    const cacheKey = `${url}:${n}`;
    if (waveformCache.has(cacheKey)) return waveformCache.get(cacheKey);
    if (waveformLoading.has(cacheKey)) return waveformLoading.get(cacheKey);
    const loading = (async () => {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.arrayBuffer();
      const context = new (window.AudioContext || window.webkitAudioContext)();
      try {
        const decoded = await context.decodeAudioData(data.slice(0));
        const channel = decoded.getChannelData(0);
        const step = Math.max(1, Math.floor(channel.length / n));
        const result = Array.from({ length: n }, (_, i) => {
          const start = i * step;
          const end = Math.min(channel.length, start + step);
          let sum = 0;
          for (let j = start; j < end; j++) sum += channel[j] * channel[j];
          return Math.max(0.06, Math.sqrt(sum / Math.max(1, end - start)));
        });
        const max = Math.max(...result, 0.01);
        const normalized = result.map((value) => value / max);
        waveformCache.set(cacheKey, normalized);
        return normalized;
      } finally {
        context.close().catch(() => {});
      }
    })();
    waveformLoading.set(cacheKey, loading);
    try {
      return await loading;
    } finally {
      waveformLoading.delete(cacheKey);
    }
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  const modelOf = (id) => state.models.find((m) => m.id === id);
  const modelName = (id) => (modelOf(id) || { name: id || "—" }).name;
  const compatVoices = () =>
    state.voices.filter((v) => !Array.isArray(v.compatible_models) || !v.compatible_models.length || v.compatible_models.includes(state.selectedModelId));

  function jobReq(j) {
    const raw = j.request || {};
    const settings = raw.settings || {};
    return {
      schemaVersion: raw.schemaVersion || 1,
      source: raw.source || "desktop",
      model_id: j.model_id || raw.model_id || raw.modelId || null,
      voice_id: j.voice_id || raw.voice_id || raw.voiceId || null,
      text: j.text != null ? j.text : (raw.text || ""),
      prompt_text: raw.prompt_text != null ? raw.prompt_text : (raw.promptText || ""),
      output_format: j.output_format || raw.output_format || raw.outputFormat || (j.output || {}).format || "wav",
      watermark_enabled: j.watermark_enabled != null ? !!j.watermark_enabled : !!(raw.watermark_enabled || raw.watermarkEnabled),
      watermark_run: j.watermark_run || raw.watermark_run || raw.watermarkRun || null,
      settings,
    };
  }

  const selectedVoice = () => state.voices.find((v) => v.id === state.selectedVoiceId);
  function needsTranscript(id) {
    if (id === "f5-hindi-urdu" || id === "voxcpm-ane") return true;
    if (id === "cosyvoice3-mlx") return ["zero_shot", "instruct"].includes(state.settings.cosy.mode);
    if (id === "qwen3-tts-mlx") return !state.settings.qwen.autoTranscribe;
    return false;
  }
  function refReady() {
    if (state.refMode === "saved") return !!selectedVoice() && compatVoices().some((v) => v.id === state.selectedVoiceId);
    if (state.refMode === "upload" || state.refMode === "record") return !!state.refFile;
    return false;
  }
  function transcriptReady(id) {
    if (["upload", "record"].includes(state.refMode) && state.promptText.trim()) return true;
    if (state.refMode === "saved" && selectedVoice() && selectedVoice().has_transcript) return true;
    if (id === "voxcpm-ane" && state.settings.voxcpm.voice.trim()) return true;
    return false;
  }

  function statusPill(status) {
    const map = {
      queued: ["var(--tx-2)", "var(--bg-3)", "Queued"],
      preparing: ["var(--blue)", "var(--blue-dim)", "Preparing"],
      generating: ["var(--accent)", "var(--accent-dim)", "Generating"],
      watermarking: ["var(--purple)", "var(--purple-dim)", "Watermarking"],
      converting: ["var(--warn)", "var(--warn-dim)", "Converting"],
      completed: ["var(--accent)", "var(--accent-dim)", "Completed"],
      failed: ["var(--err)", "var(--err-dim)", "Failed"],
      cancelled: ["var(--tx-3)", "var(--bg-3)", "Cancelled"],
    };
    return map[status] || map.queued;
  }

  // One job runs at a time (the service queue is serialized), so a second
  // submit would silently queue. Block it and say why.
  function runBlockReason() {
    if (state.jobs.some((j) => ACTIVE_STATES.includes(j.status))) return "Generating — hold on, this run is still going.";
    return validate();
  }

  function validate() {
    if (!state.text.trim()) return "Script text is required.";
    const id = state.selectedModelId;
    if (NEEDS_REF.has(id) && !refReady()) return "Choose, upload, or record a reference voice.";
    if (id === "voxcpm-ane" && !state.settings.voxcpm.voice.trim() && !refReady()) return "VoxCPM needs a cached voice or reference audio.";
    if (needsTranscript(id) && !transcriptReady(id)) {
      return state.refMode === "saved"
        ? "This saved voice needs a transcript. Edit it in Voices before generating."
        : "A reference transcript is required.";
    }
    if (id === "cosyvoice3-mlx" && state.settings.cosy.mode === "instruct" && !state.instructText.trim()) return "CosyVoice instruct mode needs instruction text.";
    if (id === "index-tts2" && state.settings.index.emoMode === "emo_ref" && !state.emoFile) return "Choose emotion reference audio.";
    if (id === "index-tts2" && state.settings.index.emoMode === "emo_vector") {
      try {
        const vals = JSON.parse(state.settings.index.emoVector);
        if (!Array.isArray(vals) || vals.length !== 8 || vals.some((v) => !Number.isFinite(Number(v)))) throw new Error();
      } catch { return "Emotion vector must contain exactly 8 numbers."; }
    }
    return null;
  }

  /* ================= audio (real playback) ================= */

  const voiceAudio = new Audio();
  voiceAudio.addEventListener("ended", () => { state.playingVoiceId = null; render(); renderSheet(true); });
  voiceAudio.addEventListener("error", () => {
    if (state.playingVoiceId) { state.playingVoiceId = null; flash("Could not play voice preview.", "error"); render(); renderSheet(true); }
  });

  function previewVoice(id) {
    if (state.playingVoiceId === id) {
      voiceAudio.pause();
      state.playingVoiceId = null;
    } else {
      voiceAudio.src = `/api/voices/${encodeURIComponent(id)}/audio`;
      voiceAudio.play().catch(() => {});
      state.playingVoiceId = id;
    }
    render();
    renderSheet(true);
  }

  const outAudio = new Audio();
  let outPlaying = false;
  outAudio.addEventListener("ended", () => { outPlaying = false; renderMiniplayer(); });
  outAudio.addEventListener("pause", () => { outPlaying = false; renderMiniplayer(); });
  outAudio.addEventListener("play", () => { outPlaying = true; renderMiniplayer(); });
  outAudio.addEventListener("timeupdate", renderMiniplayer);
  outAudio.addEventListener("loadedmetadata", renderMiniplayer);
  outAudio.addEventListener("error", () => {
    if (state.outputJobId) {
      outPlaying = false;
      state.outputJobId = null;
      outAudio.removeAttribute("src");
      flash("This run's audio is no longer available.", "error");
      renderMiniplayer();
    }
  });

  // ---- Favourites: starred runs double as the saved-phrase board ----
  // Replaying a starred run costs nothing because its audio is already on
  // disk, so these are the fast path for everyday utterances.
  function jobLabel(j) {
    const t = (j.label || "").trim();
    if (t) return t;
    const s = (j.text || "").trim();
    return s.length > 30 ? `${s.slice(0, 30).trimEnd()}…` : s || "Untitled run";
  }
  function favouriteJobs() {
    return state.jobs
      .filter((j) => j.favorite && j.status === "completed" && j.output)
      .sort((a, b) => (b.favorited_at || 0) - (a.favorited_at || 0));
  }
  async function patchJob(id, patch, okMsg) {
    try {
      const job = await api(`/api/generation-jobs/${encodeURIComponent(id)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
      });
      state.jobs = state.jobs.map((j) => (j.id === job.id ? job : j));
      if (okMsg) flash(okMsg, "success");
      render();
      // An open job sheet is not redrawn by render(); refresh it so the star
      // and title reflect what was just saved.
      if (state.sheet === "job") renderSheet(true);
    } catch (e) { flash(e.message || "Could not update this run.", "error"); }
  }
  async function toggleFavorite(id) {
    const j = state.jobs.find((x) => x.id === id);
    if (!j) return;
    if (j.favorite) { await patchJob(id, { favorite: false }, "Removed from quick phrases."); return; }
    const suggested = (j.label || "").trim() || (j.text || "").trim().slice(0, 40);
    const name = window.prompt("Name this quick phrase — it plays instantly, no waiting.", suggested);
    if (name === null) return;
    await patchJob(id, { favorite: true, label: name.trim() || null }, "Saved to quick phrases.");
  }
  async function renameJob(id) {
    const j = state.jobs.find((x) => x.id === id);
    if (!j) return;
    const name = window.prompt("Rename this run.", (j.label || "").trim() || (j.text || "").trim().slice(0, 40));
    if (name === null) return;
    await patchJob(id, { label: name.trim() || null }, name.trim() ? "Run renamed." : "Name cleared.");
  }

  function playOutput(jobId, autoplay) {
    const url = `/api/generation-jobs/${encodeURIComponent(jobId)}/audio`;
    if (state.outputJobId !== jobId || !outAudio.src.endsWith(url)) outAudio.src = url;
    state.outputJobId = jobId;
    if (autoplay) outAudio.play().catch(() => {});
    render();
  }
  function toggleOutput() {
    if (outPlaying) outAudio.pause();
    else outAudio.play().catch(() => {});
  }
  function seekOutput(clientX) {
    const wave = $("mp-wave");
    const rect = wave.getBoundingClientRect();
    const duration = outAudio.duration || Number(($("mp-duration").dataset || {}).seconds) || 0;
    if (!duration || !rect.width) return;
    outAudio.currentTime = Math.max(0, Math.min(duration, ((clientX - rect.left) / rect.width) * duration));
    renderMiniplayer();
  }
  function downloadOutput(jobId) {
    const job = state.jobs.find((j) => j.id === jobId);
    const a = document.createElement("a");
    a.href = `/api/generation-jobs/${encodeURIComponent(jobId)}/audio`;
    a.download = (job && job.output && job.output.filename) || "output";
    document.body.appendChild(a);
    a.click();
    a.remove();
    flash("Download started.", "success");
  }

  const localAudio = new Audio();
  function previewLocal(url) {
    if (!url) return;
    if (localAudio.src === url && !localAudio.paused) localAudio.pause();
    else {
      localAudio.src = url;
      localAudio.play().catch(() => flash("Could not play this recording.", "error"));
    }
  }
  function replaceObjectUrl(key, file) {
    if (state[key]) URL.revokeObjectURL(state[key]);
    state[key] = file ? URL.createObjectURL(file) : "";
  }
  function clearReference() {
    localAudio.pause();
    replaceObjectUrl("refPreviewUrl", null);
    state.refFile = null;
    state.refFileName = "";
  }
  function clearEmotion() {
    localAudio.pause();
    replaceObjectUrl("emoPreviewUrl", null);
    state.emoFile = null;
    state.emoFileName = "";
  }
  function clearEditAudio() {
    localAudio.pause();
    replaceObjectUrl("editPreviewUrl", null);
    state.editFile = null;
    state.editFileName = "";
  }
  function clearVerifyAudio() {
    localAudio.pause();
    replaceObjectUrl("verifyPreviewUrl", null);
    state.verifyFile = null;
    state.verifyFileName = "";
    state.verifyResult = null;
    state.verifyError = "";
  }
  function invalidateVerifyResult() {
    state.verifyResult = null;
    state.verifyError = "";
  }

  /* ================= data loading / polling ================= */

  async function loadModels() {
    const data = await api("/api/models");
    state.models = data.models || [];
    for (const model of state.models) {
      const group = MODEL_GROUP[model.id];
      if (group && model.defaults) state.settings[group] = { ...state.settings[group], ...model.defaults };
    }
    if (!state.selectedModelId && state.models.length) state.selectedModelId = state.models[0].id;
  }
  async function loadStatus() {
    try {
      const data = await api("/api/status");
      state.statusMap = data.models || {};
    } catch { /* status is cosmetic; ignore */ }
  }
  async function loadWatermarkRuns() {
    try {
      const data = await api("/api/watermark/runs");
      state.verifyRuns = data.runs || [];
      state.verifyDefaultRunId = data.default_run_id || "";
      if (state.verifyRunId && !state.verifyRuns.some((run) => run.id === state.verifyRunId)) state.verifyRunId = "";
    } catch {
      state.verifyRuns = [];
      state.verifyDefaultRunId = "";
    }
  }
  async function loadVoices() {
    const data = await api("/api/voices");
    state.voices = data.voices || [];
    if (!state.selectedVoiceId) {
      const cv = compatVoices();
      if (cv.length) state.selectedVoiceId = cv[0].id;
    }
  }

  let pollTimer = null;
  async function loadJobs() {
    let data;
    try {
      data = await api("/api/generation-jobs");
      state.connected = true;
    } catch {
      state.connected = false;
      schedulePoll();
      return;
    }
    state.jobs = data.jobs || [];
    if (!state.activeJobId) {
      const persistedActive = state.jobs.find((j) => ACTIVE_STATES.includes(j.status));
      if (persistedActive) state.activeJobId = persistedActive.id;
    }
    if (state.activeJobId) {
      const active = state.jobs.find((j) => j.id === state.activeJobId);
      if (active && !ACTIVE_STATES.includes(active.status)) {
        state.activeJobId = null;
        if (active.status === "completed") {
          flash("Generation complete.", "success");
          playOutput(active.id, true);
        } else if (active.status === "failed") {
          flash(active.error || "Generation failed.", "error");
        }
      }
      if (!active) state.activeJobId = null;
    }
    render();
    schedulePoll();
  }
  function schedulePoll() {
    clearTimeout(pollTimer);
    const anyActive = state.activeJobId || state.jobs.some((j) => ACTIVE_STATES.includes(j.status));
    pollTimer = setTimeout(loadJobs, anyActive ? 2000 : 8000);
  }

  /* ================= actions ================= */

  function appendModelFields(fd, id) {
    const s = state.settings;
    if (id === "index-tts2") {
      const x = s.index;
      [["emo_mode", x.emoMode], ["emo_alpha", x.emoAlpha], ["use_random", x.useRandom], ["emo_vector", x.emoVector], ["emo_text", x.emoText], ["max_text_tokens_per_segment", x.maxTextTokens], ["max_mel_tokens", x.maxMelTokens], ["fast_mode", x.fastMode], ["do_sample", x.doSample], ["temperature", x.temperature], ["top_p", x.topP], ["top_k", x.topK], ["num_beams", x.numBeams], ["repetition_penalty", x.repetitionPenalty], ["length_penalty", x.lengthPenalty]].forEach(([k, v]) => fd.append(k, String(v)));
    } else if (id === "chatterbox-multilingual") {
      const x = s.chatterbox;
      [["language_id", x.language], ["use_prompt_audio", x.usePrompt], ["cfg_weight", x.cfgWeight], ["temperature", x.temperature], ["exaggeration", x.exaggeration], ["fast_mode", x.fastMode], ["enable_chunking", x.enableChunking], ["max_chunk_chars", x.maxChunkChars], ["crossfade_ms", x.crossfadeMs], ["enable_df", x.enableDf], ["enable_novasr", x.enableNovasr]].forEach(([k, v]) => fd.append(k, String(v)));
    } else if (id === "f5-hindi-urdu") {
      const x = s.f5;
      [["roman_mode", x.romanMode], ["overrides_enabled", x.overridesEnabled], ["overrides_text", x.overridesText], ["cross_fade_duration", x.crossFade], ["nfe_step", x.nfeStep], ["speed", x.speed], ["remove_silence", x.removeSilence], ["seed", x.seed]].forEach(([k, v]) => fd.append(k, String(v)));
    } else if (id === "cosyvoice3-mlx") {
      const x = s.cosy;
      [["cosy_model", x.model], ["mode", x.mode], ["language", x.language], ["speed", x.speed], ["instruct_text", state.instructText || x.instructText]].forEach(([k, v]) => fd.append(k, String(v)));
    } else if (id === "qwen3-tts-mlx") {
      const x = s.qwen;
      [["qwen_model", x.model], ["auto_transcribe", x.autoTranscribe], ["qwen_language", x.language], ["qwen_speed", x.speed], ["qwen_temperature", x.temperature], ["qwen_max_tokens", x.maxTokens]].forEach(([k, v]) => fd.append(k, String(v)));
    } else if (id === "pocket-tts") {
      const x = s.pocket;
      [["voice", x.voice], ["temperature", x.temperature], ["lsd_decode_steps", x.lsdDecodeSteps], ["eos_threshold", x.eosThreshold], ["noise_clamp", x.noiseClamp], ["truncate_prompt", x.truncatePrompt]].forEach(([k, v]) => fd.append(k, String(v)));
    } else if (id === "voxcpm-ane") {
      const x = s.voxcpm;
      [["voice", x.voice], ["cfg_value", x.cfgValue], ["inference_timesteps", x.inferenceTimesteps], ["max_length", x.maxLength]].forEach(([k, v]) => fd.append(k, String(v)));
    }
  }

  async function runGenerate() {
    const reason = runBlockReason();
    if (reason) { flash(reason, "error"); return; }
    const id = state.selectedModelId;
    const useVoice = state.refMode === "saved" && state.selectedVoiceId && compatVoices().some((v) => v.id === state.selectedVoiceId);
    const wm = state.watermarkEnabled && WM_SUPPORTED.has(id);

    const fd = new FormData();
    fd.append("model_id", id);
    fd.append("text", state.text);
    if (useVoice) fd.append("voice_id", state.selectedVoiceId);
    if ((state.refMode === "upload" || state.refMode === "record") && state.refFile) fd.append("prompt_audio", state.refFile, state.refFileName || "reference.webm");
    const submittedPromptText = ["upload", "record"].includes(state.refMode) ? state.promptText.trim() : "";
    if (submittedPromptText) fd.append("prompt_text", submittedPromptText);
    if (id === "index-tts2" && state.settings.index.emoMode === "emo_ref" && state.emoFile) fd.append("emo_audio", state.emoFile, state.emoFileName || "emotion.wav");
    fd.append("output_format", state.outputFormat);
    if (wm) fd.append("watermark", "1");
    if (wm && state.watermarkRun) fd.append("watermark_run", state.watermarkRun);
    appendModelFields(fd, id);
    fd.append("request_snapshot", JSON.stringify({
      schemaVersion: 2,
      source: "mobile",
      modelId: id,
      voiceId: useVoice ? state.selectedVoiceId : null,
      text: state.text,
      promptText: submittedPromptText,
      outputFormat: state.outputFormat,
      watermarkEnabled: wm,
      watermarkRun: state.watermarkRun,
      settings: state.settings,
    }));

    const btn = $("run-btn");
    btn.disabled = true;
    try {
      const job = await api("/api/generation-jobs", { method: "POST", body: fd });
      state.activeJobId = job.id;
      outAudio.pause();
      render();
      loadJobs();
    } catch (e) {
      flash(e.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function cancelActiveJob() {
    const id = state.activeJobId;
    if (!id) return;
    try {
      await api(`/api/generation-jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      state.activeJobId = null;
      flash("Generation cancelled.", "warn");
      loadJobs();
    } catch (e) {
      flash(e.message, "error");
    }
  }

  function deleteVoice(v) {
    const starred = favouriteJobs().filter((j) => j.voice_id === v.id).length;
    const items = [
      "The reference recording stored on this device",
      "Its transcript, preprocessing metadata, and cached model embeddings",
    ];
    if (starred) items.push(`${starred} saved phrase${starred === 1 ? "" : "s"} will keep playing, but cannot be regenerated in this voice`);
    openConfirm({
      title: `Delete “${v.name || v.id}”?`,
      items,
      note: "Clips already generated with this voice stay in Jobs.",
      onConfirm: () => reallyDeleteVoice(v),
    });
  }

  async function reallyDeleteVoice(v) {
    try {
      await api(`/api/voices/${encodeURIComponent(v.id)}`, { method: "DELETE" });
      if (state.selectedVoiceId === v.id) state.selectedVoiceId = "";
      await loadVoices();
      flash("Voice deleted.", "success");
      render();
    } catch (e) {
      flash(e.message, "error");
    }
  }

  async function openEditVoice(v) {
    try {
      const meta = await api(`/api/voices/${encodeURIComponent(v.id)}`);
      state.editVoiceId = v.id;
      state.editOriginal = {
        name: meta.name || v.name || "",
        promptText: meta.prompt_text || "",
      };
      state.editName = state.editOriginal.name;
      state.editText = state.editOriginal.promptText;
      clearEditAudio();
      state.editRecording = false;
      openSheet("edit");
    } catch (e) {
      flash(e.message, "error");
    }
  }

  function editVoiceDirty() {
    if (!state.editOriginal) return false;
    return !!state.editFile ||
      state.editName !== state.editOriginal.name ||
      state.editText !== state.editOriginal.promptText;
  }

  async function saveEditedVoice() {
    if (!state.editVoiceId || !state.editName.trim()) return;
    const fd = new FormData();
    fd.append("name", state.editName.trim());
    fd.append("prompt_text", state.editText.trim());
    if (state.editFile) fd.append("prompt_audio", state.editFile, state.editFileName || "replacement.webm");
    try {
      await api(`/api/voices/${encodeURIComponent(state.editVoiceId)}`, { method: "PATCH", body: fd });
      const editedId = state.editVoiceId;
      state.editOriginal = null;
      closeSheet(true);
      await loadVoices();
      state.selectedVoiceId = state.selectedVoiceId || editedId;
      flash("Voice updated.", "success");
      render();
    } catch (e) {
      flash(e.message, "error");
    }
  }

  function deleteJob(j) {
    const items = [
      "The generated audio file stored on this device",
      "Its submitted settings, metadata, and local history entry",
    ];
    if (j.favorite) items.push("Its shortcut in Quick Phrases");
    openConfirm({
      title: `Delete “${jobLabel(j)}”?`,
      items,
      note: "The saved reference voice, if any, is kept.",
      onConfirm: () => reallyDeleteJob(j),
    });
  }

  async function reallyDeleteJob(j) {
    try {
      await api(`/api/generation-jobs/${encodeURIComponent(j.id)}`, { method: "DELETE" });
      if (state.outputJobId === j.id) { outAudio.pause(); state.outputJobId = null; }
      closeSheet();
      flash("Run deleted.", "success");
      loadJobs();
    } catch (e) {
      flash(e.message, "error");
    }
  }

  async function saveVoice() {
    if (!state.addFile || !state.addName.trim()) return;
    const fd = new FormData();
    fd.append("name", state.addName.trim());
    fd.append("prompt_audio", state.addFile, state.addFile.name || "recording.webm");
    if (state.addText.trim()) fd.append("prompt_text", state.addText.trim());
    try {
      await api("/api/voices", { method: "POST", body: fd });
      closeSheet();
      await loadVoices();
      flash("Voice saved to library.", "success");
      render();
    } catch (e) {
      flash(e.message, "error");
    }
  }

  async function saveReferenceVoice() {
    if (!state.refFile) return;
    const name = window.prompt("Name this reference voice:");
    if (!name || !name.trim()) return;
    const fd = new FormData();
    fd.append("name", name.trim());
    fd.append("prompt_audio", state.refFile, state.refFileName || "reference.webm");
    if (state.promptText.trim()) fd.append("prompt_text", state.promptText.trim());
    try {
      const saved = await api("/api/voices", { method: "POST", body: fd });
      await loadVoices();
      state.selectedVoiceId = saved.id;
      state.refMode = "saved";
      state.promptText = "";
      clearReference();
      flash("Reference saved to the voice library.", "success");
      render();
    } catch (e) { flash(e.message, "error"); }
  }

  async function verifyWatermark() {
    if (state.verifyLoading || !state.verifyFile || !state.verifyRuns.length || !state.connected) return;
    const fd = new FormData();
    fd.append("audio", state.verifyFile, state.verifyFileName || "verification.webm");
    fd.append("wm_threshold", String(state.verifyThreshold));
    if (state.verifyRunId) fd.append("watermark_run", state.verifyRunId);
    state.verifyLoading = true;
    state.verifyResult = null;
    state.verifyError = "";
    renderVerify();
    try {
      state.verifyResult = await api("/api/watermark/detect", { method: "POST", body: fd });
      flash(state.verifyResult.detected ? "Watermark detected." : "No watermark detected.", state.verifyResult.detected ? "success" : "warn");
    } catch (e) {
      state.verifyError = e.message || "Watermark verification failed.";
    } finally {
      state.verifyLoading = false;
      renderVerify();
    }
  }

  /* --- recording (real MediaRecorder) --- */
  let recorder = null;
  let recorderStream = null;
  let recChunks = [];
  let recordingTarget = null;
  function discardRecording() {
    if (recorder && recorder.state === "recording") {
      recorder.ondataavailable = null;
      recorder.onstop = () => {
        if (recorderStream) recorderStream.getTracks().forEach((track) => track.stop());
        recorder = null;
        recorderStream = null;
        recordingTarget = null;
      };
      recorder.stop();
    } else if (recorderStream) {
      recorderStream.getTracks().forEach((track) => track.stop());
      recorderStream = null;
    }
    state.addRecording = false;
    state.refRecording = false;
    state.editRecording = false;
    state.verifyRecording = false;
  }
  async function toggleRecord(target = "add") {
    const recordingState = { add: "addRecording", ref: "refRecording", edit: "editRecording", verify: "verifyRecording" }[target] || "refRecording";
    const active = state[recordingState];
    if (active) {
      recorder && recorder.stop();
      return;
    }
    if (recorder && recorder.state === "recording") {
      flash("Stop the current recording first.", "warn");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorderStream = stream;
      recChunks = [];
      recorder = new MediaRecorder(stream);
      recordingTarget = target;
      recorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(recChunks, { type: recorder.mimeType || "audio/webm" });
        const ext = (recorder.mimeType || "").includes("mp4") ? "m4a" : "webm";
        const file = new File([blob], `recording_live.${ext}`, { type: blob.type });
        if (recordingTarget === "add") {
          state.addFile = file;
          state.addFileName = file.name;
          state.addRecording = false;
        } else if (recordingTarget === "edit") {
          clearEditAudio();
          state.editFile = file;
          state.editFileName = file.name;
          replaceObjectUrl("editPreviewUrl", file);
          state.editRecording = false;
        } else if (recordingTarget === "verify") {
          clearVerifyAudio();
          state.verifyFile = file;
          state.verifyFileName = file.name;
          replaceObjectUrl("verifyPreviewUrl", file);
          state.verifyRecording = false;
        } else {
          clearReference();
          state.refFile = file;
          state.refFileName = file.name;
          replaceObjectUrl("refPreviewUrl", file);
          state.refRecording = false;
        }
        recorder = null;
        recorderStream = null;
        recordingTarget = null;
        render();
        renderSheet(true);
      };
      recorder.start();
      if (target === "add") state.addRecording = true;
      else if (target === "edit") state.editRecording = true;
      else if (target === "verify") state.verifyRecording = true;
      else state.refRecording = true;
      render();
      renderSheet(true);
    } catch {
      flash("Microphone unavailable or permission denied.", "error");
    }
  }

  /* ================= render ================= */

  function render() {
    renderHeader();
    renderTabs();
    renderGenerate();
    renderVoicesPage();
    renderJobsPage();
    renderVerify();
    renderTransport();
    renderMiniplayer();
    renderSheet();
    renderConfirm();
  }

  function openConfirm(action) {
    state.confirmAction = action;
    renderConfirm();
  }

  function closeConfirm() {
    state.confirmAction = null;
    renderConfirm();
  }

  function renderConfirm() {
    const action = state.confirmAction;
    const backdrop = $("confirm-backdrop");
    backdrop.classList.toggle("hidden", !action);
    if (!action) return;
    $("confirm-title").textContent = action.title;
    $("confirm-note").textContent = action.note || "";
    $("confirm-note").classList.toggle("hidden", !action.note);
    const items = $("confirm-items");
    items.innerHTML = "";
    action.items.forEach((text) => {
      const row = el("div", "confirm-item");
      row.appendChild(el("span", "confirm-bullet", "•"));
      row.appendChild(el("span", null, text));
      items.appendChild(row);
    });
    $("confirm-delete").textContent = action.confirmLabel || "Delete permanently";
  }

  function renderHeader() {
    $("clock").textContent = state.connected
      ? new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : "offline";
    $("clock").style.color = state.connected ? "" : "var(--err)";
  }

  function renderTabs() {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === state.surface));
    ["generate", "voices", "jobs", "verify"].forEach((s) => $(`surface-${s}`).classList.toggle("hidden", s !== state.surface));

    const vb = $("badge-voices");
    vb.textContent = String(state.voices.length);
    vb.classList.toggle("hidden", !state.voices.length);
    const activeCount = state.jobs.filter((j) => ACTIVE_STATES.includes(j.status)).length;
    const jb = $("badge-jobs");
    jb.textContent = String(activeCount);
    jb.classList.toggle("hidden", !activeCount);
  }

  function renderGenerate() {
    const id = state.selectedModelId;
    const ms = state.statusMap[id] || {};
    const active = state.activeJobId ? state.jobs.find((j) => j.id === state.activeJobId && ACTIVE_STATES.includes(j.status)) : null;
    const runningCount = state.jobs.filter((j) => ACTIVE_STATES.includes(j.status)).length;
    $("generate-status").textContent = runningCount ? `${runningCount} running` : "";
    $("model-name").textContent = modelName(id);
    $("model-meta").textContent = `${ms.loaded ? "loaded" : "idle"} · ${ms.device || "—"} · ${ms.total_generations || 0} runs`;
    $("model-dot").style.background = ms.loaded ? "var(--accent)" : "var(--line-3)";

    // voice section
    const show = SHOWS_REF(id);
    $("voice-section").classList.toggle("hidden", !show);
    const refRequired = NEEDS_REF.has(id) || (id === "voxcpm-ane" && !state.settings.voxcpm.voice.trim());
    $("voice-required").classList.toggle("hidden", !refRequired);
    if (show) {
      const allowedModes = id === "chatterbox-multilingual" || id === "pocket-tts" ? ["saved", "upload", "record", "none"] : ["saved", "upload", "record"];
      if (!allowedModes.includes(state.refMode)) state.refMode = allowedModes[0];
      const tabs = $("ref-tabs");
      tabs.innerHTML = "";
      [["saved", "Saved"], ["upload", "Upload"], ["record", "Record"], ["none", "None"]].filter(([k]) => allowedModes.includes(k)).forEach(([k, label]) => {
        const b = el("button", "ref-tab" + (state.refMode === k ? " sel" : ""), label);
        b.addEventListener("click", () => { state.refMode = k; render(); });
        tabs.appendChild(b);
      });
      const locked = !!active;
      const lockedRef = $("ref-locked");
      lockedRef.classList.toggle("hidden", !locked);
      if (locked) {
        const submitted = jobReq(active);
        const voice = submitted.voice_id && state.voices.find((v) => v.id === submitted.voice_id);
        lockedRef.textContent = voice
          ? `Using saved voice: ${voice.name || voice.id}`
          : submitted.prompt_text ? "Using the submitted reference audio and transcript." : "Reference settings are locked for this run.";
      }
      tabs.classList.toggle("hidden", locked);
      $("ref-saved").classList.toggle("hidden", locked || state.refMode !== "saved");
      $("ref-file").classList.toggle("hidden", locked || !["upload", "record"].includes(state.refMode));
      const cv = compatVoices();
      const carousel = $("voice-carousel");
      carousel.innerHTML = "";
      $("voice-none").classList.toggle("hidden", cv.length > 0);
      carousel.classList.toggle("hidden", cv.length === 0);
      for (const vo of cv) {
        const sel = state.selectedVoiceId === vo.id;
        const playing = state.playingVoiceId === vo.id;
        const chip = el("div", "voice-chip" + (sel ? " sel" : ""));
        const pick = el("button", "vc-pick");
        pick.appendChild(el("span", "radio"));
        pick.appendChild(el("span", "vc-name", vo.name || vo.id));
        pick.addEventListener("click", () => { state.selectedVoiceId = vo.id; render(); });
        chip.appendChild(pick);
        const bottom = el("div", "vc-bottom");
        const pv = el("button", "preview-btn" + (playing ? " playing" : ""), playing ? "❚❚" : "▶");
        pv.addEventListener("click", () => previewVoice(vo.id));
        bottom.appendChild(pv);
        bottom.appendChild(el("span", "vc-dur", fmtSec(vo.duration_s)));
        if (vo.has_transcript) bottom.appendChild(el("span", "txt-badge", "TXT"));
        chip.appendChild(bottom);
        carousel.appendChild(chip);
      }

      const staged = !!state.refFile;
      $("choose-ref-file").classList.toggle("hidden", staged || state.refMode !== "upload");
      $("record-ref-file").classList.toggle("hidden", staged || state.refMode !== "record");
      $("record-ref-file").classList.toggle("recording", state.refRecording);
      $("record-ref-file").textContent = state.refRecording ? "■ Stop recording" : "● Record";
      $("ref-file-staged").classList.toggle("hidden", !staged);
      $("ref-file-name").textContent = state.refFileName;
      $("save-ref-voice").classList.toggle("hidden", !staged);

      const showTranscript = !locked && ["upload", "record"].includes(state.refMode) && (needsTranscript(id) || id === "qwen3-tts-mlx");
      $("prompt-text-wrap").classList.toggle("hidden", !showTranscript);
      $("prompt-text").value = state.promptText;
      $("prompt-text-hint").textContent = id === "qwen3-tts-mlx" && state.settings.qwen.autoTranscribe
          ? "Optional — Qwen will auto-transcribe when left blank."
          : needsTranscript(id) ? "Required for this model and mode." : "Optional.";
      $("instruct-text-wrap").classList.toggle("hidden", locked || !(id === "cosyvoice3-mlx" && state.settings.cosy.mode === "instruct"));
      $("instruct-text").value = state.instructText;
      const emotionRef = id === "index-tts2" && state.settings.index.emoMode === "emo_ref";
      $("emotion-ref-wrap").classList.toggle("hidden", locked || !emotionRef);
      $("emo-file-staged").classList.toggle("hidden", !state.emoFile);
      $("choose-emo-file").classList.toggle("hidden", !!state.emoFile);
      $("emo-file-name").textContent = state.emoFileName;
    }

    // script + counters
    const script = $("script");
    if (script.value !== state.text) script.value = state.text;
    const count = $("char-count");
    count.textContent = `${state.text.length} chars`;
    count.classList.toggle("err", state.text.length === 0);

    // quick phrases — starred runs replay stored audio, so there is no wait
    const favs = favouriteJobs();
    $("phrase-section").classList.toggle("hidden", favs.length === 0);
    const pcar = $("phrase-carousel");
    pcar.innerHTML = "";
    for (const j of favs) {
      const playing = state.outputJobId === j.id && outPlaying;
      const chip = el("button", "phrase-chip" + (playing ? " playing" : ""));
      chip.appendChild(el("span", "pc-icon", playing ? "❚❚" : "▶"));
      chip.appendChild(el("span", "pc-name", jobLabel(j)));
      chip.title = j.text || "";
      chip.setAttribute("aria-label", `${playing ? "Stop" : "Play"} quick phrase: ${jobLabel(j)}`);
      chip.addEventListener("click", () => {
        if (playing) outAudio.pause();
        else playOutput(j.id, true);
      });
      pcar.appendChild(chip);
    }

    // options summary
    const wmOn = state.watermarkEnabled && WM_SUPPORTED.has(id);
    const group = MODEL_GROUP[id];
    const x = state.settings[group] || {};
    const speed = x.speed != null ? `${Number(x.speed).toFixed(2)}× · ` : "";
    const temp = x.temperature != null ? `temp ${Number(x.temperature).toFixed(2)} · ` : "";
    $("options-summary").textContent = `${speed}${temp}${state.outputFormat.toUpperCase()}${wmOn ? " · WM" : ""}`;

    // run bar
    const reason = runBlockReason();
    const blocked = !!reason;
    $("block-reason").textContent = reason || "";
    $("block-reason").classList.toggle("hidden", !blocked);
    $("run-btn").classList.toggle("blocked", blocked);
    $("run-btn").setAttribute("aria-disabled", blocked ? "true" : "false");
    $("run-label").textContent = runningCount ? "Generating…" : blocked ? "Run" : "Run generation";
  }

  function renderTransport() {
    const active = state.activeJobId ? state.jobs.find((j) => j.id === state.activeJobId) : null;
    const running = !!active && ACTIVE_STATES.includes(active.status);
    $("transport-running").classList.toggle("hidden", !running || state.surface !== "generate");
    $("runbar").classList.toggle("hidden", running || state.surface !== "generate");
    if (!running) return;

    const labels = { queued: "Waiting in queue", preparing: "Preparing model", generating: "Synthesizing speech", watermarking: "Applying watermark", converting: "Converting output" };
    const details = {
      queued: "The run will begin when earlier jobs finish.",
      preparing: "Loading the model and preparing reference audio.",
      generating: "The model is producing audio. It does not report a reliable percentage.",
      watermarking: "Embedding the selected watermark into the generated audio.",
      converting: "Encoding the final download format.",
    };
    $("active-phase").textContent = labels[active.status] || "Working";
    const elapsedFrom = active.started_at || active.created_at || Date.now() / 1000;
    const elapsed = Math.max(0, Math.floor(Date.now() / 1000 - elapsedFrom));
    const queued = state.jobs
      .filter((job) => ACTIVE_STATES.includes(job.status))
      .sort((a, b) => Number(a.created_at || 0) - Number(b.created_at || 0));
    const queueIndex = queued.findIndex((job) => job.id === active.id);
    const queueText = active.status === "queued" && queueIndex >= 0 ? ` · position ${queueIndex + 1} of ${queued.length}` : "";
    $("active-meta").textContent = `${modelName(jobReq(active).model_id)} · ${elapsed}s elapsed${queueText}`;
    $("active-detail").textContent = details[active.status] || "";

    const wm = !!jobReq(active).watermark_enabled;
    const shown = ["queued", "preparing", "generating", "watermarking", "converting"].filter((p) => p !== "watermarking" || wm);
    const curIdx = shown.indexOf(active.status);
    const track = $("phase-track");
    track.innerHTML = "";
    shown.forEach((p, i) => {
      const seg = el("div", "phase-seg" + (i < curIdx ? " done" : i === curIdx ? " active" : ""));
      if (i === curIdx) seg.appendChild(el("span", "sweep"));
      track.appendChild(seg);
    });
  }

  function renderMiniplayer() {
    const out = state.outputJobId ? state.jobs.find((j) => j.id === state.outputJobId && j.status === "completed") : null;
    const playerSurface = state.surface === "generate" || state.surface === "jobs";
    const showMp = !!out && playerSurface && (state.surface === "jobs" || !state.activeJobId);
    $("miniplayer").classList.toggle("hidden", !showMp);
    if (!showMp) return;

    const o = out.output || {};
    const duration = Number(outAudio.duration || o.duration_s || 0);
    const current = Math.min(Number(outAudio.currentTime || 0), duration || Infinity);
    $("mp-name").textContent = o.filename || out.id;
    $("mp-meta").textContent = `${(o.format || "").toUpperCase()} · tap waveform to seek`;
    const favorite = $("mp-favorite");
    favorite.textContent = out.favorite ? "★" : "☆";
    favorite.title = out.favorite ? "Remove from Quick Phrases" : "Save to Quick Phrases";
    favorite.setAttribute("aria-label", out.favorite ? "Remove playing audio from quick phrases" : "Save playing audio as a quick phrase");
    favorite.setAttribute("aria-pressed", out.favorite ? "true" : "false");
    $("mp-current").textContent = fmtSec(current);
    $("mp-duration").textContent = fmtSec(duration);
    $("mp-duration").dataset.seconds = String(duration);
    $("mp-toggle").innerHTML = outPlaying
      ? '<svg width="13" height="13" viewBox="0 0 13 13"><rect x="2" y="1" width="3.5" height="11" rx="1" fill="currentColor"/><rect x="7.5" y="1" width="3.5" height="11" rx="1" fill="currentColor"/></svg>'
      : '<svg width="14" height="14" viewBox="0 0 14 14"><path d="M2 1.5 L12.5 7 L2 12.5 Z" fill="currentColor"/></svg>';

    const n = 30;
    const url = `/api/generation-jobs/${encodeURIComponent(out.id)}/audio`;
    const cacheKey = `${url}:${n}`;
    const pk = waveformCache.get(cacheKey) || peaks(out.id, n);
    const frac = duration ? Math.min(1, current / duration) : 0;
    const cut = Math.round(frac * n);
    const wave = $("mp-wave");
    wave.innerHTML = "";
    pk.forEach((p, i) => {
      const bar = el("span");
      bar.style.height = Math.round(15 + p * 85) + "%";
      bar.style.background = i < cut ? "var(--accent)" : "var(--tx-3)";
      bar.style.opacity = i < cut ? "1" : "0.32";
      wave.appendChild(bar);
    });
    if (!waveformCache.has(cacheKey) && !wave.dataset.loading) {
      wave.dataset.loading = "1";
      audioPeaks(url, n)
        .then(() => {
          delete wave.dataset.loading;
          if (state.outputJobId === out.id) renderMiniplayer();
        })
        .catch(() => { delete wave.dataset.loading; });
    }
  }

  function renderVoicesPage() {
    $("voice-count").textContent = `${state.voices.length} saved`;
    const list = $("voice-list");
    list.innerHTML = "";
    if (!state.voices.length) {
      const empty = el("div", "empty-state");
      empty.appendChild(el("div", "empty-title", "No voices yet"));
      empty.appendChild(el("div", "empty-sub", "Add a reference voice to reuse it across models."));
      list.appendChild(empty);
      return;
    }
    for (const vo of state.voices) {
      const playing = state.playingVoiceId === vo.id;
      const card = el("div", "voice-card" + (playing ? " playing" : ""));

      const top = el("div", "vcard-top");
      const play = el("button", "vcard-play" + (playing ? " playing" : ""), playing ? "❚❚" : "▶");
      play.addEventListener("click", () => previewVoice(vo.id));
      top.appendChild(play);

      const mid = el("div", "vcard-mid");
      const nameRow = el("div", "vcard-name-row");
      nameRow.appendChild(el("span", "vcard-name", vo.name || vo.id));
      if (vo.has_transcript) nameRow.appendChild(el("span", "txt-badge", "TXT"));
      mid.appendChild(nameRow);
      const meta = el("div", "vcard-meta");
      meta.appendChild(el("span", null, fmtSec(vo.duration_s)));
      meta.appendChild(el("span", null, fmtAgo(vo.created_at)));
      mid.appendChild(meta);
      top.appendChild(mid);

      const use = el("button", "use-btn", "Use");
      use.addEventListener("click", () => {
        state.selectedVoiceId = vo.id;
        if (Array.isArray(vo.compatible_models) && vo.compatible_models.length && !vo.compatible_models.includes(state.selectedModelId)) {
          state.selectedModelId = vo.compatible_models[0];
        }
        state.surface = "generate";
        flash("Voice loaded into Generate.", "success");
        render();
      });
      top.appendChild(use);

      const edit = el("button", "edit-btn", "Edit");
      edit.addEventListener("click", () => openEditVoice(vo));
      top.appendChild(edit);

      const del = el("button", "del-btn");
      del.title = "Delete voice";
      del.innerHTML = '<svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><path d="M2 3.5h10M5.5 3.5V2h3v1.5M3.5 3.5l.6 8.5h5.8l.6-8.5"/></svg>';
      del.addEventListener("click", () => deleteVoice(vo));
      top.appendChild(del);
      card.appendChild(top);

      const wave = el("div", "vcard-wave");
      peaks(vo.id, 26).forEach((p) => {
        const bar = el("span");
        bar.style.height = Math.round(15 + p * 85) + "%";
        wave.appendChild(bar);
      });
      card.appendChild(wave);

      const chips = el("div", "vcard-chips");
      const cms = Array.isArray(vo.compatible_models) ? vo.compatible_models : [];
      cms.slice(0, 4).forEach((mid2) => chips.appendChild(el("span", "model-chip", modelName(mid2).split(" ")[0])));
      if (cms.length > 4) chips.appendChild(el("span", "model-chip", "+" + (cms.length - 4)));
      card.appendChild(chips);
      list.appendChild(card);
    }
  }

  function renderJobsPage() {
    // filters
    const filters = $("job-filters");
    filters.innerHTML = "";
    [["all", "All"], ["favorite", "★ Saved"], ["active", "Active"], ["completed", "Done"], ["failed", "Failed"]].forEach(([k, l]) => {
      const b = el("button", "filter-btn" + (state.jobFilter === k ? " sel" : ""), l);
      b.addEventListener("click", () => { state.jobFilter = k; renderJobsPage(); });
      filters.appendChild(b);
    });

    let jobs = state.jobs;
    if (state.jobFilter === "favorite") jobs = jobs.filter((j) => j.favorite);
    else if (state.jobFilter === "active") jobs = jobs.filter((j) => ACTIVE_STATES.includes(j.status));
    else if (state.jobFilter === "completed") jobs = jobs.filter((j) => j.status === "completed");
    else if (state.jobFilter === "failed") jobs = jobs.filter((j) => j.status === "failed" || j.status === "cancelled");

    const list = $("job-list");
    list.innerHTML = "";
    if (!jobs.length) {
      list.appendChild(el("div", "no-jobs", state.jobFilter === "favorite"
        ? "Nothing saved yet. Open a finished run and tap Save phrase."
        : "No jobs match this filter."));
      return;
    }
    for (const j of jobs) {
      const req = jobReq(j);
      const [col, bg, label] = statusPill(j.status);
      const card = el("button", "job-card");
      card.addEventListener("click", () => { state.jobDetailId = j.id; openSheet("job"); });

      const top = el("div", "job-top");
      const pill = el("span", "status-pill");
      pill.style.color = col;
      pill.style.background = bg;
      if (ACTIVE_STATES.includes(j.status)) pill.appendChild(el("span", "live-dot"));
      pill.appendChild(document.createTextNode(label));
      top.appendChild(pill);
      if (j.favorite) top.appendChild(el("span", "pc-icon", "★"));
      top.appendChild(el("span", "job-model", modelName(req.model_id)));
      top.appendChild(el("span", "job-ago", fmtAgo(j.created_at)));
      card.appendChild(top);

      if (j.label) {
        card.appendChild(el("div", "job-text named", j.label));
        card.appendChild(el("div", "job-script", req.text || "(no script recorded)"));
      } else {
        card.appendChild(el("div", "job-text", req.text || "(no script recorded)"));
      }

      const facts = el("div", "job-facts");
      facts.appendChild(el("span", null, (req.output_format || (j.output || {}).format || "?").toUpperCase()));
      facts.appendChild(el("span", null, fmtDur(j.worker_duration_ms)));
      card.appendChild(facts);
      list.appendChild(card);
    }
  }

  function renderVerify() {
    const hasRuns = state.verifyRuns.length > 0;
    const selectedRun = state.verifyRuns.find((run) => run.id === state.verifyRunId);
    const defaultRun = state.verifyRuns.find((run) => run.id === state.verifyDefaultRunId) || state.verifyRuns[0];
    const status = $("verify-run-status");
    status.textContent = hasRuns ? "Detector ready" : "Detector unavailable";
    status.classList.toggle("unavailable", !hasRuns);

    document.querySelectorAll("[data-verify-mode]").forEach((button) => {
      button.classList.toggle("sel", button.dataset.verifyMode === state.verifyMode);
    });
    const staged = !!state.verifyFile;
    $("choose-verify-file").classList.toggle("hidden", staged || state.verifyMode !== "upload");
    $("record-verify-file").classList.toggle("hidden", staged || state.verifyMode !== "record");
    $("record-verify-file").classList.toggle("recording", state.verifyRecording);
    $("record-verify-file").textContent = state.verifyRecording ? "■ Stop recording" : "● Start recording";
    $("verify-file-staged").classList.toggle("hidden", !staged);
    $("verify-file-name").textContent = state.verifyFileName;

    const advanced = $("verify-advanced-body");
    advanced.classList.toggle("hidden", !state.verifyAdvancedOpen);
    $("verify-advanced-toggle").setAttribute("aria-expanded", String(state.verifyAdvancedOpen));
    $("verify-advanced-caret").textContent = state.verifyAdvancedOpen ? "▴" : "▾";
    const runSummary = selectedRun ? (selectedRun.label || selectedRun.id) : "Automatic";
    $("verify-advanced-summary").textContent = `${runSummary} · ${state.verifyThreshold.toFixed(2)}`;

    const runSelect = $("verify-run-select");
    runSelect.innerHTML = "";
    const automatic = document.createElement("option");
    automatic.value = "";
    automatic.textContent = defaultRun ? `Automatic (${defaultRun.label || defaultRun.id})` : "Automatic";
    runSelect.appendChild(automatic);
    state.verifyRuns.forEach((run) => {
      const option = document.createElement("option");
      option.value = run.id;
      option.textContent = `${run.label || run.id}${run.status ? ` · ${run.status}` : ""}`;
      runSelect.appendChild(option);
    });
    runSelect.value = state.verifyRunId;
    runSelect.disabled = !hasRuns || state.verifyLoading;
    $("verify-threshold").value = String(state.verifyThreshold);
    $("verify-threshold").disabled = state.verifyLoading;
    $("verify-threshold-value").textContent = state.verifyThreshold.toFixed(2);
    $("verify-reset").disabled = state.verifyLoading;

    let reason = "";
    if (!state.connected) reason = "The hub is offline. Reconnect to verify audio.";
    else if (!hasRuns) reason = "No trained detector is available. Create or select a watermark run from the desktop app.";
    else if (!state.verifyFile) reason = "Choose or record an audio clip first.";
    const blocked = !!reason || state.verifyLoading;
    $("verify-block-reason").textContent = reason;
    $("verify-block-reason").classList.toggle("hidden", !reason);
    $("verify-btn").disabled = blocked;
    $("verify-btn").classList.toggle("blocked", !!reason);
    $("verify-btn").classList.toggle("loading", state.verifyLoading);
    $("verify-btn-label").textContent = state.verifyLoading ? "Checking audio…" : "Check watermark";

    const error = $("verify-error");
    error.textContent = state.verifyError;
    error.classList.toggle("hidden", !state.verifyError);

    const result = $("verify-result");
    result.innerHTML = "";
    result.classList.toggle("hidden", !state.verifyResult);
    result.classList.remove("detected", "clean");
    if (!state.verifyResult) return;
    const detected = !!state.verifyResult.detected;
    const probability = Math.max(0, Math.min(1, Number(state.verifyResult.wm_prob || 0)));
    result.classList.add(detected ? "detected" : "clean");
    const top = el("div", "verify-result-top");
    top.appendChild(el("span", "verify-result-icon", detected ? "✓" : "—"));
    const title = el("div");
    title.appendChild(el("div", "verify-result-title", detected ? "Watermark detected" : "No watermark detected"));
    title.appendChild(el("div", "verify-subtitle", detected ? "Embedded provenance evidence was found." : "The score did not meet the selected threshold."));
    top.appendChild(title);
    top.appendChild(el("span", "verify-result-confidence", `${(probability * 100).toFixed(1)}%`));
    result.appendChild(top);
    const model = state.verifyResult.model || {};
    const resultRunId = (state.verifyResult.run || {}).id || state.verifyRunId || state.verifyDefaultRunId || "automatic";
    const resultRun = state.verifyRuns.find((run) => run.id === resultRunId);
    const facts = el("div", "verify-result-facts");
    [
      ["Raw score", probability.toFixed(3)],
      ["Source model", model.name || model.tts_model_id || "Source model unavailable"],
      ["Detector run", (resultRun && (resultRun.label || resultRun.id)) || resultRunId],
      ["Threshold", state.verifyThreshold.toFixed(2)],
    ].forEach(([key, value]) => {
      facts.appendChild(el("span", "k", key));
      facts.appendChild(el("span", "v", value));
    });
    result.appendChild(facts);
  }

  /* ================= sheets ================= */

  function openSheet(name) {
    state.sheet = name;
    render();
  }
  function closeSheet(force = false) {
    if (!force && state.sheet === "edit" && editVoiceDirty()) {
      if (!window.confirm("Discard unsaved voice changes?")) return;
    }
    state.sheet = null;
    state.jobDetailId = null;
    if ((state.addRecording || state.editRecording) && recorder) discardRecording();
    clearEditAudio();
    state.editVoiceId = null;
    state.editOriginal = null;
    render();
  }

  /* Rebuild the sheet body only when the sheet changes (or on demand), so background
     job polling never wipes in-progress text input or an active slider drag. */
  let renderedSheet = null;
  function renderSheet(force = false) {
    const open = !!state.sheet;
    $("sheet-backdrop").classList.toggle("hidden", !open);
    if (!open) { renderedSheet = null; return; }
    if (!force && renderedSheet === state.sheet) return;
    renderedSheet = state.sheet;
    const detailJob = state.jobDetailId ? state.jobs.find((job) => job.id === state.jobDetailId) : null;
    $("sheet-title").textContent = state.sheet === "job" && detailJob
      ? jobLabel(detailJob)
      : ({ model: "Choose model", voice: "Choose voice", settings: "Options", add: "New reference voice", edit: "Edit voice" }[state.sheet] || "");
    const body = $("sheet-body");
    body.innerHTML = "";
    if (state.sheet === "model") renderModelSheet(body);
    else if (state.sheet === "voice") renderVoiceSheet(body);
    else if (state.sheet === "settings") renderSettingsSheet(body);
    else if (state.sheet === "job") renderJobSheet(body);
    else if (state.sheet === "add") renderAddSheet(body);
    else if (state.sheet === "edit") renderEditVoiceSheet(body);
  }

  function renderModelSheet(body) {
    for (const m of state.models) {
      const s2 = state.statusMap[m.id] || {};
      const isSel = m.id === state.selectedModelId;
      const btn = el("button", "model-opt" + (isSel ? " sel" : ""));
      const top = el("div", "mo-top");
      const dot = el("span", "dot");
      dot.style.background = s2.loaded ? "var(--accent)" : "var(--line-3)";
      top.appendChild(dot);
      top.appendChild(el("span", "mo-name", m.name || m.id));
      top.appendChild(el("span", "mo-meta", `${s2.loaded ? "loaded" : "idle"} · ${s2.device || "—"}`));
      btn.appendChild(top);
      const chips = el("div", "mo-chips");
      if (NEEDS_REF.has(m.id)) chips.appendChild(el("span", "cap-chip ref", "ref"));
      if (NEEDS_TRANSCRIPT.has(m.id)) chips.appendChild(el("span", "cap-chip transcript", "transcript"));
      if (WM_SUPPORTED.has(m.id)) chips.appendChild(el("span", "cap-chip watermark", "watermark"));
      if (chips.children.length) btn.appendChild(chips);
      btn.addEventListener("click", () => {
        state.selectedModelId = m.id;
        const cv = compatVoices();
        if (!cv.some((x) => x.id === state.selectedVoiceId)) state.selectedVoiceId = cv.length ? cv[0].id : "";
        state.sheet = null;
        render();
      });
      body.appendChild(btn);
    }
  }

  function renderVoiceSheet(body) {
    const cv = compatVoices();
    if (!cv.length) {
      body.appendChild(el("div", "sheet-empty", "No compatible voices for this model."));
      return;
    }
    for (const vo of cv) {
      const sel = state.selectedVoiceId === vo.id;
      const playing = state.playingVoiceId === vo.id;
      const row = el("div", "voice-row" + (sel ? " sel" : ""));
      const pick = el("button", "vr-pick");
      pick.appendChild(el("span", "radio"));
      pick.appendChild(el("span", "vr-name", vo.name || vo.id));
      pick.appendChild(el("span", "vr-dur", fmtSec(vo.duration_s)));
      pick.addEventListener("click", () => { state.selectedVoiceId = vo.id; state.sheet = null; render(); });
      row.appendChild(pick);
      const pv = el("button", "vr-preview" + (playing ? " playing" : ""), playing ? "❚❚" : "▶");
      pv.addEventListener("click", () => previewVoice(vo.id));
      row.appendChild(pv);
      body.appendChild(row);
    }
  }

  function renderSettingsSheet(body) {
    const col = el("div", "settings-col");
    const id = state.selectedModelId;
    const group = MODEL_GROUP[id];
    const x = state.settings[group];
    const defs = (modelOf(id) || {}).defaults || FALLBACK_SETTINGS[group];

    const actions = el("div", "settings-actions");
    actions.appendChild(el("div", "settings-summary", `Settings for ${modelName(id)}. Changes apply to the next queued run.`));
    const reset = el("button", "reset-btn", "Reset");
    reset.addEventListener("click", () => {
      state.settings[group] = clone(defs);
      if (group === "cosy") state.instructText = state.settings.cosy.instructText || "";
      renderGenerate();
      renderSheet(true);
      flash("Model settings reset.", "success");
    });
    actions.appendChild(reset);
    col.appendChild(actions);

    const schemas = {
      index: {
        basics: [
          ["select", "Emotion mode", "emoMode", [["speaker", "Speaker"], ["emo_ref", "Reference audio"], ["emo_vector", "Vector"], ["emo_text", "Emotion text"]]],
          ["slider", "Emotion strength", "emoAlpha", 0, 1, 0.05, "Scales the selected emotion without changing the speaker identity."],
          ["emotionVector", "Emotion vector", "emoVector", null, null, null, "Blend the eight canonical IndexTTS2 emotions.", () => x.emoMode === "emo_vector"],
          ["textarea", "Emotion text", "emoText", null, null, null, "Describe the intended delivery in natural language.", () => x.emoMode === "emo_text"],
        ],
        advanced: [
          ["toggle", "Sampling", "doSample", null, null, null, "Stochastic sampling adds variation; disable it for greedy decoding."],
          ["slider", "Temperature", "temperature", 0.1, 1.5, 0.05],
          ["slider", "Top P", "topP", 0.1, 1, 0.05],
          ["number", "Top K", "topK", 0, 200, 1],
          ["number", "Beams", "numBeams", 1, 10, 1],
          ["slider", "Repetition penalty", "repetitionPenalty", 1, 15, 0.5],
          ["slider", "Length penalty", "lengthPenalty", -2, 2, 0.1],
          ["toggle", "Random emotion", "useRandom"],
          ["toggle", "Fast mode", "fastMode", null, null, null, "Trades some quality for faster inference."],
          ["number", "Text tokens / segment", "maxTextTokens", 20, 500, 1],
          ["number", "Maximum mel tokens", "maxMelTokens", 100, 4000, 50],
        ],
      },
      chatterbox: {
        basics: [
          ["select", "Language", "language", [["hi", "Hindi"], ["en", "English"], ["es", "Spanish"], ["fr", "French"], ["de", "German"], ["zh", "Chinese"]]],
          ["toggle", "Use reference voice", "usePrompt", null, null, null, "Use the selected or uploaded prompt audio for voice cloning."],
          ["slider", "Exaggeration", "exaggeration", 0, 1, 0.05, "Raises or softens expressive delivery."],
          ["slider", "CFG weight", "cfgWeight", 0, 1, 0.05, "Controls adherence to the reference voice."],
          ["slider", "Temperature", "temperature", 0.1, 1.5, 0.05],
          ["toggle", "Fast mode", "fastMode"],
        ],
        advanced: [
          ["toggle", "Long-form chunking", "enableChunking"],
          ["number", "Maximum chunk characters", "maxChunkChars", 30, 1000, 1, "Maximum text length per generated segment.", () => x.enableChunking],
          ["slider", "Crossfade (ms)", "crossfadeMs", 0, 500, 5, "Blends adjacent long-form chunks.", () => x.enableChunking],
          ["toggle", "DeepFilter denoise", "enableDf"],
          ["toggle", "NovaSR upscale", "enableNovasr"],
        ],
      },
      f5: {
        basics: [
          ["toggle", "Roman → Devanagari", "romanMode"],
          ["toggle", "Apply pronunciation overrides", "overridesEnabled"],
          ["number", "Speed", "speed", 0.5, 2, 0.05],
          ["number", "NFE steps", "nfeStep", 4, 128, 1],
          ["toggle", "Remove silence", "removeSilence"],
        ],
        advanced: [
          ["number", "Crossfade duration", "crossFade", 0, 1, 0.01],
          ["number", "Seed (-1 = random)", "seed", -1, 2147483647, 1],
          ["textarea", "Pronunciation overrides", "overridesText"],
        ],
      },
      cosy: {
        basics: [
          ["select", "Model variant", "model", [["8bit", "8-bit"], ["4bit", "4-bit"], ["bf16", "BF16"]]],
          ["select", "Mode", "mode", [["zero_shot", "Zero-shot"], ["cross_lingual", "Cross-lingual"], ["instruct", "Instruct"]]],
          ["text", "Language", "language"],
          ["number", "Speed", "speed", 0.5, 2, 0.05],
        ],
        advanced: [],
      },
      qwen: {
        basics: [
          ["select", "Model", "model", [["mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", "1.7B 8-bit"], ["mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit", "0.6B 8-bit"]]],
          ["toggle", "Auto-transcribe reference", "autoTranscribe", null, null, null, "When enabled, Qwen derives the transcript from uploaded reference audio."],
          ["select", "Language", "language", [["auto", "Auto detect"], ["en", "English"], ["zh", "Chinese"], ["ja", "Japanese"], ["ko", "Korean"], ["de", "German"], ["fr", "French"], ["es", "Spanish"], ["ru", "Russian"]]],
          ["slider", "Speed", "speed", 0.5, 2, 0.05],
          ["slider", "Temperature", "temperature", 0.1, 1.5, 0.05],
        ],
        advanced: [["number", "Maximum tokens", "maxTokens", 100, 4000, 10, "Caps the generated acoustic token sequence."]],
      },
      pocket: {
        basics: [
          ["text", "Built-in / HF voice", "voice"],
          ["number", "Temperature", "temperature", 0.1, 2, 0.05],
          ["number", "Decode steps", "lsdDecodeSteps", 1, 64, 1],
          ["number", "EOS threshold", "eosThreshold", 0, 1, 0.05],
          ["toggle", "Truncate long prompt", "truncatePrompt"],
        ],
        advanced: [["text", "Noise clamp (blank = default)", "noiseClamp"]],
      },
      voxcpm: {
        basics: [
          ["text", "Cached voice name", "voice"],
          ["number", "CFG value", "cfgValue", 0, 10, 0.1],
          ["number", "Inference steps", "inferenceTimesteps", 1, 100, 1],
          ["number", "Maximum length", "maxLength", 128, 8192, 1],
        ],
        advanced: [],
      },
    };

    function emotionVectorRow(label, key, help) {
      const wrap = el("div", "emotion-vector");
      const top = el("div", "emotion-vector-head");
      top.appendChild(el("label", null, label));
      const resetVector = el("button", "emotion-reset", "Reset vector");
      resetVector.addEventListener("click", () => {
        x[key] = "[0,0,0,0,0,0,0,0]";
        renderGenerate();
        renderSheet(true);
      });
      top.appendChild(resetVector);
      wrap.appendChild(top);
      if (help) wrap.appendChild(el("div", "setting-help", help));
      let vector;
      try { vector = JSON.parse(x[key]); } catch { vector = []; }
      if (!Array.isArray(vector)) vector = [];
      vector = Array.from({ length: 8 }, (_, i) => Math.max(0, Math.min(1, Number(vector[i]) || 0)));
      const labels = [
        ["Happy", "cheerful, bright"], ["Angry", "tense, intense"], ["Sad", "sorrowful, downcast"],
        ["Afraid", "fearful, anxious"], ["Disgust", "repulsed, scornful"], ["Melancholic", "wistful, subdued"],
        ["Surprised", "startled, animated"], ["Calm", "neutral, steady"],
      ];
      labels.forEach(([name, tag], index) => {
        const item = el("div", "emotion-item");
        const head = el("div", "slider-head");
        const copy = el("div");
        copy.appendChild(el("div", "emotion-name", name));
        copy.appendChild(el("div", "emotion-tag", tag));
        head.appendChild(copy);
        head.appendChild(el("span", "slider-val", vector[index].toFixed(2)));
        item.appendChild(head);
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0"; slider.max = "1"; slider.step = "0.05"; slider.value = String(vector[index]);
        slider.addEventListener("input", () => {
          vector[index] = Number(slider.value);
          x[key] = JSON.stringify(vector.map((value) => Math.round(value * 100) / 100));
          head.querySelector(".slider-val").textContent = vector[index].toFixed(2);
          renderGenerate();
        });
        item.appendChild(slider);
        wrap.appendChild(item);
      });
      return wrap;
    }

    function settingRow(spec) {
      const [type, label, key, a, b, c, help, condition] = spec;
      if (typeof condition === "function" && !condition()) return null;
      if (type === "emotionVector") return emotionVectorRow(label, key, help);
      const row = el("div", "setting-row" + (type === "slider" || help ? " stacked" : ""));
      const labelNode = el("label", null, label);
      row.appendChild(labelNode);
      let control;
      if (type === "toggle") {
        control = el("button", "setting-toggle" + (x[key] ? " on" : ""));
        control.setAttribute("role", "switch");
        control.setAttribute("aria-checked", String(!!x[key]));
        control.addEventListener("click", () => { x[key] = !x[key]; renderGenerate(); renderSheet(true); });
      } else if (type === "select") {
        control = document.createElement("select");
        for (const [value, text] of a) {
          const opt = document.createElement("option"); opt.value = value; opt.textContent = text; control.appendChild(opt);
        }
        control.value = x[key];
        control.addEventListener("change", () => { x[key] = control.value; renderGenerate(); renderSheet(true); });
      } else if (type === "textarea") {
        control = document.createElement("textarea");
        control.rows = 3; control.value = x[key] == null ? "" : x[key];
        control.addEventListener("input", () => { x[key] = control.value; renderGenerate(); });
      } else if (type === "slider") {
        const head = el("div", "slider-head");
        head.appendChild(labelNode);
        const value = el("span", "slider-val", Number(x[key]).toFixed(c < 1 ? (c < 0.1 ? 2 : 1) : 0));
        head.appendChild(value);
        row.replaceChildren(head);
        control = document.createElement("input");
        control.type = "range";
        control.min = a; control.max = b; control.step = c; control.value = x[key];
        control.addEventListener("input", () => {
          x[key] = Number(control.value);
          value.textContent = Number(x[key]).toFixed(c < 1 ? (c < 0.1 ? 2 : 1) : 0);
          renderGenerate();
        });
      } else {
        control = document.createElement("input");
        control.type = type;
        control.value = x[key] == null ? "" : x[key];
        if (type === "number") { control.min = a; control.max = b; control.step = c; }
        control.addEventListener("input", () => { x[key] = type === "number" ? Number(control.value) : control.value; renderGenerate(); });
      }
      control.classList.add("setting-control");
      row.appendChild(control);
      if (help) row.appendChild(el("div", "setting-help", help));
      return row;
    }

    for (const bucket of ["basics", "advanced"]) {
      const specs = schemas[group][bucket];
      if (!specs.length) continue;
      const wrap = el("div", "setting-group");
      const head = el("button", "setting-group-head");
      head.appendChild(el("span", null, bucket === "basics" ? "Basics" : "Advanced"));
      head.appendChild(el("span", null, state.settingsOpen[bucket] ? "−" : "+"));
      head.addEventListener("click", () => { state.settingsOpen[bucket] = !state.settingsOpen[bucket]; renderSheet(true); });
      wrap.appendChild(head);
      if (state.settingsOpen[bucket]) {
        const section = el("div", "setting-group-body");
        specs.forEach((spec) => { const row = settingRow(spec); if (row) section.appendChild(row); });
        wrap.appendChild(section);
      }
      col.appendChild(wrap);
    }

    const fmtWrap = el("div");
    fmtWrap.appendChild(el("div", "fmt-label", "Format"));
    const tabs = el("div", "fmt-tabs");
    ["wav", "mp3", "flac"].forEach((f) => {
      const b = el("button", "fmt-tab" + (state.outputFormat === f ? " sel" : ""), f.toUpperCase());
      b.addEventListener("click", () => { state.outputFormat = f; renderGenerate(); renderSheet(true); });
      tabs.appendChild(b);
    });
    fmtWrap.appendChild(tabs);
    col.appendChild(fmtWrap);

    const wmOk = WM_SUPPORTED.has(state.selectedModelId);
    const wmOn = wmOk && state.watermarkEnabled;
    const wmRow = el("button", "wm-row" + (wmOk ? "" : " unsupported"));
    const track = el("span", "wm-track" + (wmOn ? " on" : ""));
    track.appendChild(el("span", "wm-knob"));
    wmRow.appendChild(track);
    wmRow.appendChild(el("span", "wm-label", `Watermark · ${!wmOk ? "Not supported" : wmOn ? "On" : "Off"}`));
    wmRow.addEventListener("click", () => {
      if (wmOk) { state.watermarkEnabled = !state.watermarkEnabled; renderGenerate(); renderSheet(true); }
      else flash(`Watermark not supported by ${modelName(state.selectedModelId)}.`, "warn");
    });
    col.appendChild(wmRow);
    body.appendChild(col);
  }

  function renderJobSheet(body) {
    const dj = state.jobDetailId ? state.jobs.find((j) => j.id === state.jobDetailId) : null;
    if (!dj) { closeSheet(); return; }
    const req = jobReq(dj);
    const col = el("div", "dj-col");

    if (dj.error) {
      const err = el("div", "dj-error");
      err.appendChild(el("b", null, "Error"));
      err.appendChild(document.createTextNode(dj.error));
      col.appendChild(err);
    }

    if (dj.status === "completed") {
      const actions = el("div", "dj-actions");
      const play = el("button", "dj-play", "▶ Play");
      play.addEventListener("click", () => {
        closeSheet();
        playOutput(dj.id, true);
      });
      const dl = el("button", "dj-dl", "↓ Download");
      dl.addEventListener("click", () => downloadOutput(dj.id));
      actions.appendChild(play);
      actions.appendChild(dl);
      col.appendChild(actions);

      const favActions = el("div", "dj-actions");
      const star = el("button", "dj-restore", dj.favorite ? "★ Saved" : "☆ Save phrase");
      star.setAttribute("aria-pressed", dj.favorite ? "true" : "false");
      star.addEventListener("click", () => toggleFavorite(dj.id));
      const ren = el("button", "dj-restore", "Rename");
      ren.addEventListener("click", () => renameJob(dj.id));
      favActions.appendChild(star);
      favActions.appendChild(ren);
      col.appendChild(favActions);
    }

    const voiceName = req.voice_id
      ? ((state.voices.find((x) => x.id === req.voice_id) || {}).name || req.voice_id)
      : "ad-hoc / none";
    const facts = el("div", "dj-facts");
    [
      ["Status", statusPill(dj.status)[2]],
      ["Model", modelName(req.model_id)],
      ["Voice", voiceName],
      ["Format", (req.output_format || (dj.output || {}).format || "?").toUpperCase()],
      ["Watermark", req.watermark_enabled ? "enabled" : "off"],
      ["Duration", fmtDur(dj.worker_duration_ms)],
    ].forEach(([k, v2]) => {
      const row = el("div", "dj-fact");
      row.appendChild(el("span", "k", k));
      row.appendChild(el("span", "v", v2));
      facts.appendChild(row);
    });
    col.appendChild(facts);

    const scriptWrap = el("div");
    scriptWrap.appendChild(el("div", "dj-script-label", "Script"));
    scriptWrap.appendChild(el("div", "dj-script", req.text || "(no script recorded)"));
    col.appendChild(scriptWrap);

    const actions2 = el("div", "dj-actions");
    const restore = el("button", "dj-restore", "Restore settings");
    restore.addEventListener("click", () => {
      if (req.model_id) state.selectedModelId = req.model_id;
      if (req.voice_id) state.selectedVoiceId = req.voice_id;
      if (req.text != null) state.text = req.text;
      if (req.prompt_text != null) state.promptText = req.prompt_text;
      if (req.output_format) state.outputFormat = req.output_format;
      state.watermarkEnabled = !!req.watermark_enabled;
      state.watermarkRun = req.watermark_run || null;
      if (req.settings && Object.keys(req.settings).length) {
        for (const [group, values] of Object.entries(req.settings)) {
          if (state.settings[group] && values && typeof values === "object") state.settings[group] = { ...state.settings[group], ...values };
        }
      }
      state.instructText = state.settings.cosy.instructText || state.instructText;
      state.refMode = req.voice_id ? "saved" : state.refMode;
      state.surface = "generate";
      closeSheet();
      flash("Run settings restored.", "success");
    });
    const delBtn = el("button", "dj-delete", ACTIVE_STATES.includes(dj.status) ? "Cancel run" : "Delete");
    delBtn.addEventListener("click", () => {
      if (ACTIVE_STATES.includes(dj.status)) {
        api(`/api/generation-jobs/${encodeURIComponent(dj.id)}/cancel`, { method: "POST" })
          .then(() => { closeSheet(); flash("Generation cancelled.", "warn"); loadJobs(); })
          .catch((e) => flash(e.message, "error"));
      } else {
        deleteJob(dj);
      }
    });
    actions2.appendChild(restore);
    actions2.appendChild(delBtn);
    col.appendChild(actions2);
    body.appendChild(col);
  }

  function renderAddSheet(body) {
    const col = el("div", "add-col");

    if (!state.addFile) {
      const row = el("div", "add-src-row");
      const up = el("button", "add-src", "＋ Upload .wav / .mp3");
      up.addEventListener("click", () => $("file-input").click());
      const rec = el("button", "add-src" + (state.addRecording ? " recording" : ""), state.addRecording ? "■ Stop" : "● Record");
      rec.addEventListener("click", () => toggleRecord("add"));
      row.appendChild(up);
      row.appendChild(rec);
      col.appendChild(row);
    } else {
      const staged = el("div", "staged-row");
      staged.appendChild(el("span", "staged-dot"));
      staged.appendChild(el("span", "staged-name", state.addFileName));
      const rep = el("button", "staged-replace", "Replace");
      rep.addEventListener("click", () => { state.addFile = null; state.addFileName = ""; renderSheet(true); });
      staged.appendChild(rep);
      col.appendChild(staged);
    }

    const name = document.createElement("input");
    name.className = "add-input";
    name.placeholder = "Voice name";
    name.value = state.addName;
    name.addEventListener("input", () => { state.addName = name.value; syncSave(); });
    col.appendChild(name);

    const text = document.createElement("textarea");
    text.className = "add-input add-textarea";
    text.placeholder = "Reference transcript (optional — improves cloning for some models)";
    text.rows = 3;
    text.value = state.addText;
    text.addEventListener("input", () => { state.addText = text.value; });
    col.appendChild(text);

    const hint = el("div", "add-hint");
    col.appendChild(hint);

    const save = el("button", "add-save", "Save voice");
    save.addEventListener("click", saveVoice);
    col.appendChild(save);

    function syncSave() {
      const can = !!state.addFile && !!state.addName.trim();
      save.classList.toggle("ready", can);
      save.disabled = !can;
      const msg = !state.addFile
        ? "Choose or record audio to enable saving."
        : !state.addName.trim() ? "Name the voice to save it." : "";
      hint.textContent = msg;
      hint.classList.toggle("hidden", !msg);
    }
    syncSave();
    body.appendChild(col);
  }

  function renderEditVoiceSheet(body) {
    const col = el("div", "add-col edit-voice-col");
    const audio = el("div", "edit-audio");
    const preview = el("button", "edit-preview", "▶ Preview");
    preview.addEventListener("click", () => {
      if (state.editPreviewUrl) previewLocal(state.editPreviewUrl);
      else if (state.editVoiceId) previewVoice(state.editVoiceId);
    });
    audio.appendChild(preview);
    audio.appendChild(el("div", "edit-audio-copy", state.editFile ? `Replacement: ${state.editFileName}` : "Current saved reference audio"));
    col.appendChild(audio);

    const sources = el("div", "add-src-row");
    const upload = el("button", "add-src", state.editFile ? "＋ Replace upload" : "＋ Upload replacement");
    upload.addEventListener("click", () => $("edit-file-input").click());
    const record = el("button", "add-src" + (state.editRecording ? " recording" : ""), state.editRecording ? "■ Stop" : "● Record replacement");
    record.addEventListener("click", () => toggleRecord("edit"));
    sources.appendChild(upload);
    sources.appendChild(record);
    col.appendChild(sources);

    if (state.editFile) {
      const remove = el("button", "edit-remove-audio", "Keep current saved audio");
      remove.addEventListener("click", () => { clearEditAudio(); renderSheet(true); });
      col.appendChild(remove);
    }

    const name = document.createElement("input");
    name.className = "add-input";
    name.placeholder = "Voice name";
    name.value = state.editName;
    name.addEventListener("input", () => { state.editName = name.value; syncSave(); });
    col.appendChild(name);

    const text = document.createElement("textarea");
    text.className = "add-input add-textarea";
    text.placeholder = "Reference transcript";
    text.rows = 4;
    text.value = state.editText;
    text.addEventListener("input", () => { state.editText = text.value; syncSave(); });
    col.appendChild(text);

    const hint = el("div", "add-hint");
    col.appendChild(hint);
    const save = el("button", "add-save", "Save changes");
    save.addEventListener("click", saveEditedVoice);
    col.appendChild(save);

    function syncSave() {
      const can = !!state.editName.trim() && editVoiceDirty();
      save.disabled = !can;
      save.classList.toggle("ready", can);
      hint.textContent = !state.editName.trim() ? "Voice name is required." : !editVoiceDirty() ? "No changes yet." : "";
      hint.classList.toggle("hidden", !hint.textContent);
    }
    syncSave();
    body.appendChild(col);
  }

  /* ================= static wiring ================= */

  $("model-row").addEventListener("click", () => openSheet("model"));
  $("options-row").addEventListener("click", () => openSheet("settings"));
  $("manage-voices").addEventListener("click", () => { state.surface = "voices"; render(); });
  $("voice-none-create").addEventListener("click", () => { state.surface = "voices"; openSheet("add"); });
  $("run-btn").addEventListener("click", runGenerate);
  $("cancel-job").addEventListener("click", cancelActiveJob);
  $("choose-ref-file").addEventListener("click", () => $("ref-file-input").click());
  $("replace-ref-file").addEventListener("click", () => {
    if (state.refMode === "record") { clearReference(); render(); }
    else $("ref-file-input").click();
  });
  $("record-ref-file").addEventListener("click", () => toggleRecord("ref"));
  $("preview-ref-file").addEventListener("click", () => previewLocal(state.refPreviewUrl));
  $("save-ref-voice").addEventListener("click", saveReferenceVoice);
  $("choose-emo-file").addEventListener("click", () => $("emo-file-input").click());
  $("replace-emo-file").addEventListener("click", () => $("emo-file-input").click());
  $("preview-emo-file").addEventListener("click", () => previewLocal(state.emoPreviewUrl));
  $("add-voice-btn").addEventListener("click", () => {
    state.addName = ""; state.addText = ""; state.addFile = null; state.addFileName = ""; state.addRecording = false;
    openSheet("add");
  });
  $("mp-toggle").addEventListener("click", toggleOutput);
  $("mp-wave").addEventListener("pointerdown", (event) => seekOutput(event.clientX));
  $("mp-favorite").addEventListener("click", () => state.outputJobId && toggleFavorite(state.outputJobId));
  $("mp-download").addEventListener("click", () => state.outputJobId && downloadOutput(state.outputJobId));
  document.querySelectorAll("[data-verify-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.verifyMode === button.dataset.verifyMode) return;
      if (state.verifyRecording && recorder) discardRecording();
      state.verifyMode = button.dataset.verifyMode;
      clearVerifyAudio();
      renderVerify();
    });
  });
  $("choose-verify-file").addEventListener("click", () => $("verify-file-input").click());
  $("record-verify-file").addEventListener("click", () => toggleRecord("verify"));
  $("preview-verify-file").addEventListener("click", () => previewLocal(state.verifyPreviewUrl));
  $("replace-verify-file").addEventListener("click", () => {
    if (state.verifyMode === "record") {
      clearVerifyAudio();
      renderVerify();
    } else {
      $("verify-file-input").click();
    }
  });
  $("remove-verify-file").addEventListener("click", () => { clearVerifyAudio(); renderVerify(); });
  $("verify-btn").addEventListener("click", verifyWatermark);
  $("verify-advanced-toggle").addEventListener("click", () => {
    state.verifyAdvancedOpen = !state.verifyAdvancedOpen;
    renderVerify();
  });
  $("verify-run-select").addEventListener("change", () => {
    state.verifyRunId = $("verify-run-select").value;
    invalidateVerifyResult();
    renderVerify();
  });
  $("verify-threshold").addEventListener("input", () => {
    state.verifyThreshold = Number($("verify-threshold").value);
    invalidateVerifyResult();
    renderVerify();
  });
  $("verify-reset").addEventListener("click", () => {
    state.verifyRunId = "";
    state.verifyThreshold = 0.35;
    invalidateVerifyResult();
    renderVerify();
  });
  $("sheet-close").addEventListener("click", closeSheet);
  $("sheet-backdrop").addEventListener("click", (e) => { if (e.target === $("sheet-backdrop")) closeSheet(); });
  $("sheet").addEventListener("click", (e) => e.stopPropagation());
  $("confirm-cancel").addEventListener("click", closeConfirm);
  $("confirm-backdrop").addEventListener("click", (e) => { if (e.target === $("confirm-backdrop")) closeConfirm(); });
  $("confirm-delete").addEventListener("click", () => {
    const action = state.confirmAction;
    closeConfirm();
    if (action && action.onConfirm) action.onConfirm();
  });
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      if (state.refRecording && t.dataset.tab !== "generate" && recorder) recorder.stop();
      if (state.verifyRecording && t.dataset.tab !== "verify" && recorder) discardRecording();
      state.surface = t.dataset.tab;
      render();
    })
  );
  $("script").addEventListener("input", () => {
    state.text = $("script").value;
    const count = $("char-count");
    count.textContent = `${state.text.length} chars`;
    count.classList.toggle("err", state.text.length === 0);
    // refresh run-bar state without re-rendering the textarea
    const reason = runBlockReason();
    $("block-reason").textContent = reason || "";
    $("block-reason").classList.toggle("hidden", !reason);
    $("run-btn").classList.toggle("blocked", !!reason);
    $("run-btn").setAttribute("aria-disabled", reason ? "true" : "false");
    $("run-label").textContent = reason ? "Run" : "Run generation";
  });
  $("prompt-text").addEventListener("input", () => {
    state.promptText = $("prompt-text").value;
    renderGenerate();
  });
  $("instruct-text").addEventListener("input", () => {
    state.instructText = $("instruct-text").value;
    state.settings.cosy.instructText = state.instructText;
    renderGenerate();
  });
  $("file-input").addEventListener("change", () => {
    const f = $("file-input").files[0];
    if (f) {
      state.addFile = f;
      state.addFileName = f.name;
      $("file-input").value = "";
      renderSheet(true);
    }
  });
  $("ref-file-input").addEventListener("change", () => {
    const f = $("ref-file-input").files[0];
    if (f) {
      clearReference();
      state.refFile = f; state.refFileName = f.name;
      replaceObjectUrl("refPreviewUrl", f);
      $("ref-file-input").value = "";
      render();
    }
  });
  $("emo-file-input").addEventListener("change", () => {
    const f = $("emo-file-input").files[0];
    if (f) {
      clearEmotion();
      state.emoFile = f; state.emoFileName = f.name;
      replaceObjectUrl("emoPreviewUrl", f);
      $("emo-file-input").value = "";
      render();
    }
  });
  $("edit-file-input").addEventListener("change", () => {
    const f = $("edit-file-input").files[0];
    if (f) {
      clearEditAudio();
      state.editFile = f;
      state.editFileName = f.name;
      replaceObjectUrl("editPreviewUrl", f);
      $("edit-file-input").value = "";
      renderSheet(true);
    }
  });
  $("verify-file-input").addEventListener("change", () => {
    const f = $("verify-file-input").files[0];
    if (f) {
      clearVerifyAudio();
      state.verifyFile = f;
      state.verifyFileName = f.name;
      replaceObjectUrl("verifyPreviewUrl", f);
      $("verify-file-input").value = "";
      renderVerify();
    }
  });

  // theme
  const savedTheme = localStorage.getItem("ttshub-theme");
  if (savedTheme === "light" || savedTheme === "dark") app.dataset.theme = savedTheme;
  $("theme-toggle").addEventListener("click", () => {
    app.dataset.theme = app.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("ttshub-theme", app.dataset.theme);
    document.querySelector('meta[name="theme-color"]').content = app.dataset.theme === "dark" ? "#0c0d10" : "#eef0f3";
  });

  // clock + progress ticker
  setInterval(() => {
    renderHeader();
    if (state.activeJobId) renderTransport();
    if (outPlaying) renderMiniplayer();
  }, 500);

  document.addEventListener("visibilitychange", () => { if (!document.hidden) loadJobs(); });
  window.addEventListener("beforeunload", () => {
    discardRecording();
    ["refPreviewUrl", "emoPreviewUrl", "editPreviewUrl", "verifyPreviewUrl"].forEach((key) => { if (state[key]) URL.revokeObjectURL(state[key]); });
  });

  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/mobile/sw.js").catch(() => {});

  /* ================= boot ================= */

  (async () => {
    try {
      await Promise.all([loadModels(), loadVoices(), loadStatus(), loadWatermarkRuns()]);
      state.connected = true;
      if (!state.selectedVoiceId) {
        const cv = compatVoices();
        if (cv.length) state.selectedVoiceId = cv[0].id;
      }
    } catch {
      state.connected = false;
      flash("Cannot reach the hub — is the laptop running?", "error");
    }
    render();
    loadJobs();
    setInterval(loadStatus, 30000);
  })();
})();
