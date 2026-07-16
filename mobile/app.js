/* TTS Hub mobile — Claude Design "TTS Hub Mobile" (stack layout) wired to the real FastAPI backend. */
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
    addName: "",
    addText: "",
    addFile: null,       // File or Blob staged for the new voice
    addFileName: "",
    addRecording: false,
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
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
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
    if (state.promptText.trim()) return true;
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

  function validate() {
    if (!state.text.trim()) return "Script text is required.";
    const id = state.selectedModelId;
    if (NEEDS_REF.has(id) && !refReady()) return "Choose, upload, or record a reference voice.";
    if (id === "voxcpm-ane" && !state.settings.voxcpm.voice.trim() && !refReady()) return "VoxCPM needs a cached voice or reference audio.";
    if (needsTranscript(id) && !transcriptReady(id)) return "A reference transcript is required.";
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
  outAudio.addEventListener("error", () => {
    if (state.outputJobId) { outPlaying = false; flash("Could not play output audio.", "error"); renderMiniplayer(); }
  });

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
    if (!state.outputJobId) {
      const latestCompleted = state.jobs.find((j) => j.status === "completed" && j.output);
      if (latestCompleted) state.outputJobId = latestCompleted.id;
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
    const reason = validate();
    if (reason) { flash(reason, "error"); return; }
    const id = state.selectedModelId;
    const useVoice = state.refMode === "saved" && state.selectedVoiceId && compatVoices().some((v) => v.id === state.selectedVoiceId);
    const wm = state.watermarkEnabled && WM_SUPPORTED.has(id);

    const fd = new FormData();
    fd.append("model_id", id);
    fd.append("text", state.text);
    if (useVoice) fd.append("voice_id", state.selectedVoiceId);
    if ((state.refMode === "upload" || state.refMode === "record") && state.refFile) fd.append("prompt_audio", state.refFile, state.refFileName || "reference.webm");
    if (state.promptText.trim()) fd.append("prompt_text", state.promptText.trim());
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
      promptText: state.promptText,
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

  async function deleteVoice(v) {
    if (!window.confirm(`Delete “${v.name || v.id}”? This cannot be undone.`)) return;
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

  async function deleteJob(j) {
    if (!window.confirm("Delete this completed run and its audio? This cannot be undone.")) return;
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
      clearReference();
      flash("Reference saved to the voice library.", "success");
      render();
    } catch (e) { flash(e.message, "error"); }
  }

  /* --- recording (real MediaRecorder) --- */
  let recorder = null;
  let recorderStream = null;
  let recChunks = [];
  let recordingTarget = null;
  async function toggleRecord(target = "add") {
    const active = target === "add" ? state.addRecording : state.refRecording;
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
    renderTransport();
    renderMiniplayer();
    renderSheet();
  }

  function renderHeader() {
    $("clock").textContent = state.connected
      ? new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : "offline";
    $("clock").style.color = state.connected ? "" : "var(--err)";
  }

  function renderTabs() {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === state.surface));
    ["generate", "voices", "jobs"].forEach((s) => $(`surface-${s}`).classList.toggle("hidden", s !== state.surface));

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
      $("ref-saved").classList.toggle("hidden", state.refMode !== "saved");
      $("ref-file").classList.toggle("hidden", !["upload", "record"].includes(state.refMode));
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

      const usingSavedTranscript = state.refMode === "saved" && selectedVoice() && selectedVoice().has_transcript && !state.promptText.trim();
      const showTranscript = state.refMode !== "none" && (needsTranscript(id) || id === "qwen3-tts-mlx");
      $("prompt-text-wrap").classList.toggle("hidden", !showTranscript || usingSavedTranscript);
      $("prompt-text").value = state.promptText;
      $("prompt-text-hint").textContent = usingSavedTranscript
        ? "Using the saved voice transcript."
        : id === "qwen3-tts-mlx" && state.settings.qwen.autoTranscribe
          ? "Optional — Qwen will auto-transcribe when left blank."
          : needsTranscript(id) ? "Required for this model and mode." : "Optional.";
      $("instruct-text-wrap").classList.toggle("hidden", !(id === "cosyvoice3-mlx" && state.settings.cosy.mode === "instruct"));
      $("instruct-text").value = state.instructText;
      const emotionRef = id === "index-tts2" && state.settings.index.emoMode === "emo_ref";
      $("emotion-ref-wrap").classList.toggle("hidden", !emotionRef);
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

    // options summary
    const wmOn = state.watermarkEnabled && WM_SUPPORTED.has(id);
    const group = MODEL_GROUP[id];
    const x = state.settings[group] || {};
    const speed = x.speed != null ? `${Number(x.speed).toFixed(2)}× · ` : "";
    const temp = x.temperature != null ? `temp ${Number(x.temperature).toFixed(2)} · ` : "";
    $("options-summary").textContent = `${speed}${temp}${state.outputFormat.toUpperCase()}${wmOn ? " · WM" : ""}`;

    // run bar
    const reason = validate();
    const blocked = !!reason;
    $("block-reason").textContent = reason || "";
    $("block-reason").classList.toggle("hidden", !blocked);
    $("run-btn").classList.toggle("blocked", blocked);
    $("run-label").textContent = blocked ? "Run" : "Run generation";
  }

  function renderTransport() {
    const active = state.activeJobId ? state.jobs.find((j) => j.id === state.activeJobId) : null;
    const running = !!active && ACTIVE_STATES.includes(active.status);
    $("transport-running").classList.toggle("hidden", !running || state.surface !== "generate");
    $("runbar").classList.toggle("hidden", running || state.surface !== "generate");
    if (!running) return;

    const labels = { queued: "Queued", preparing: "Preparing", generating: "Generating", watermarking: "Watermark", converting: "Convert" };
    $("active-phase").textContent = labels[active.status] || "Working";
    const elapsed = Math.max(0, Math.floor(Date.now() / 1000 - (active.created_at || Date.now() / 1000)));
    $("active-meta").textContent = `${modelName(jobReq(active).model_id)} · ${elapsed}s elapsed`;

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
    const showMp = !!out && !state.activeJobId && state.surface !== "jobs";
    $("miniplayer").classList.toggle("hidden", !showMp);
    if (!showMp) return;

    const o = out.output || {};
    $("mp-name").textContent = o.filename || out.id;
    $("mp-meta").textContent = `${fmtSec(o.duration_s)} · ${(o.format || "").toUpperCase()}`;
    $("mp-toggle").innerHTML = outPlaying
      ? '<svg width="13" height="13" viewBox="0 0 13 13"><rect x="2" y="1" width="3.5" height="11" rx="1" fill="currentColor"/><rect x="7.5" y="1" width="3.5" height="11" rx="1" fill="currentColor"/></svg>'
      : '<svg width="14" height="14" viewBox="0 0 14 14"><path d="M2 1.5 L12.5 7 L2 12.5 Z" fill="currentColor"/></svg>';

    const n = 30, pk = peaks(out.id, n);
    const dur = o.duration_s || 0;
    const frac = outPlaying && dur ? Math.min(1, outAudio.currentTime / dur) : -1;
    const cut = Math.round(frac * n);
    const wave = $("mp-wave");
    wave.innerHTML = "";
    pk.forEach((p, i) => {
      const bar = el("span");
      bar.style.height = Math.round(15 + p * 85) + "%";
      bar.style.background = frac >= 0 && i < cut ? "var(--accent)" : frac >= 0 ? "var(--tx-3)" : "var(--accent)";
      bar.style.opacity = frac < 0 ? "0.4" : i < cut ? "1" : "0.32";
      wave.appendChild(bar);
    });
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
    [["all", "All"], ["active", "Active"], ["completed", "Done"], ["failed", "Failed"]].forEach(([k, l]) => {
      const b = el("button", "filter-btn" + (state.jobFilter === k ? " sel" : ""), l);
      b.addEventListener("click", () => { state.jobFilter = k; renderJobsPage(); });
      filters.appendChild(b);
    });

    let jobs = state.jobs;
    if (state.jobFilter === "active") jobs = jobs.filter((j) => ACTIVE_STATES.includes(j.status));
    else if (state.jobFilter === "completed") jobs = jobs.filter((j) => j.status === "completed");
    else if (state.jobFilter === "failed") jobs = jobs.filter((j) => j.status === "failed" || j.status === "cancelled");

    const list = $("job-list");
    list.innerHTML = "";
    if (!jobs.length) {
      list.appendChild(el("div", "no-jobs", "No jobs match this filter."));
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
      top.appendChild(el("span", "job-model", modelName(req.model_id)));
      top.appendChild(el("span", "job-ago", fmtAgo(j.created_at)));
      card.appendChild(top);

      card.appendChild(el("div", "job-text", req.text || "(no script recorded)"));

      const facts = el("div", "job-facts");
      facts.appendChild(el("span", null, (req.output_format || (j.output || {}).format || "?").toUpperCase()));
      facts.appendChild(el("span", null, fmtDur(j.worker_duration_ms)));
      card.appendChild(facts);
      list.appendChild(card);
    }
  }

  /* ================= sheets ================= */

  function openSheet(name) {
    state.sheet = name;
    render();
  }
  function closeSheet() {
    state.sheet = null;
    state.jobDetailId = null;
    if (state.addRecording && recorder) recorder.stop();
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
    $("sheet-title").textContent =
      { model: "Choose model", voice: "Choose voice", settings: "Options", job: "Run detail", add: "New reference voice" }[state.sheet] || "";
    const body = $("sheet-body");
    body.innerHTML = "";
    if (state.sheet === "model") renderModelSheet(body);
    else if (state.sheet === "voice") renderVoiceSheet(body);
    else if (state.sheet === "settings") renderSettingsSheet(body);
    else if (state.sheet === "job") renderJobSheet(body);
    else if (state.sheet === "add") renderAddSheet(body);
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
          ["number", "Emotion strength", "emoAlpha", 0, 1, 0.05],
          ["toggle", "Random emotion", "useRandom"],
          ["number", "Text tokens / segment", "maxTextTokens", 20, 500, 1],
          ["number", "Maximum mel tokens", "maxMelTokens", 100, 3000, 10],
          ["toggle", "Fast mode", "fastMode"],
        ],
        advanced: [
          ["toggle", "Sampling", "doSample"],
          ["number", "Temperature", "temperature", 0.1, 2, 0.05],
          ["number", "Top P", "topP", 0, 1, 0.05],
          ["number", "Top K", "topK", 0, 200, 1],
          ["number", "Beams", "numBeams", 1, 10, 1],
          ["number", "Repetition penalty", "repetitionPenalty", 0, 20, 0.1],
          ["number", "Length penalty", "lengthPenalty", -5, 5, 0.1],
          ["textarea", "Emotion vector (8 values)", "emoVector", () => x.emoMode === "emo_vector"],
          ["textarea", "Emotion text", "emoText", () => x.emoMode === "emo_text"],
        ],
      },
      chatterbox: {
        basics: [
          ["text", "Language code", "language"],
          ["toggle", "Use reference voice", "usePrompt"],
          ["number", "CFG weight", "cfgWeight", 0, 1, 0.05],
          ["number", "Temperature", "temperature", 0.1, 2, 0.05],
          ["number", "Exaggeration", "exaggeration", 0, 2, 0.05],
          ["toggle", "Fast mode", "fastMode"],
        ],
        advanced: [
          ["toggle", "Long-form chunking", "enableChunking"],
          ["number", "Maximum chunk characters", "maxChunkChars", 30, 1000, 1],
          ["number", "Crossfade (ms)", "crossfadeMs", 0, 500, 1],
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
          ["toggle", "Auto-transcribe reference", "autoTranscribe"],
          ["text", "Language", "language"],
          ["number", "Speed", "speed", 0.5, 2, 0.05],
          ["number", "Temperature", "temperature", 0.1, 2, 0.05],
        ],
        advanced: [["number", "Maximum tokens", "maxTokens", 100, 4000, 10]],
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

    function settingRow(spec) {
      const [type, label, key, a, b, c] = spec;
      if (typeof a === "function" && !a()) return null;
      const row = el("div", "setting-row");
      row.appendChild(el("label", null, label));
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
      } else {
        control = document.createElement("input");
        control.type = type;
        control.value = x[key] == null ? "" : x[key];
        if (type === "number") { control.min = a; control.max = b; control.step = c; }
        control.addEventListener("input", () => { x[key] = type === "number" ? Number(control.value) : control.value; renderGenerate(); });
      }
      control.classList.add("setting-control");
      row.appendChild(control);
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
        state.surface = "generate";
        state.sheet = null;
        state.jobDetailId = null;
        playOutput(dj.id, true);
      });
      const dl = el("button", "dj-dl", "↓ Download");
      dl.addEventListener("click", () => downloadOutput(dj.id));
      actions.appendChild(play);
      actions.appendChild(dl);
      col.appendChild(actions);
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
  $("mp-download").addEventListener("click", () => state.outputJobId && downloadOutput(state.outputJobId));
  $("sheet-close").addEventListener("click", closeSheet);
  $("sheet-backdrop").addEventListener("click", (e) => { if (e.target === $("sheet-backdrop")) closeSheet(); });
  $("sheet").addEventListener("click", (e) => e.stopPropagation());
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      if (state.refRecording && t.dataset.tab !== "generate" && recorder) recorder.stop();
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
    const reason = validate();
    $("block-reason").textContent = reason || "";
    $("block-reason").classList.toggle("hidden", !reason);
    $("run-btn").classList.toggle("blocked", !!reason);
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
    if (recorder && recorder.state === "recording") recorder.stop();
    if (recorderStream) recorderStream.getTracks().forEach((track) => track.stop());
    ["refPreviewUrl", "emoPreviewUrl"].forEach((key) => { if (state[key]) URL.revokeObjectURL(state[key]); });
  });

  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/mobile/sw.js").catch(() => {});

  /* ================= boot ================= */

  (async () => {
    try {
      await Promise.all([loadModels(), loadVoices(), loadStatus()]);
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
