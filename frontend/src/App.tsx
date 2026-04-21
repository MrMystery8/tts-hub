import { useEffect, useMemo, useRef, useState, useTransition } from 'react';
import type { AppSettings, AppSnapshot, AppHistoryItem, ModelSpec, ModelStatus, SurfaceId, VoiceSummary, WatermarkRun, WatermarkRunDetails } from './types';
import { appendModelParams, buildDefaultSettings, isWatermarkSupported, requiresReferenceAudio, requiresTranscript, WATERMARK_MODEL_MAP, MODEL_ICONS } from './model-settings';
import { createVoice, deleteVoice, fetchInfo, fetchModels, fetchStatus, fetchVoices, fetchWatermarkRunDetails, fetchWatermarkRuns, generateAudio, getVoiceMeta, unloadModel } from './api';
import { sessionStore } from './storage';
import './styles.css';

type StatusKind = 'info' | 'success' | 'warning' | 'error' | 'loading';

type PromptSource =
  | { kind: 'none' }
  | { kind: 'upload'; file: Blob; url: string; info: string; revoke: boolean }
  | { kind: 'record'; blob: Blob; url: string; info: string; revoke: boolean }
  | { kind: 'saved'; voiceId: string; url: string; info: string; revoke: false };

type RecorderState = {
  stream: MediaStream | null;
  audioContext: AudioContext | null;
  source: MediaStreamAudioSourceNode | null;
  processor: ScriptProcessorNode | null;
  buffers: Float32Array[];
  sampleRate: number;
};

type AppController = ReturnType<typeof useAppController>;

function normalizeModelName(name: string): string {
  return name.split(' (')[0] || name;
}

function formatBytes(bytes: number | undefined | null): string {
  if (bytes === undefined || bytes === null) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDuration(seconds: number | undefined | null): string {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

function formatDateTime(ts: number | undefined | null): string {
  if (ts === undefined || ts === null || !Number.isFinite(ts)) return 'Unknown';
  return new Date(ts).toLocaleString();
}

function mergeSettings(saved: Partial<AppSettings> | null | undefined): AppSettings {
  const defaults = buildDefaultSettings();
  if (!saved) return defaults;
  return {
    index: { ...defaults.index, ...(saved.index || {}) },
    chatterbox: { ...defaults.chatterbox, ...(saved.chatterbox || {}) },
    f5: { ...defaults.f5, ...(saved.f5 || {}) },
    cosy: { ...defaults.cosy, ...(saved.cosy || {}) },
    qwen: { ...defaults.qwen, ...(saved.qwen || {}) },
    pocket: { ...defaults.pocket, ...(saved.pocket || {}) },
    voxcpm: { ...defaults.voxcpm, ...(saved.voxcpm || {}) },
  };
}

function loadSnapshot(): AppSnapshot {
  return {
    selectedModelId: sessionStore.load<string | null>('selectedModel', null),
    activeSurface: sessionStore.load<SurfaceId>('activeSurface', 'generate'),
    text: sessionStore.load('text', ''),
    promptText: sessionStore.load('promptText', ''),
    selectedVoiceId: sessionStore.load('selectedVoiceId', ''),
    outputFormat: sessionStore.load<'wav' | 'mp3' | 'flac'>('outputFormat', 'wav'),
    watermarkEnabled: sessionStore.load('watermarkEnabled', false),
    watermarkRun: sessionStore.load('watermarkRun', ''),
    watermarkThresholdAuto: sessionStore.load('watermarkThresholdAuto', '1') !== '0',
    watermarkThresholdManual: sessionStore.load('watermarkThresholdManual', ''),
    settings: mergeSettings(sessionStore.load<Partial<AppSettings> | null>('settings', null)),
    history: sessionStore.load<AppHistoryItem[]>('history', []),
  };
}

function createWavBlob(buffers: Float32Array[], sampleRate: number): Blob {
  let length = 0;
  for (const chunk of buffers) length += chunk.length;
  const samples = new Float32Array(length);
  let offset = 0;
  for (const chunk of buffers) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset2: number, str: string) => {
    for (let i = 0; i < str.length; i += 1) view.setUint8(offset2 + i, str.charCodeAt(i));
  };
  const floatTo16BitPCM = (offset2: number, input: Float32Array) => {
    for (let i = 0; i < input.length; i += 1) {
      const s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset2 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPCM(44, samples);
  return new Blob([buffer], { type: 'audio/wav' });
}

function useAppController() {
  const snapshot = loadSnapshot();

  const [models, setModels] = useState<ModelSpec[]>([]);
  const [modelStatuses, setModelStatuses] = useState<Record<string, ModelStatus>>({});
  const [voices, setVoices] = useState<VoiceSummary[]>([]);
  const [watermarkRuns, setWatermarkRuns] = useState<WatermarkRun[]>([]);
  const [watermarkDefaultRunId, setWatermarkDefaultRunId] = useState<string | null>(null);
  const [watermarkRunDetails, setWatermarkRunDetails] = useState<WatermarkRunDetails | null>(null);
  const [watermarkRecommendedThreshold, setWatermarkRecommendedThreshold] = useState<number | null>(null);
  const [infoAvailable, setInfoAvailable] = useState<boolean | null>(null);

  const [selectedModelId, setSelectedModelId] = useState<string | null>(snapshot.selectedModelId);
  const [activeSurface, setActiveSurfaceState] = useState<SurfaceId>(snapshot.activeSurface);
  const [text, setText] = useState(snapshot.text);
  const [promptText, setPromptText] = useState(snapshot.promptText);
  const [selectedVoiceId, setSelectedVoiceId] = useState(snapshot.selectedVoiceId);
  const [outputFormat, setOutputFormat] = useState<'wav' | 'mp3' | 'flac'>(snapshot.outputFormat);
  const [watermarkEnabled, setWatermarkEnabled] = useState(snapshot.watermarkEnabled);
  const [watermarkRun, setWatermarkRun] = useState(snapshot.watermarkRun);
  const [watermarkThresholdAuto, setWatermarkThresholdAuto] = useState(snapshot.watermarkThresholdAuto);
  const [watermarkThresholdManual, setWatermarkThresholdManual] = useState(snapshot.watermarkThresholdManual);
  const [settings, setSettings] = useState<AppSettings>(snapshot.settings);
  const [history, setHistory] = useState<AppHistoryItem[]>(snapshot.history);
  const [statusMessage, setStatusMessage] = useState('Loading React UI...');
  const [statusKind, setStatusKind] = useState<StatusKind>('loading');
  const [isGenerating, setIsGenerating] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  const [promptSource, setPromptSource] = useState<PromptSource>({ kind: 'none' });
  const [emotionFile, setEmotionFile] = useState<File | null>(null);
  const [watermarkDetectFile, setWatermarkDetectFile] = useState<File | null>(null);
  const [watermarkDetectPreviewUrl, setWatermarkDetectPreviewUrl] = useState('');
  const [watermarkDetectFileInfo, setWatermarkDetectFileInfo] = useState('');
  const [watermarkDetectResult, setWatermarkDetectResult] = useState('No file analyzed yet');
  const [recordingTime, setRecordingTime] = useState('0:00');
  const [isRecording, setIsRecording] = useState(false);
  const [systemClock, setSystemClock] = useState(() => new Date().toLocaleString());
  const [outputUrl, setOutputUrl] = useState('');
  const [outputFileName, setOutputFileName] = useState('output.wav');

  const promptFileRef = useRef<HTMLInputElement | null>(null);
  const emotionFileRef = useRef<HTMLInputElement | null>(null);
  const watermarkDetectFileRef = useRef<HTMLInputElement | null>(null);
  const recorderRef = useRef<RecorderState>({
    stream: null,
    audioContext: null,
    source: null,
    processor: null,
    buffers: [],
    sampleRate: 48000,
  });
  const recordingTimerRef = useRef<number | null>(null);

  const selectedModel = useMemo(() => models.find((model) => model.id === selectedModelId) || null, [models, selectedModelId]);
  const selectedModelStatus = useMemo<ModelStatus>(() => (selectedModelId ? modelStatuses[selectedModelId] || {} : {}), [modelStatuses, selectedModelId]);
  const selectedVoice = useMemo(() => voices.find((voice) => voice.id === selectedVoiceId) || null, [selectedVoiceId, voices]);
  const supportsWatermark = useMemo(() => isWatermarkSupported(selectedModelId), [selectedModelId]);
  const transcriptRequired = useMemo(() => requiresTranscript(selectedModelId, settings), [selectedModelId, settings]);
  const needsReferenceAudio = useMemo(() => requiresReferenceAudio(selectedModelId), [selectedModelId]);
  const surfaceTitle = useMemo(() => {
    switch (activeSurface) {
      case 'models':
        return 'Models';
      case 'voices':
        return 'Voices';
      case 'history':
        return 'History';
      case 'watermark-lab':
        return 'Watermark Lab';
      case 'system-status':
        return 'System Status';
      case 'advanced-settings':
        return 'Advanced Settings';
      default:
        return 'Generate';
    }
  }, [activeSurface]);

  const setSurface = (surface: SurfaceId) => {
    setActiveSurfaceState(surface);
    sessionStore.save('activeSurface', surface);
    window.history.replaceState(null, '', `#${surface}`);
  };

  const setStatus = (message: string, kind: StatusKind = 'info') => {
    setStatusMessage(message);
    setStatusKind(kind);
  };

  const updateSettings = <K extends keyof AppSettings>(key: K, next: AppSettings[K]) => {
    setSettings((current) => ({ ...current, [key]: next }));
  };

  const updateModelStatus = async (modelId?: string) => {
    try {
      const response = await fetchStatus(modelId);
      setModelStatuses(response.models || {});
    } catch {
      // keep old status if the request fails
    }
  };

  const refreshWatermarkDetails = async (runId?: string | null) => {
    const targetRun = (runId || watermarkRun || watermarkDefaultRunId || '').trim();
    if (!watermarkRuns.length && !targetRun) {
      setWatermarkRunDetails(null);
      setWatermarkRecommendedThreshold(null);
      return;
    }
    try {
      const details = await fetchWatermarkRunDetails(targetRun || undefined);
      setWatermarkRunDetails(details);
      const threshold = typeof details.metrics?.thr_at_fpr_1pct === 'number' ? (details.metrics.thr_at_fpr_1pct as number) : null;
      setWatermarkRecommendedThreshold(Number.isFinite(threshold as number) ? (threshold as number) : null);
    } catch (error) {
      setWatermarkRunDetails({ id: targetRun || '', error: error instanceof Error ? error.message : String(error) });
      setWatermarkRecommendedThreshold(null);
    }
  };

  const loadPromptAudioFromBlob = async (blob: Blob, label: string, kind: 'upload' | 'record' = 'upload') => {
    const url = URL.createObjectURL(blob);
    setPromptSource(
      kind === 'upload'
        ? { kind: 'upload', file: blob, url, info: `${label} • ${blob.type || 'audio'} • ${formatBytes(blob.size)}`, revoke: true }
        : { kind: 'record', blob, url, info: `${label} • ${blob.type || 'audio'} • ${formatBytes(blob.size)}`, revoke: true },
    );
    await sessionStore.saveAudio('promptAudio', blob);
    setSelectedVoiceId('');
    await sessionStore.save('selectedVoiceId', '');
  };

  const clearPromptAudio = async () => {
    setPromptSource({ kind: 'none' });
    setSelectedVoiceId('');
    await sessionStore.saveAudio('promptAudio', null);
    await sessionStore.save('selectedVoiceId', '');
    if (promptFileRef.current) promptFileRef.current.value = '';
  };

  const selectSavedVoice = async (voiceId: string, options?: { silent?: boolean; voiceList?: VoiceSummary[] }) => {
    const normalized = voiceId.trim();
    if (!normalized) {
      setSelectedVoiceId('');
      setPromptSource({ kind: 'none' });
      await sessionStore.save('selectedVoiceId', '');
      return;
    }

    const list = options?.voiceList || voices;
    const summary = list.find((voice) => voice.id === normalized) || null;
    setSelectedVoiceId(normalized);
    await sessionStore.save('selectedVoiceId', normalized);
    await sessionStore.saveAudio('promptAudio', null);

    const previewUrl = `/api/voices/${encodeURIComponent(normalized)}/audio?ts=${Date.now()}`;
    setPromptSource({
      kind: 'saved',
      voiceId: normalized,
      url: previewUrl,
      info: summary
        ? `Saved voice: ${summary.name}${summary.duration_s ? ` • ${formatDuration(summary.duration_s)}` : ''}`
        : 'Saved voice',
      revoke: false,
    });

    try {
      const meta = await getVoiceMeta(normalized);
      const prompt = String(meta.prompt_text || '').trim();
      if (prompt) setPromptText(prompt);
    } catch {
      // no-op
    }

    if (!options?.silent) {
      setStatus(summary ? `Using saved voice: ${summary.name}` : 'Using saved voice.', 'success');
    }
  };

  const clearVoicePreview = async () => {
    await clearPromptAudio();
    setEmotionFile(null);
    if (emotionFileRef.current) emotionFileRef.current.value = '';
    if (promptFileRef.current) promptFileRef.current.value = '';
  };

  const handlePromptFileChange = async (file: File | null) => {
    if (!file) return;
    setSelectedVoiceId('');
    await sessionStore.save('selectedVoiceId', '');
    const url = URL.createObjectURL(file);
    setPromptSource({
      kind: 'upload',
      file,
      url,
      info: `${file.name} • ${file.type || 'audio'} • ${formatBytes(file.size)}`,
      revoke: true,
    });
    await sessionStore.saveAudio('promptAudio', file);
  };

  const handlePromptRecording = async () => {
    if (isRecording) {
      const recorder = recorderRef.current;
      if (recordingTimerRef.current) window.clearInterval(recordingTimerRef.current);
      recorder.stream?.getTracks().forEach((track) => track.stop());
      recorder.processor?.disconnect();
      recorder.source?.disconnect();
      if (recorder.audioContext) await recorder.audioContext.close();
      const blob = createWavBlob(recorder.buffers, recorder.sampleRate);
      recorder.buffers = [];
      recorder.audioContext = null;
      recorder.processor = null;
      recorder.source = null;
      recorder.stream = null;
      setIsRecording(false);
      setRecordingTime('0:00');
      setStatus('Recording saved.', 'success');
      await loadPromptAudioFromBlob(blob, 'Recorded', 'record');
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('Microphone access is not available in this browser.', 'error');
      return;
    }

    try {
      setStatus('Recording...', 'loading');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const recorder = recorderRef.current;
      recorder.stream = stream;
      recorder.audioContext = audioContext;
      recorder.source = source;
      recorder.processor = processor;
      recorder.sampleRate = audioContext.sampleRate;
      recorder.buffers = [];
      processor.onaudioprocess = (event) => {
        recorder.buffers.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      setIsRecording(true);
      const startedAt = Date.now();
      recordingTimerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startedAt) / 1000;
        setRecordingTime(formatDuration(elapsed));
      }, 100);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error), 'error');
    }
  };

  const handleEmotionFileChange = (file: File | null) => {
    setEmotionFile(file);
  };

  const handleWatermarkDetectFileChange = (file: File | null) => {
    setWatermarkDetectFile(file);
    if (!file) {
      setWatermarkDetectFileInfo('');
      setWatermarkDetectResult('No file analyzed yet');
      return;
    }
    setWatermarkDetectFileInfo(`Selected: ${file.name}${formatBytes(file.size) ? ` • ${formatBytes(file.size)}` : ''}`);
    setWatermarkDetectResult('Ready to analyze.');
  };

  const handleGenerate = async () => {
    const modelId = selectedModelId;
    if (!modelId) {
      setStatus('Please select a model.', 'error');
      return;
    }
    const textValue = text.trim();
    if (!textValue) {
      setStatus('Please enter some text.', 'error');
      return;
    }

    const referenceAudio = promptSource.kind === 'upload' || promptSource.kind === 'record';
    const hasSavedVoice = promptSource.kind === 'saved';
    const promptTranscript = promptText.trim();

    if (requiresReferenceAudio(modelId) && !referenceAudio && !hasSavedVoice) {
      setStatus('This model requires a reference audio or saved voice.', 'error');
      return;
    }
    if (modelId === 'cosyvoice3-mlx' && settings.cosy.mode === 'zero_shot' && !promptTranscript) {
      setStatus('CosyVoice3 zero_shot requires a reference transcript.', 'error');
      return;
    }
    if (modelId === 'cosyvoice3-mlx' && settings.cosy.mode === 'instruct' && !settings.cosy.instructText.trim()) {
      setStatus('CosyVoice3 instruct mode requires instruction text.', 'error');
      return;
    }
    if (modelId === 'voxcpm-ane') {
      const cachedVoice = settings.voxcpm.voice.trim();
      if (!cachedVoice && !referenceAudio && !hasSavedVoice) {
        setStatus('VoxCPM-ANE requires a voice or prompt audio.', 'error');
        return;
      }
      if (!cachedVoice && !promptTranscript) {
        setStatus('VoxCPM-ANE requires a transcript when no cached voice is selected.', 'error');
        return;
      }
    }

    const form = new FormData();
    form.append('model_id', modelId);
    form.append('text', textValue);
    form.append('output_format', outputFormat);
    if (promptTranscript) form.append('prompt_text', promptTranscript);

    if (hasSavedVoice) {
      form.append('voice_id', selectedVoiceId);
    } else if (referenceAudio) {
      if (promptSource.kind === 'upload') {
        form.append('prompt_audio', promptSource.file);
      } else if (promptSource.kind === 'record') {
        const file = new File([promptSource.blob], 'prompt.wav', { type: 'audio/wav' });
        form.append('prompt_audio', file);
      }
    }

    if (modelId === 'index-tts2' && settings.index.emoMode === 'emo_ref' && emotionFile) {
      form.append('emo_audio', emotionFile);
    }

    appendModelParams(form, modelId, settings, { emotionAudio: emotionFile });

    if (watermarkEnabled) {
      form.append('watermark', '1');
      const runId = watermarkRun.trim();
      if (runId) form.append('watermark_run', runId);
    }

    setIsGenerating(true);
    setStatus(`Generating with ${normalizeModelName(selectedModel?.name || modelId)}...`, 'loading');

    try {
      const result = await generateAudio(form);
      const url = URL.createObjectURL(result.blob);
      setHistory((current) => [
        {
          modelId,
          timestamp: Date.now(),
          format: outputFormat,
          url,
          watermarkEnabled,
          watermarkRun: watermarkRun || null,
          voiceId: selectedVoiceId || null,
          settingsSummary: `${normalizeModelName(selectedModel?.name || modelId)} • ${outputFormat.toUpperCase()}`,
        },
        ...current,
      ].slice(0, 20));
      setOutputUrl(url);
      setOutputFileName(`${modelId}${watermarkEnabled ? '_wm' : ''}.${outputFormat}`);
      setStatus('Generation complete.', 'success');
      void updateModelStatus(modelId);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error), 'error');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSaveVoice = async () => {
    const source = promptSource.kind === 'upload' || promptSource.kind === 'record';
    if (!source) {
      setStatus('Upload or record a voice first.', 'error');
      return;
    }

    const name = (window.prompt('Saved voice name:', '') || '').trim();
    if (!name) return;

    const form = new FormData();
    form.append('name', name);
    if (promptSource.kind === 'upload') {
      form.append('prompt_audio', promptSource.file);
    } else if (promptSource.kind === 'record') {
      const file = new File([promptSource.blob], 'prompt.wav', { type: 'audio/wav' });
      form.append('prompt_audio', file);
    }
    const prompt = promptText.trim();
    if (prompt) form.append('prompt_text', prompt);

    setStatus('Saving voice...', 'loading');
    try {
      const meta = await createVoice(form);
      const voiceId = String(meta.id || '').trim();
      const refreshed = await fetchVoices();
      setVoices(refreshed);
      if (voiceId) {
        await selectSavedVoice(voiceId, { silent: true, voiceList: refreshed });
      }
      setStatus('Voice saved.', 'success');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error), 'error');
    }
  };

  const handleDeleteVoice = async () => {
    if (!selectedVoiceId) return;
    if (!window.confirm('Delete this saved voice? This cannot be undone.')) return;
    setStatus('Deleting voice...', 'loading');
    try {
      await deleteVoice(selectedVoiceId);
      await clearPromptAudio();
      const refreshed = await fetchVoices();
      setVoices(refreshed);
      setSelectedVoiceId('');
      setStatus('Voice deleted.', 'success');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error), 'error');
    }
  };

  const handleUnloadModel = async () => {
    if (!selectedModelId) return;
    setStatus(`Unloading ${normalizeModelName(selectedModel?.name || selectedModelId)}...`, 'loading');
    try {
      await unloadModel(selectedModelId);
      setStatus(`Unloaded ${normalizeModelName(selectedModel?.name || selectedModelId)}.`, 'success');
      void updateModelStatus();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error), 'error');
    }
  };

  const handleResetAll = async () => {
    await clearPromptAudio();
    setPromptText('');
    setEmotionFile(null);
    setText('');
    setOutputFormat('wav');
    setWatermarkEnabled(false);
    setWatermarkRun('');
    setWatermarkThresholdAuto(true);
    setWatermarkThresholdManual('');
    setOutputUrl('');
    setOutputFileName('output.wav');
    setStatus('Reset complete.', 'success');
  };

  const handleClearSession = async () => {
    await sessionStore.clearAll();
    setSelectedModelId(models[0]?.id || null);
    setActiveSurfaceState('generate');
    setText('');
    setPromptText('');
    setSelectedVoiceId('');
    setOutputFormat('wav');
    setWatermarkEnabled(false);
    setWatermarkRun('');
    setWatermarkThresholdAuto(true);
    setWatermarkThresholdManual('');
    setSettings(buildDefaultSettings());
    setHistory([]);
    setPromptSource({ kind: 'none' });
    setEmotionFile(null);
    setWatermarkDetectFile(null);
    setWatermarkDetectPreviewUrl('');
    setWatermarkDetectFileInfo('');
    setWatermarkDetectResult('No file analyzed yet');
    setOutputUrl('');
    setOutputFileName('output.wav');
    setStatus('Session cleared.', 'success');
  };

  const handleWatermarkDetect = async () => {
    const file = watermarkDetectFile;
    if (!file) {
      setWatermarkDetectResult('Please choose an audio file first.');
      return;
    }
    const threshold = watermarkThresholdAuto
      ? (watermarkRecommendedThreshold ?? 0.35)
      : Number.parseFloat(watermarkThresholdManual || '0.35');
    const form = new FormData();
    form.append('audio', file);
    form.append('wm_threshold', String(Number.isFinite(threshold) ? threshold : 0.35));
    if (watermarkRun.trim()) form.append('watermark_run', watermarkRun.trim());

    setWatermarkDetectResult('Analyzing...');
    try {
      const response = await fetch('/api/watermark/detect', { method: 'POST', body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);

      const lines: string[] = [];
      lines.push(`Detected: ${data.detected ? 'YES' : 'NO'}`);
      if (typeof data.wm_prob === 'number') lines.push(`WM prob: ${data.wm_prob.toFixed(3)}`);
      if (data.detected && data.model) {
        lines.push(`Model: ${data.model.name || 'Unknown'}${data.model.id !== undefined ? ` (ID ${data.model.id})` : ''}`);
      }
      if (data.run?.id) lines.push(`Run: ${data.run.id}`);
      setWatermarkDetectResult(lines.join('\n'));
    } catch (error) {
      setWatermarkDetectResult(`Error: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleWatermarkRunChange = async (runId: string) => {
    setWatermarkRun(runId);
    sessionStore.save('watermarkRun', runId);
    await refreshWatermarkDetails(runId);
  };

  const handleWatermarkThresholdAutoChange = (checked: boolean) => {
    setWatermarkThresholdAuto(checked);
    sessionStore.save('watermarkThresholdAuto', checked ? '1' : '0');
  };

  const handleWatermarkThresholdManualChange = (value: string) => {
    setWatermarkThresholdManual(value);
    sessionStore.save('watermarkThresholdManual', value);
  };

  const handleModelSelect = (modelId: string) => {
    setSelectedModelId(modelId);
    sessionStore.save('selectedModel', modelId);
  };

  const handleSurfaceSelect = (surface: SurfaceId) => {
    setSurface(surface);
  };

  useEffect(() => {
    const onHashChange = () => {
      const hash = window.location.hash.replace(/^#/, '') as SurfaceId;
      if (hash && ['generate', 'models', 'voices', 'history', 'watermark-lab', 'system-status', 'advanced-settings'].includes(hash)) {
        setActiveSurfaceState(hash);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await sessionStore.init();
        const [modelsData, infoData, voicesData, watermarkData, statusData] = await Promise.all([
          fetchModels(),
          fetchInfo(),
          fetchVoices(),
          fetchWatermarkRuns(),
          fetchStatus(),
        ]);
        if (cancelled) return;

        setModels(modelsData);
        setVoices(voicesData);
        setInfoAvailable(!!infoData.ffmpeg?.available);
        setWatermarkRuns(watermarkData.runs || []);
        setWatermarkDefaultRunId(watermarkData.default_run_id || null);
        setModelStatuses(statusData.models || {});

        const nextModelId = snapshot.selectedModelId && modelsData.some((model) => model.id === snapshot.selectedModelId)
          ? snapshot.selectedModelId
          : (modelsData[0]?.id || null);
        setSelectedModelId(nextModelId);

        if (snapshot.selectedVoiceId && voicesData.some((voice) => voice.id === snapshot.selectedVoiceId)) {
          await selectSavedVoice(snapshot.selectedVoiceId, { silent: true, voiceList: voicesData });
        } else {
          const savedAudio = await sessionStore.loadAudio('promptAudio');
          if (savedAudio) {
            const url = URL.createObjectURL(savedAudio);
            setPromptSource({
              kind: 'record',
              blob: savedAudio,
              url,
              info: `Restored • ${savedAudio.type || 'audio'} • ${formatBytes(savedAudio.size)}`,
              revoke: true,
            });
          }
        }

        const initialRunId = snapshot.watermarkRun || watermarkData.default_run_id || '';
        if (initialRunId) {
          setWatermarkRun(initialRunId);
          await refreshWatermarkDetails(initialRunId);
        }

        setHydrated(true);
        setStatus('Ready.', 'info');
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error), 'error');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setSystemClock(new Date().toLocaleString()), 30000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    sessionStore.save('selectedModel', selectedModelId);
    sessionStore.save('activeSurface', activeSurface);
    sessionStore.save('text', text);
    sessionStore.save('promptText', promptText);
    sessionStore.save('selectedVoiceId', selectedVoiceId);
    sessionStore.save('outputFormat', outputFormat);
    sessionStore.save('watermarkEnabled', watermarkEnabled);
    sessionStore.save('watermarkRun', watermarkRun);
    sessionStore.save('watermarkThresholdAuto', watermarkThresholdAuto ? '1' : '0');
    sessionStore.save('watermarkThresholdManual', watermarkThresholdManual);
    sessionStore.save('settings', settings);
    sessionStore.save('history', history.map((item) => ({ ...item, url: null })));
  }, [
    activeSurface,
    history,
    outputFormat,
    promptText,
    selectedModelId,
    selectedVoiceId,
    settings,
    text,
    watermarkEnabled,
    watermarkRun,
    watermarkThresholdAuto,
    watermarkThresholdManual,
  ]);

  useEffect(() => {
    return () => {
      if (promptSource.kind === 'upload' || promptSource.kind === 'record') {
        URL.revokeObjectURL(promptSource.url);
      }
    };
  }, [promptSource]);

  useEffect(() => {
    return () => {
      if (watermarkDetectPreviewUrl) URL.revokeObjectURL(watermarkDetectPreviewUrl);
    };
  }, [watermarkDetectPreviewUrl]);

  useEffect(() => {
    if (watermarkDetectFile) {
      const url = URL.createObjectURL(watermarkDetectFile);
      setWatermarkDetectPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    setWatermarkDetectPreviewUrl('');
    return undefined;
  }, [watermarkDetectFile]);

  useEffect(() => {
    void refreshWatermarkDetails(watermarkRun || watermarkDefaultRunId || '');
  }, [watermarkDefaultRunId, watermarkRuns.length]);

  useEffect(() => {
    if (!hydrated) return;
    void updateModelStatus(selectedModelId || undefined);
  }, [hydrated, selectedModelId]);

  return {
    models,
    modelStatuses,
    voices,
    watermarkRuns,
    watermarkDefaultRunId,
    watermarkRunDetails,
    watermarkRecommendedThreshold,
    selectedModel,
    selectedModelStatus,
    selectedVoice,
    selectedModelId,
    activeSurface,
    setActiveSurface: handleSurfaceSelect,
    setSelectedModelId: handleModelSelect,
    text,
    setText,
    promptText,
    setPromptText,
    selectedVoiceId,
    setSelectedVoiceId,
    outputFormat,
    setOutputFormat,
    watermarkEnabled,
    setWatermarkEnabled,
    watermarkRun,
    setWatermarkRun: handleWatermarkRunChange,
    watermarkThresholdAuto,
    setWatermarkThresholdAuto: handleWatermarkThresholdAutoChange,
    watermarkThresholdManual,
    setWatermarkThresholdManual: handleWatermarkThresholdManualChange,
    settings,
    setSettings,
    updateSettings,
    history,
    setHistory,
    statusMessage,
    statusKind,
    setStatus,
    isGenerating,
    hydrated,
    promptSource,
    emotionFile,
    setEmotionFile: handleEmotionFileChange,
    watermarkDetectFile,
    watermarkDetectPreviewUrl,
    watermarkDetectFileInfo,
    watermarkDetectResult,
    recordingTime,
    isRecording,
    systemClock,
    infoAvailable,
    supportsWatermark,
    transcriptRequired,
    needsReferenceAudio,
    promptFileRef,
    emotionFileRef,
    watermarkDetectFileRef,
    setPromptFile: handlePromptFileChange,
    setPromptRecording: handlePromptRecording,
    clearPromptAudio,
    clearVoicePreview,
    selectSavedVoice,
    handleSaveVoice,
    handleDeleteVoice,
    handleGenerate,
    handleUnloadModel,
    handleResetAll,
    handleClearSession,
    handleWatermarkDetect,
    setWatermarkDetectFile: handleWatermarkDetectFileChange,
    surfaceTitle,
    outputUrl,
    outputFileName,
    normalizeModelName,
    setWatermarkDetectResult,
    setWatermarkDetectFileInfo,
    setWatermarkRuns,
    setWatermarkDefaultRunId,
    setWatermarkRunDetails,
    setWatermarkRecommendedThreshold,
    setModels,
    setVoices,
    setModelStatuses,
    setInfoAvailable,
    setWatermarkDetectPreviewUrl,
  };
}

function Badge({ tone, children }: { tone: 'neutral' | 'success' | 'warning' | 'info' | 'danger'; children: React.ReactNode }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Panel({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`.trim()}>
      <div className="panel-header">
        <div>
          <div className="panel-title">{title}</div>
          {subtitle ? <div className="panel-subtitle">{subtitle}</div> : null}
        </div>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

function SurfaceButton({
  active,
  label,
  surface,
  onClick,
}: {
  active: boolean;
  label: string;
  surface: SurfaceId;
  onClick: (surface: SurfaceId) => void;
}) {
  return (
    <button className={`surface-nav-item${active ? ' active' : ''}`} type="button" data-surface-link={surface} onClick={() => onClick(surface)}>
      {label}
    </button>
  );
}

function HistoryRow({ item, modelName }: { item: AppHistoryItem; modelName: string }) {
  return (
    <div className="history-item">
      <div className="history-icon">{MODEL_ICONS[item.modelId] || '🔊'}</div>
      <div className="history-info">
        <div className="history-title">{modelName}</div>
        <div className="history-meta">
          {formatDateTime(item.timestamp)} · {String(item.format || '').toUpperCase()}
          {item.watermarkEnabled ? ' · Watermark' : ''}
        </div>
      </div>
      <div className="history-actions">
        {item.url ? (
          <a className="btn btn-secondary" href={item.url} download={`${item.modelId}.${item.format}`}>
            Download
          </a>
        ) : null}
      </div>
    </div>
  );
}

function ModelCard({
  model,
  status,
  active,
  onClick,
}: {
  model: ModelSpec;
  status: ModelStatus;
  active: boolean;
  onClick: (modelId: string) => void;
}) {
  const loaded = !!status.loaded;
  const total = typeof status.total_generations === 'number' ? status.total_generations : 0;
  const device = status.device || 'unknown';
  const modelBadges = [
    <Badge key="loaded" tone={loaded ? 'success' : 'neutral'}>{loaded ? 'Loaded' : 'Ready'}</Badge>,
    <Badge key="device" tone="neutral">{device}</Badge>,
    <Badge key="runs" tone="neutral">{total} runs</Badge>,
    isWatermarkSupported(model.id) ? <Badge key="wm" tone="info">Watermark</Badge> : null,
  ].filter(Boolean);

  return (
    <button className={`model-card${active ? ' active' : ''}`} type="button" onClick={() => onClick(model.id)}>
      <div className="model-icon">{MODEL_ICONS[model.id] || '🔊'}</div>
      <div className="model-info">
        <div className="model-name">{normalizeModelName(model.name)}</div>
        <div className="model-desc">{model.description}</div>
        <div className="model-badges">{modelBadges}</div>
      </div>
    </button>
  );
}

function shellStatusMessage(kind: StatusKind, message: string): React.ReactNode {
  const icon = kind === 'success' ? '✓' : kind === 'error' ? '!' : kind === 'loading' ? '…' : 'i';
  return (
    <div className={`status-bar ${kind === 'error' ? 'error' : kind === 'success' ? 'success' : kind === 'loading' ? 'loading' : ''}`}>
      <span className="status-icon">{icon}</span>
      <span className="status-text">{message}</span>
    </div>
  );
}

function GenerateSurface({ controller }: { controller: AppController }) {
  const {
    selectedModel,
    selectedModelStatus,
    selectedVoice,
    selectedVoiceId,
    voices,
    supportsWatermark,
    transcriptRequired,
    needsReferenceAudio,
    promptSource,
    promptText,
    setPromptText,
    text,
    setText,
    outputFormat,
    setOutputFormat,
    watermarkEnabled,
    setWatermarkEnabled,
    watermarkRun,
    setWatermarkRun,
    watermarkRuns,
    watermarkRecommendedThreshold,
    handleGenerate,
    handleSaveVoice,
    handleDeleteVoice,
    handleUnloadModel,
    handleResetAll,
    setPromptFile,
    setPromptRecording,
    clearVoicePreview,
    selectSavedVoice,
    setActiveSurface,
    setSelectedModelId,
    setEmotionFile,
    emotionFile,
    emotionFileRef,
    promptFileRef,
    promptSource: source,
    outputUrl,
    outputFileName,
    statusMessage,
    statusKind,
    isGenerating,
    settings,
    updateSettings,
  } = controller;

  const currentModelName = selectedModel ? normalizeModelName(selectedModel.name) : 'Select a model';
  const currentModelStatus = selectedModelStatus || {};
  const voiceCount = voices.length;
  const latestHistory = controller.history.slice(0, 3);

  const canDeleteVoice = !!selectedVoiceId;
  const needsTranscriptCopy = transcriptRequired ? 'Transcript required for this model.' : 'Transcript optional for this model.';
  const referenceCopy = needsReferenceAudio ? 'Reference audio or saved voice required.' : 'Reference audio optional.';

  return (
    <div className="surface surface-generate">
      <div className="surface-grid surface-grid-generate">
        <div className="surface-stack">
          <Panel
            title="Voice Reference"
            subtitle={referenceCopy}
            actions={<button className="btn btn-secondary" type="button" onClick={clearVoicePreview} disabled={!source || source.kind === 'none'}>Clear</button>}
          >
            <div className="form-row compact">
              <label className="form-label" htmlFor="savedVoiceSelect">Saved Voice</label>
              <select
                id="savedVoiceSelect"
                value={selectedVoiceId}
                onChange={(event) => void selectSavedVoice(event.currentTarget.value)}
              >
                <option value="">(Use uploaded/recorded)</option>
                {voices.map((voice) => (
                  <option key={voice.id} value={voice.id}>
                    {voice.name}
                    {voice.duration_s ? ` (${voice.duration_s.toFixed(1)}s)` : ''}
                  </option>
                ))}
              </select>
              <button className="btn btn-secondary" id="saveVoiceBtn" type="button" onClick={() => void handleSaveVoice()}>
                Save…
              </button>
              <button className="btn btn-danger" id="deleteVoiceBtn" type="button" onClick={() => void handleDeleteVoice()} disabled={!canDeleteVoice}>
                Delete
              </button>
            </div>
            <div className="form-hint">Saved voices persist on disk and can be reused without re-uploading.</div>

            <div className="voice-section">
              <div className="voice-method">
                <label className="file-upload" htmlFor="promptFile">
                  <div className="file-upload-icon">📁</div>
                  <div className="file-upload-text">Upload Audio</div>
                  <div className="file-upload-hint">WAV, MP3, or M4A</div>
                </label>
                <input
                  ref={promptFileRef}
                  type="file"
                  id="promptFile"
                  accept="audio/*"
                  onChange={(event) => void setPromptFile(event.currentTarget.files?.[0] || null)}
                />
              </div>
              <div className="voice-method">
                <div className="voice-method-icon">🎤</div>
                <div className="voice-method-title">Record Voice</div>
                <div className="voice-method-desc">Use your microphone</div>
                <button className={`record-btn${controller.isRecording ? ' recording' : ''}`} id="recToggle" type="button" onClick={() => void setPromptRecording()}>
                  {controller.isRecording ? '■' : '●'}
                </button>
                <div className="form-hint">{controller.isRecording ? `Recording... ${controller.recordingTime}` : 'One click recording to WAV'}</div>
              </div>
            </div>

            {source.kind !== 'none' ? (
              <div className="voice-preview">
                <audio id="promptPreview" className="audio-player" controls src={source.url} />
                <div className="form-hint" id="promptInfo">{source.info}</div>
              </div>
            ) : null}

            <div className="form-group">
              <label className="form-label" htmlFor="promptText">Reference Transcript</label>
              <textarea
                id="promptText"
                rows={3}
                value={promptText}
                onChange={(event) => setPromptText(event.currentTarget.value)}
                placeholder="Enter the exact text spoken in the reference audio"
              />
              <div className="form-hint">{needsTranscriptCopy}</div>
            </div>

            {selectedModel?.id === 'index-tts2' ? (
              <div className="form-group">
                <label className="form-label" htmlFor="emoFile">Emotion Reference Audio</label>
                <label className="file-upload" htmlFor="emoFile">
                  <div className="file-upload-text">Upload emotion audio</div>
                </label>
                <input
                  ref={emotionFileRef}
                  type="file"
                  id="emoFile"
                  accept="audio/*"
                  onChange={(event) => setEmotionFile(event.currentTarget.files?.[0] || null)}
                />
                <div className="form-hint">{emotionFile ? `Emotion file: ${emotionFile.name}` : 'Optional unless the emotion mode is set to emotion reference.'}</div>
              </div>
            ) : null}
          </Panel>

          <Panel title="Text to Speak" subtitle="Enter the text you want to synthesize">
            <div className="form-group">
              <textarea id="text" rows={5} value={text} onChange={(event) => setText(event.currentTarget.value)} placeholder="Type what you want the model to say..." />
            </div>
            <div className="form-row split">
              <div className="form-row compact">
                <label className="form-label" htmlFor="outputFormat">Output Format</label>
                <select id="outputFormat" value={outputFormat} onChange={(event) => setOutputFormat(event.currentTarget.value as 'wav' | 'mp3' | 'flac')}>
                  <option value="wav">WAV</option>
                  <option value="mp3">MP3</option>
                  <option value="flac">FLAC</option>
                </select>
              </div>
              <div className="action-row">
                <button className="btn btn-secondary" id="reset" type="button" onClick={() => void handleResetAll()}>
                  Reset All
                </button>
                <button className="btn btn-primary btn-lg" id="generate" type="button" onClick={() => void handleGenerate()} disabled={isGenerating}>
                  <span id="generateText">{isGenerating ? 'Generating…' : 'Generate'}</span>
                </button>
              </div>
            </div>
            <div className="status-bar-shell" id="statusBar">
              {shellStatusMessage(statusKind, statusMessage)}
            </div>
          </Panel>

          <Panel title="Output" subtitle="Generated audio will appear here">
            <div className="output-player">
              {outputUrl ? <audio id="output" className="audio-player" controls src={outputUrl} /> : <div className="output-placeholder" id="outputPlaceholder">No audio generated yet</div>}
              <div className={`output-actions${outputUrl ? '' : ' hidden'}`} id="outputActions">
                <a className="btn btn-primary" id="download" href={outputUrl || '#'} download={outputFileName}>
                  Download
                </a>
              </div>
            </div>
          </Panel>
        </div>

        <div className="surface-stack">
          <Panel
            title={currentModelName}
            subtitle={selectedModel?.description || 'Select a model to begin.'}
            actions={
              <>
                <button className="btn btn-secondary" type="button" onClick={() => setActiveSurface('models')}>Models</button>
                <button className="btn btn-secondary" type="button" onClick={() => setActiveSurface('advanced-settings')}>Advanced Settings</button>
              </>
            }
          >
            <div className="status-badges">
              <Badge tone={currentModelStatus.loaded ? 'success' : 'neutral'}>{currentModelStatus.loaded ? 'Loaded' : 'Ready'}</Badge>
              {currentModelStatus.device ? <Badge tone="neutral">{currentModelStatus.device}</Badge> : null}
              {typeof currentModelStatus.total_generations === 'number' ? <Badge tone="neutral">{currentModelStatus.total_generations} runs</Badge> : null}
              {supportsWatermark ? <Badge tone="info">Watermark Supported</Badge> : null}
            </div>
            <div className="summary-grid">
              <div className="summary-card">
                <div className="summary-label">Reference</div>
                <div className="summary-value">{selectedVoice ? selectedVoice.name : promptSource.kind !== 'none' ? 'Temporary reference' : 'No reference selected'}</div>
              </div>
              <div className="summary-card">
                <div className="summary-label">Clock</div>
                <div className="summary-value">{controller.systemClock}</div>
              </div>
              <div className="summary-card">
                <div className="summary-label">Voice Library</div>
                <div className="summary-value">{voiceCount} saved voices</div>
              </div>
            </div>
          </Panel>

          {selectedModel?.id === 'qwen3-tts-mlx' ? (
            <Panel title="Current Model Settings" subtitle="Qwen3-TTS MLX controls stay available from Generate for fast workflow parity">
              <div className="form-group">
                <label className="form-label" htmlFor="qwenModel">Model</label>
                <select id="qwenModel" value={settings.qwen.model} onChange={(event) => updateSettings('qwen', { ...settings.qwen, model: event.currentTarget.value })}>
                  <option value="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit">1.7B Base (8-bit, default)</option>
                  <option value="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit">0.6B Base (8-bit, faster)</option>
                </select>
              </div>
              <div className="checkbox-group">
                <input id="qwenAutoTranscribe" type="checkbox" checked={settings.qwen.autoTranscribe} onChange={(event) => updateSettings('qwen', { ...settings.qwen, autoTranscribe: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="qwenAutoTranscribe">Auto-transcribe reference audio if transcript is missing</label>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenLanguage">Language</label>
                <input id="qwenLanguage" type="text" value={settings.qwen.language} onChange={(event) => updateSettings('qwen', { ...settings.qwen, language: event.currentTarget.value })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenSpeed">Speed</label>
                <input id="qwenSpeed" type="number" value={settings.qwen.speed} onChange={(event) => updateSettings('qwen', { ...settings.qwen, speed: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenTemperature">Temperature</label>
                <input id="qwenTemperature" type="number" value={settings.qwen.temperature} onChange={(event) => updateSettings('qwen', { ...settings.qwen, temperature: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenMaxTokens">Max Tokens</label>
                <input id="qwenMaxTokens" type="number" value={settings.qwen.maxTokens} onChange={(event) => updateSettings('qwen', { ...settings.qwen, maxTokens: Number(event.currentTarget.value) })} />
              </div>
            </Panel>
          ) : null}

          <Panel title="Watermark Summary" subtitle="Embed provenance without leaving the Generate surface">
            <div className="form-group">
              <div className="checkbox-group">
                <input id="watermarkEnabled" type="checkbox" checked={watermarkEnabled} onChange={(event) => controller.setWatermarkEnabled(event.currentTarget.checked)} disabled={!supportsWatermark || !watermarkRuns.length} />
                <label className="checkbox-label" htmlFor="watermarkEnabled">Embed watermark in output</label>
              </div>
              <div className="form-hint" id="watermarkHint">
                {supportsWatermark ? 'Supported models: IndexTTS2 and Qwen3-TTS MLX.' : 'Watermarking is not supported for the selected model.'}
              </div>
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="watermarkRun">Watermark run</label>
              <select id="watermarkRun" value={watermarkRun} onChange={(event) => void setWatermarkRun(event.currentTarget.value)}>
                <option value="">Auto (latest completed)</option>
                {watermarkRuns.map((run) => (
                  <option key={run.id} value={run.id}>
                    {run.label || run.id}
                    {run.status ? ` · ${run.status}` : ''}
                  </option>
                ))}
              </select>
              <div className="form-hint">Pick a completed run with `encoder.pt` + `decoder.pt`.</div>
            </div>
            <pre className="output-placeholder" id="watermarkRunDetails">{controller.watermarkRunDetails?.report_excerpt || controller.watermarkRunDetails?.error || 'Select a run to see details.'}</pre>
            <div className="form-hint">
              {typeof watermarkRecommendedThreshold === 'number'
                ? `Recommended threshold: ${watermarkRecommendedThreshold.toFixed(3)}`
                : 'Threshold will default to the watermark run recommendation or 0.35.'}
            </div>
          </Panel>

          <Panel title="Recent History" subtitle="Latest outputs from this session">
            <div className="history-list">
              {latestHistory.length ? latestHistory.map((item) => (
                <HistoryRow key={`${item.modelId}-${item.timestamp}`} item={item} modelName={normalizeModelName(controller.models.find((model) => model.id === item.modelId)?.name || item.modelId)} />
              )) : <div className="history-empty" id="historyEmpty">No generations yet</div>}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function ModelsSurface({ controller }: { controller: AppController }) {
  const { models, modelStatuses, selectedModelId, setSelectedModelId, setActiveSurface, handleUnloadModel } = controller;
  const selectedModel = models.find((model) => model.id === selectedModelId) || null;
  const status = selectedModel ? modelStatuses[selectedModel.id] || {} : {};

  return (
    <div className="surface surface-models">
      <div className="surface-grid surface-grid-two">
        <div className="surface-stack">
          <Panel title="Model Selection" subtitle="Choose a model, inspect readiness, and keep load/unload actions close to the model card">
            <div className="model-nav model-nav-surface">
              {models.map((model) => (
                <ModelCard
                  key={model.id}
                  model={model}
                  status={modelStatuses[model.id] || {}}
                  active={selectedModelId === model.id}
                  onClick={setSelectedModelId}
                />
              ))}
            </div>
          </Panel>
        </div>
        <div className="surface-stack">
          <Panel
            title={selectedModel ? normalizeModelName(selectedModel.name) : 'No model selected'}
            subtitle={selectedModel?.description || 'Pick a model to see detail.'}
            actions={
              <>
                <button className="btn btn-secondary" type="button" onClick={() => setActiveSurface('generate')}>Generate</button>
                <button className="btn btn-danger" type="button" onClick={() => void handleUnloadModel()} disabled={!selectedModelId}>Unload</button>
              </>
            }
          >
            <div className="status-badges">
              <Badge tone={status.loaded ? 'success' : 'neutral'}>{status.loaded ? 'Loaded' : 'Ready'}</Badge>
              {status.device ? <Badge tone="neutral">{status.device}</Badge> : null}
              {typeof status.total_generations === 'number' ? <Badge tone="neutral">{status.total_generations} runs</Badge> : null}
              {selectedModel?.id && isWatermarkSupported(selectedModel.id) ? <Badge tone="info">Watermark</Badge> : null}
            </div>
            <div className="summary-grid">
              <div className="summary-card">
                <div className="summary-label">Worker</div>
                <div className="summary-value">{selectedModel?.worker_entry || 'Internal worker'}</div>
              </div>
              <div className="summary-card">
                <div className="summary-label">Last run</div>
                <div className="summary-value">{typeof status.last_generation_duration_ms === 'number' ? `${Math.round(status.last_generation_duration_ms)} ms` : 'n/a'}</div>
              </div>
              <div className="summary-card">
                <div className="summary-label">Total generations</div>
                <div className="summary-value">{typeof status.total_generations === 'number' ? status.total_generations : 0}</div>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function VoicesSurface({ controller }: { controller: AppController }) {
  const { voices, selectedVoiceId, selectSavedVoice, handleSaveVoice, handleDeleteVoice, promptSource, promptText, setPromptText, clearVoicePreview, setPromptFile, setPromptRecording, promptFileRef, emotionFileRef } = controller;
  const selectedVoice = voices.find((voice) => voice.id === selectedVoiceId) || null;

  return (
    <div className="surface surface-voices">
      <div className="surface-grid surface-grid-two">
        <div className="surface-stack">
          <Panel title="Voice Input" subtitle="Upload, record, preview, and save reference voices">
            <div className="voice-section">
              <div className="voice-method">
                <label className="file-upload" htmlFor="promptFile">
                  <div className="file-upload-icon">📁</div>
                  <div className="file-upload-text">Upload Audio</div>
                  <div className="file-upload-hint">WAV, MP3, or M4A</div>
                </label>
                <input ref={promptFileRef} type="file" id="promptFile" accept="audio/*" onChange={(event) => void setPromptFile(event.currentTarget.files?.[0] || null)} />
              </div>
              <div className="voice-method">
                <div className="voice-method-icon">🎤</div>
                <div className="voice-method-title">Record Voice</div>
                <div className="voice-method-desc">Use your microphone</div>
                <button className={`record-btn${controller.isRecording ? ' recording' : ''}`} id="recToggle" type="button" onClick={() => void setPromptRecording()}>
                  {controller.isRecording ? '■' : '●'}
                </button>
              </div>
            </div>

            {promptSource.kind !== 'none' ? (
              <div className="voice-preview">
                <audio id="promptPreview" className="audio-player" controls src={promptSource.url} />
                <div className="form-hint" id="promptInfo">{promptSource.info}</div>
              </div>
            ) : null}

            <div className="form-row compact">
              <label className="form-label" htmlFor="savedVoiceSelect">Saved Voice</label>
              <select id="savedVoiceSelect" value={selectedVoiceId} onChange={(event) => void selectSavedVoice(event.currentTarget.value)}>
                <option value="">(Use uploaded/recorded)</option>
                {voices.map((voice) => (
                  <option key={voice.id} value={voice.id}>{voice.name}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="promptText">Reference Transcript</label>
              <textarea id="promptText" rows={3} value={promptText} onChange={(event) => setPromptText(event.currentTarget.value)} />
              <div className="form-hint">Saved for reuse in transcript-aware models.</div>
            </div>

            <div className="action-row">
              <button className="btn btn-secondary" id="saveVoiceBtn" type="button" onClick={() => void handleSaveVoice()}>Save voice</button>
              <button className="btn btn-danger" id="deleteVoiceBtn" type="button" onClick={() => void handleDeleteVoice()} disabled={!selectedVoiceId}>Delete voice</button>
              <button className="btn btn-secondary" type="button" onClick={() => void clearVoicePreview()}>Clear</button>
            </div>
          </Panel>
        </div>
        <div className="surface-stack">
          <Panel title="Saved Voices" subtitle={`${voices.length} reusable voice references`}>
            <div className="voice-library">
              {voices.length ? voices.map((voice) => (
                <div key={voice.id} className={`voice-card${selectedVoiceId === voice.id ? ' active' : ''}`}>
                  <div className="voice-card-title">{voice.name}</div>
                  <div className="voice-card-meta">{formatDuration(voice.duration_s)} · {formatDateTime(voice.created_at)}</div>
                  <div className="model-badges">
                    {voice.has_caches?.['qwen3-tts-mlx'] ? <Badge tone="info">Qwen cache</Badge> : null}
                    {voice.has_caches?.['index-tts2'] ? <Badge tone="info">Index cache</Badge> : null}
                    {voice.has_caches?.['chatterbox-multilingual'] ? <Badge tone="info">Chatterbox cache</Badge> : null}
                  </div>
                  <div className="action-row">
                    <button className="btn btn-secondary" type="button" onClick={() => void selectSavedVoice(voice.id)}>Use</button>
                  </div>
                </div>
              )) : <div className="history-empty">No saved voices yet.</div>}
            </div>
            {selectedVoice ? <div className="form-hint">Selected voice: {selectedVoice.name}</div> : null}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function HistorySurface({ controller }: { controller: AppController }) {
  const { history, models, setActiveSurface, outputUrl, outputFileName } = controller;

  return (
    <div className="surface surface-history">
      <div className="surface-grid surface-grid-two">
        <div className="surface-stack">
          <Panel title="Generation History" subtitle="Session-oriented review of recent outputs">
            <div className="history-list">
              {history.length ? history.map((item) => (
                <HistoryRow key={`${item.modelId}-${item.timestamp}`} item={item} modelName={normalizeModelName(models.find((model) => model.id === item.modelId)?.name || item.modelId)} />
              )) : <div className="history-empty" id="historyEmpty">No generations yet</div>}
            </div>
          </Panel>
        </div>
        <div className="surface-stack">
          <Panel title="Current Output" subtitle="Replay the most recent render without leaving history">
            {outputUrl ? <audio id="output" className="audio-player" controls src={outputUrl} /> : <div className="output-placeholder" id="outputPlaceholder">No audio generated yet</div>}
            <div className={`output-actions${outputUrl ? '' : ' hidden'}`} id="outputActions">
              <a className="btn btn-primary" id="download" href={outputUrl || '#'} download={outputFileName}>Download</a>
            </div>
            <div className="action-row">
              <button className="btn btn-secondary" type="button" onClick={() => setActiveSurface('generate')}>Back to Generate</button>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function WatermarkLabSurface({ controller }: { controller: AppController }) {
  const {
    watermarkEnabled,
    setWatermarkEnabled,
    watermarkRuns,
    watermarkRun,
    setWatermarkRun,
    watermarkRunDetails,
    watermarkRecommendedThreshold,
    watermarkThresholdAuto,
    setWatermarkThresholdAuto,
    watermarkThresholdManual,
    setWatermarkThresholdManual,
    watermarkDetectFile,
    watermarkDetectFileInfo,
    watermarkDetectPreviewUrl,
    watermarkDetectResult,
    setWatermarkDetectFile,
    handleWatermarkDetect,
    watermarkDetectFileRef,
    supportsWatermark,
  } = controller;

  return (
    <div className="surface surface-watermark">
      <div className="surface-grid surface-grid-two">
        <div className="surface-stack">
          <Panel title="Watermark Embed" subtitle="Keep provenance controls separate from the generate flow">
            <div className="form-group">
              <div className="checkbox-group">
                <input id="watermarkEnabled" type="checkbox" checked={watermarkEnabled} onChange={(event) => setWatermarkEnabled(event.currentTarget.checked)} disabled={!supportsWatermark || !watermarkRuns.length} />
                <label className="checkbox-label" htmlFor="watermarkEnabled">Embed watermark in output</label>
              </div>
              <div className="form-hint" id="watermarkHint">{supportsWatermark ? 'Supported for IndexTTS2 and Qwen3-TTS MLX.' : 'Watermark embedding is unavailable for this model.'}</div>
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="watermarkRun">Watermark run</label>
              <select id="watermarkRun" value={watermarkRun} onChange={(event) => void setWatermarkRun(event.currentTarget.value)}>
                <option value="">Auto (latest completed)</option>
                {watermarkRuns.map((run) => (
                  <option key={run.id} value={run.id}>{run.label || run.id}{run.status ? ` · ${run.status}` : ''}</option>
                ))}
              </select>
              <div className="form-hint">Pick a run with `encoder.pt` and `decoder.pt`.</div>
            </div>
            <pre className="output-placeholder" id="watermarkRunDetails">
              {watermarkRunDetails?.report_excerpt || watermarkRunDetails?.error || 'Select a run to see details.'}
            </pre>
            <div className="form-hint">
              {typeof watermarkRecommendedThreshold === 'number' ? `Recommended threshold: ${watermarkRecommendedThreshold.toFixed(3)}` : 'Threshold recommendations will appear once a run is selected.'}
            </div>
          </Panel>
        </div>
        <div className="surface-stack">
          <Panel title="Watermark Detection" subtitle="Upload audio to detect watermark and attribute the source model">
            <label className="file-upload" id="watermarkTestDropZone" htmlFor="watermarkTestFile" title="Click to choose a file">
              <div className="file-upload-icon">🧪</div>
              <div className="file-upload-text">Choose Audio File</div>
              <div className="file-upload-hint">WAV, MP3, FLAC, M4A…</div>
            </label>
            <input
              ref={watermarkDetectFileRef}
              type="file"
              id="watermarkTestFile"
              accept="audio/*"
              onChange={(event) => void setWatermarkDetectFile(event.currentTarget.files?.[0] || null)}
            />
            <div className="form-hint" id="watermarkTestFileInfo">{watermarkDetectFileInfo}</div>
            {watermarkDetectPreviewUrl ? <audio id="watermarkTestPreview" className="audio-player" controls src={watermarkDetectPreviewUrl} /> : null}
            <div className="form-row split">
              <div className="form-row compact">
                <div className="checkbox-group">
                  <input id="watermarkThresholdAuto" type="checkbox" checked={watermarkThresholdAuto} onChange={(event) => setWatermarkThresholdAuto(event.currentTarget.checked)} />
                  <label className="checkbox-label" htmlFor="watermarkThresholdAuto">Auto threshold</label>
                </div>
                <label className="form-label" htmlFor="watermarkThreshold">Threshold</label>
                <input
                  id="watermarkThreshold"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={watermarkThresholdManual}
                  disabled={watermarkThresholdAuto}
                  onChange={(event) => setWatermarkThresholdManual(event.currentTarget.value)}
                />
              </div>
              <button className="btn btn-secondary" id="watermarkTestBtn" type="button" onClick={() => void handleWatermarkDetect()}>Analyze</button>
            </div>
            <div className="form-hint" id="watermarkThresholdHint">
              {watermarkThresholdAuto ? (typeof watermarkRecommendedThreshold === 'number' ? `Auto threshold: ${watermarkRecommendedThreshold.toFixed(3)}` : 'Auto threshold enabled.') : 'Manual threshold enabled.'}
            </div>
            <pre className="output-placeholder" id="watermarkTestResult">{watermarkDetectResult}</pre>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function SystemStatusSurface({ controller }: { controller: AppController }) {
  const {
    infoAvailable,
    systemClock,
    models,
    modelStatuses,
    selectedSurfaceTitle,
    activeSurface,
    setActiveSurface,
  } = {
    infoAvailable: controller.infoAvailable,
    systemClock: controller.systemClock,
    models: controller.models,
    modelStatuses: controller.modelStatuses,
    selectedSurfaceTitle: controller.surfaceTitle,
    activeSurface: controller.activeSurface,
    setActiveSurface: controller.setActiveSurface,
  };

  const loadedCount = Object.values(modelStatuses).filter((status) => !!status.loaded).length;

  return (
    <div className="surface surface-status">
      <div className="status-grid">
        <div className="status-card">
          <div className="status-card-label">ffmpeg</div>
          <div className="status-card-value">{infoAvailable === null ? 'Checking...' : infoAvailable ? 'Available' : 'Missing'}</div>
          <div className="status-card-meta">Runtime conversion support</div>
        </div>
        <div className="status-card">
          <div className="status-card-label">Active Surface</div>
          <div className="status-card-value">{selectedSurfaceTitle}</div>
          <div className="status-card-meta">Current workspace section</div>
        </div>
        <div className="status-card">
          <div className="status-card-label">Clock</div>
          <div className="status-card-value">{systemClock}</div>
          <div className="status-card-meta">Local system time snapshot</div>
        </div>
        <div className="status-card">
          <div className="status-card-label">Loaded Models</div>
          <div className="status-card-value">{loadedCount}</div>
          <div className="status-card-meta">Workers currently alive</div>
        </div>
      </div>

      <Panel title="Per-Model Status" subtitle="Load state, device, and run counts for each backend worker">
        <div className="system-model-table">
          {models.map((model) => {
            const status = modelStatuses[model.id] || {};
            return (
              <div key={model.id} className="system-model-row">
                <div>
                  <div className="system-model-name">{normalizeModelName(model.name)}</div>
                  <div className="system-model-meta">{model.description}</div>
                </div>
                <div className="system-model-stat">{status.loaded ? 'Loaded' : 'Idle'} / {status.device || 'unknown'}</div>
                <div className="system-model-stat">{typeof status.total_generations === 'number' ? status.total_generations : 0} runs</div>
                <div className="system-model-stat">
                  <button className="btn btn-secondary" type="button" onClick={() => setActiveSurface('models')}>Inspect</button>
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}

function AdvancedSettingsSurface({ controller }: { controller: AppController }) {
  const { selectedModel, selectedModelId, settings, updateSettings, setActiveSurface } = controller;

  const currentModel = selectedModelId || selectedModel?.id || 'index-tts2';

  return (
    <div className="surface surface-advanced">
      <div className="surface-grid surface-grid-two">
        <div className="surface-stack">
          <Panel title="Advanced Settings" subtitle="Model-specific controls are grouped and collapsed by default">
            <div className="form-group">
              <label className="form-label" htmlFor="advancedModelSelect">Current Model</label>
              <select id="advancedModelSelect" value={currentModel} onChange={(event) => controller.setSelectedModelId(event.currentTarget.value)}>
                {controller.models.map((model) => (
                  <option key={model.id} value={model.id}>{normalizeModelName(model.name)}</option>
                ))}
              </select>
            </div>
            <div className="form-hint">Keep the selected model in sync with Generate; this surface only changes the presentation of the same values.</div>
          </Panel>
        </div>

        <div className="surface-stack">
          {currentModel === 'index-tts2' ? (
            <Panel title="IndexTTS2 Settings" subtitle="Emotion control and sampling" actions={<button className="btn btn-secondary" type="button" onClick={() => setActiveSurface('generate')}>Generate</button>}>
              <div className="form-group">
                <label className="form-label" htmlFor="indexEmoMode">Emotion Mode</label>
                <select id="indexEmoMode" value={settings.index.emoMode} onChange={(event) => updateSettings('index', { ...settings.index, emoMode: event.currentTarget.value as AppSettings['index']['emoMode'] })}>
                  <option value="speaker">Same as speaker (default)</option>
                  <option value="emo_ref">Use emotion reference audio</option>
                  <option value="emo_vector">Custom emotion vector</option>
                  <option value="emo_text">Emotion from text description</option>
                </select>
              </div>
              <div className="slider-group">
                <span className="slider-label">Emotion Weight</span>
                <div className="slider-container">
                  <input id="indexEmoAlpha" type="range" min="0" max="1" step="0.01" value={settings.index.emoAlpha} onChange={(event) => updateSettings('index', { ...settings.index, emoAlpha: Number(event.currentTarget.value) })} />
                  <span className="slider-value" id="indexEmoAlphaVal">{settings.index.emoAlpha.toFixed(2)}</span>
                </div>
              </div>
              <div className="checkbox-group">
                <input id="indexUseRandom" type="checkbox" checked={settings.index.useRandom} onChange={(event) => updateSettings('index', { ...settings.index, useRandom: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="indexUseRandom">Random sampling</label>
              </div>
              {settings.index.emoMode === 'emo_vector' ? (
                <div className="form-group">
                  <label className="form-label" htmlFor="indexEmoVector">Emotion Vector</label>
                  <input id="indexEmoVector" type="text" value={settings.index.emoVector} onChange={(event) => updateSettings('index', { ...settings.index, emoVector: event.currentTarget.value })} />
                </div>
              ) : null}
              {settings.index.emoMode === 'emo_text' ? (
                <div className="form-group">
                  <label className="form-label" htmlFor="indexEmoText">Emotion Description</label>
                  <textarea id="indexEmoText" rows={2} value={settings.index.emoText} onChange={(event) => updateSettings('index', { ...settings.index, emoText: event.currentTarget.value })} />
                </div>
              ) : null}
              <details className="settings-details" open>
                <summary>Segmentation</summary>
                <div className="details-body">
                  <div className="form-group">
                    <label className="form-label" htmlFor="indexMaxTextTokens">Text Tokens / Segment</label>
                    <input id="indexMaxTextTokens" type="number" value={settings.index.maxTextTokens} onChange={(event) => updateSettings('index', { ...settings.index, maxTextTokens: Number(event.currentTarget.value) })} />
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="indexMaxMelTokens">Max Mel Tokens</label>
                    <input id="indexMaxMelTokens" type="number" value={settings.index.maxMelTokens} onChange={(event) => updateSettings('index', { ...settings.index, maxMelTokens: Number(event.currentTarget.value) })} />
                  </div>
                  <div className="checkbox-group">
                    <input id="indexFastMode" type="checkbox" checked={settings.index.fastMode} onChange={(event) => updateSettings('index', { ...settings.index, fastMode: event.currentTarget.checked })} />
                    <label className="checkbox-label" htmlFor="indexFastMode">Fast mode (greedy decoding)</label>
                  </div>
                </div>
              </details>
              <details className="settings-details" open>
                <summary>Advanced Sampling</summary>
                <div className="details-body">
                  <div className="checkbox-group">
                    <input id="indexDoSample" type="checkbox" checked={settings.index.doSample} onChange={(event) => updateSettings('index', { ...settings.index, doSample: event.currentTarget.checked })} />
                    <label className="checkbox-label" htmlFor="indexDoSample">do_sample</label>
                  </div>
                  <div className="slider-group">
                    <span className="slider-label">Temperature</span>
                    <div className="slider-container">
                      <input id="indexTemperature" type="range" min="0.1" max="2.0" step="0.1" value={settings.index.temperature} onChange={(event) => updateSettings('index', { ...settings.index, temperature: Number(event.currentTarget.value) })} />
                      <span className="slider-value" id="indexTemperatureVal">{settings.index.temperature.toFixed(1)}</span>
                    </div>
                  </div>
                  <div className="slider-group">
                    <span className="slider-label">Top P</span>
                    <div className="slider-container">
                      <input id="indexTopP" type="range" min="0" max="1" step="0.05" value={settings.index.topP} onChange={(event) => updateSettings('index', { ...settings.index, topP: Number(event.currentTarget.value) })} />
                      <span className="slider-value" id="indexTopPVal">{settings.index.topP.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="indexTopK">Top K</label>
                    <input id="indexTopK" type="number" value={settings.index.topK} onChange={(event) => updateSettings('index', { ...settings.index, topK: Number(event.currentTarget.value) })} />
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="indexNumBeams">Num Beams</label>
                    <input id="indexNumBeams" type="number" value={settings.index.numBeams} onChange={(event) => updateSettings('index', { ...settings.index, numBeams: Number(event.currentTarget.value) })} />
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="indexRepetitionPenalty">Repetition Penalty</label>
                    <input id="indexRepetitionPenalty" type="number" value={settings.index.repetitionPenalty} onChange={(event) => updateSettings('index', { ...settings.index, repetitionPenalty: Number(event.currentTarget.value) })} />
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="indexLengthPenalty">Length Penalty</label>
                    <input id="indexLengthPenalty" type="number" value={settings.index.lengthPenalty} onChange={(event) => updateSettings('index', { ...settings.index, lengthPenalty: Number(event.currentTarget.value) })} />
                  </div>
                </div>
              </details>
            </Panel>
          ) : null}

          {currentModel === 'chatterbox-multilingual' ? (
            <Panel title="Chatterbox Multilingual Settings" subtitle="Language, chunking, and enhancement controls">
              <div className="form-group">
                <label className="form-label" htmlFor="cbLanguage">Language</label>
                <select id="cbLanguage" value={settings.chatterbox.language} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, language: event.currentTarget.value })}>
                  <option value="hi">Hindi (hi)</option>
                  <option value="en">English (en)</option>
                  <option value="zh">Chinese (zh)</option>
                  <option value="ar">Arabic (ar)</option>
                  <option value="fr">French (fr)</option>
                  <option value="es">Spanish (es)</option>
                  <option value="de">German (de)</option>
                  <option value="ja">Japanese (ja)</option>
                  <option value="ko">Korean (ko)</option>
                  <option value="it">Italian (it)</option>
                  <option value="pt">Portuguese (pt)</option>
                  <option value="ru">Russian (ru)</option>
                  <option value="tr">Turkish (tr)</option>
                </select>
              </div>
              <div className="checkbox-group">
                <input id="cbUsePrompt" type="checkbox" checked={settings.chatterbox.usePrompt} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, usePrompt: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="cbUsePrompt">Use reference audio for cloning</label>
              </div>
              <details className="settings-details" open>
                <summary>Long-form (Chunking)</summary>
                <div className="details-body">
                  <div className="slider-group">
                    <span className="slider-label">CFG Weight</span>
                    <div className="slider-container">
                      <input id="cbCfgWeight" type="range" min="0" max="1" step="0.05" value={settings.chatterbox.cfgWeight} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, cfgWeight: Number(event.currentTarget.value) })} />
                      <span className="slider-value" id="cbCfgWeightVal">{settings.chatterbox.cfgWeight.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="slider-group">
                    <span className="slider-label">Temperature</span>
                    <div className="slider-container">
                      <input id="cbTemperature" type="range" min="0.1" max="2.0" step="0.1" value={settings.chatterbox.temperature} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, temperature: Number(event.currentTarget.value) })} />
                      <span className="slider-value" id="cbTemperatureVal">{settings.chatterbox.temperature.toFixed(1)}</span>
                    </div>
                  </div>
                  <div className="slider-group">
                    <span className="slider-label">Exaggeration</span>
                    <div className="slider-container">
                      <input id="cbExaggeration" type="range" min="0" max="1" step="0.05" value={settings.chatterbox.exaggeration} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, exaggeration: Number(event.currentTarget.value) })} />
                      <span className="slider-value" id="cbExaggerationVal">{settings.chatterbox.exaggeration.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="checkbox-group">
                    <input id="cbFastMode" type="checkbox" checked={settings.chatterbox.fastMode} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, fastMode: event.currentTarget.checked })} />
                    <label className="checkbox-label" htmlFor="cbFastMode">Fast mode</label>
                  </div>
                  <div className="checkbox-group">
                    <input id="cbChunking" type="checkbox" checked={settings.chatterbox.enableChunking} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, enableChunking: event.currentTarget.checked })} />
                    <label className="checkbox-label" htmlFor="cbChunking">Enable chunking</label>
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="cbMaxChunkChars">Max Chars</label>
                    <input id="cbMaxChunkChars" type="number" value={settings.chatterbox.maxChunkChars} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, maxChunkChars: Number(event.currentTarget.value) })} />
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="cbCrossfade">Crossfade (ms)</label>
                    <input id="cbCrossfade" type="number" value={settings.chatterbox.crossfadeMs} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, crossfadeMs: Number(event.currentTarget.value) })} />
                  </div>
                  <div className="checkbox-group">
                    <input id="cbEnableDf" type="checkbox" checked={settings.chatterbox.enableDf} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, enableDf: event.currentTarget.checked })} />
                    <label className="checkbox-label" htmlFor="cbEnableDf">DeepFilterNet (noise removal)</label>
                  </div>
                  <div className="checkbox-group">
                    <input id="cbEnableNovasr" type="checkbox" checked={settings.chatterbox.enableNovasr} onChange={(event) => updateSettings('chatterbox', { ...settings.chatterbox, enableNovasr: event.currentTarget.checked })} />
                    <label className="checkbox-label" htmlFor="cbEnableNovasr">NovaSR (48kHz upscaling)</label>
                  </div>
                </div>
              </details>
            </Panel>
          ) : null}

          {currentModel === 'f5-hindi-urdu' ? (
            <Panel title="F5 Hindi/Urdu Settings" subtitle="Roman mode, overrides, and synthesis controls">
              <div className="checkbox-group">
                <input id="f5RomanMode" type="checkbox" checked={settings.f5.romanMode} onChange={(event) => updateSettings('f5', { ...settings.f5, romanMode: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="f5RomanMode">Roman input (converts to Devanagari)</label>
              </div>
              <div className="checkbox-group">
                <input id="f5OverridesEnabled" type="checkbox" checked={settings.f5.overridesEnabled} onChange={(event) => updateSettings('f5', { ...settings.f5, overridesEnabled: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="f5OverridesEnabled">Enable pronunciation overrides</label>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="f5OverridesText">Override Rules</label>
                <textarea id="f5OverridesText" rows={4} value={settings.f5.overridesText} onChange={(event) => updateSettings('f5', { ...settings.f5, overridesText: event.currentTarget.value })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="f5CrossFade">Cross-fade Duration</label>
                <input id="f5CrossFade" type="number" value={settings.f5.crossFade} onChange={(event) => updateSettings('f5', { ...settings.f5, crossFade: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="f5NfeStep">NFE Steps</label>
                <input id="f5NfeStep" type="number" value={settings.f5.nfeStep} onChange={(event) => updateSettings('f5', { ...settings.f5, nfeStep: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="f5Speed">Speed</label>
                <input id="f5Speed" type="number" value={settings.f5.speed} onChange={(event) => updateSettings('f5', { ...settings.f5, speed: Number(event.currentTarget.value) })} />
              </div>
              <div className="checkbox-group">
                <input id="f5RemoveSilence" type="checkbox" checked={settings.f5.removeSilence} onChange={(event) => updateSettings('f5', { ...settings.f5, removeSilence: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="f5RemoveSilence">Remove silence</label>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="f5Seed">Seed</label>
                <input id="f5Seed" type="number" value={settings.f5.seed} onChange={(event) => updateSettings('f5', { ...settings.f5, seed: Number(event.currentTarget.value) })} />
              </div>
            </Panel>
          ) : null}

          {currentModel === 'cosyvoice3-mlx' ? (
            <Panel title="CosyVoice3-MLX Settings" subtitle="Mode switching and transcript requirements">
              <div className="form-group">
                <label className="form-label" htmlFor="cosyModel">Model Variant</label>
                <select id="cosyModel" value={settings.cosy.model} onChange={(event) => updateSettings('cosy', { ...settings.cosy, model: event.currentTarget.value })}>
                  <option value="8bit">8-bit (recommended)</option>
                  <option value="4bit">4-bit (faster)</option>
                  <option value="fp16">FP16 (highest quality)</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="cosyMode">Mode</label>
                <select id="cosyMode" value={settings.cosy.mode} onChange={(event) => updateSettings('cosy', { ...settings.cosy, mode: event.currentTarget.value as AppSettings['cosy']['mode'] })}>
                  <option value="zero_shot">Zero-shot (best quality)</option>
                  <option value="cross_lingual">Cross-lingual</option>
                  <option value="instruct">Instruct</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="cosyLang">Language</label>
                <input id="cosyLang" type="text" value={settings.cosy.language} onChange={(event) => updateSettings('cosy', { ...settings.cosy, language: event.currentTarget.value })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="cosySpeed">Speed</label>
                <input id="cosySpeed" type="number" value={settings.cosy.speed} onChange={(event) => updateSettings('cosy', { ...settings.cosy, speed: Number(event.currentTarget.value) })} />
              </div>
              {settings.cosy.mode === 'instruct' ? (
                <div className="form-group" id="cosyInstructRow">
                  <label className="form-label" htmlFor="cosyInstructText">Instruction</label>
                  <textarea id="cosyInstructText" rows={2} value={settings.cosy.instructText} onChange={(event) => updateSettings('cosy', { ...settings.cosy, instructText: event.currentTarget.value })} />
                </div>
              ) : null}
            </Panel>
          ) : null}

          {currentModel === 'qwen3-tts-mlx' ? (
            <Panel title="Qwen3-TTS MLX Settings" subtitle="Voice cloning and auto-transcribe behavior">
              <div className="form-group">
                <label className="form-label" htmlFor="qwenModel">Model</label>
                <select id="qwenModel" value={settings.qwen.model} onChange={(event) => updateSettings('qwen', { ...settings.qwen, model: event.currentTarget.value })}>
                  <option value="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit">1.7B Base (8-bit, default)</option>
                  <option value="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit">0.6B Base (8-bit, faster)</option>
                </select>
              </div>
              <div className="checkbox-group">
                <input id="qwenAutoTranscribe" type="checkbox" checked={settings.qwen.autoTranscribe} onChange={(event) => updateSettings('qwen', { ...settings.qwen, autoTranscribe: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="qwenAutoTranscribe">Auto-transcribe reference audio if transcript is missing</label>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenLanguage">Language</label>
                <input id="qwenLanguage" type="text" value={settings.qwen.language} onChange={(event) => updateSettings('qwen', { ...settings.qwen, language: event.currentTarget.value })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenSpeed">Speed</label>
                <input id="qwenSpeed" type="number" value={settings.qwen.speed} onChange={(event) => updateSettings('qwen', { ...settings.qwen, speed: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenTemperature">Temperature</label>
                <input id="qwenTemperature" type="number" value={settings.qwen.temperature} onChange={(event) => updateSettings('qwen', { ...settings.qwen, temperature: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="qwenMaxTokens">Max Tokens</label>
                <input id="qwenMaxTokens" type="number" value={settings.qwen.maxTokens} onChange={(event) => updateSettings('qwen', { ...settings.qwen, maxTokens: Number(event.currentTarget.value) })} />
              </div>
            </Panel>
          ) : null}

          {currentModel === 'pocket-tts' ? (
            <Panel title="Pocket TTS Settings" subtitle="Lightweight TTS and decoding controls">
              <div className="form-group">
                <label className="form-label" htmlFor="pocketVoice">Voice URL</label>
                <input id="pocketVoice" type="text" value={settings.pocket.voice} onChange={(event) => updateSettings('pocket', { ...settings.pocket, voice: event.currentTarget.value })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="pocketTemp">Temperature</label>
                <input id="pocketTemp" type="number" value={settings.pocket.temperature} onChange={(event) => updateSettings('pocket', { ...settings.pocket, temperature: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="pocketLsd">LSD Decode Steps</label>
                <input id="pocketLsd" type="number" value={settings.pocket.lsdDecodeSteps} onChange={(event) => updateSettings('pocket', { ...settings.pocket, lsdDecodeSteps: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="pocketEos">EOS Threshold</label>
                <input id="pocketEos" type="number" value={settings.pocket.eosThreshold} onChange={(event) => updateSettings('pocket', { ...settings.pocket, eosThreshold: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="pocketNoiseClamp">Noise Clamp</label>
                <input id="pocketNoiseClamp" type="number" value={settings.pocket.noiseClamp} onChange={(event) => updateSettings('pocket', { ...settings.pocket, noiseClamp: event.currentTarget.value })} />
              </div>
              <div className="checkbox-group">
                <input id="pocketTruncate" type="checkbox" checked={settings.pocket.truncatePrompt} onChange={(event) => updateSettings('pocket', { ...settings.pocket, truncatePrompt: event.currentTarget.checked })} />
                <label className="checkbox-label" htmlFor="pocketTruncate">Truncate prompt</label>
              </div>
            </Panel>
          ) : null}

          {currentModel === 'voxcpm-ane' ? (
            <Panel title="VoxCPM-ANE Settings" subtitle="Cached voice or prompt audio plus inference parameters">
              <div className="form-group">
                <label className="form-label" htmlFor="voxcpmVoice">Cached Voice Name</label>
                <input id="voxcpmVoice" type="text" value={settings.voxcpm.voice} onChange={(event) => updateSettings('voxcpm', { ...settings.voxcpm, voice: event.currentTarget.value })} />
                <div className="form-hint">Enter a cached voice name, or use reference audio with transcript.</div>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="voxcpmCfg">CFG Value</label>
                <input id="voxcpmCfg" type="number" value={settings.voxcpm.cfgValue} onChange={(event) => updateSettings('voxcpm', { ...settings.voxcpm, cfgValue: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="voxcpmSteps">Inference Steps</label>
                <input id="voxcpmSteps" type="number" value={settings.voxcpm.inferenceTimesteps} onChange={(event) => updateSettings('voxcpm', { ...settings.voxcpm, inferenceTimesteps: Number(event.currentTarget.value) })} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="voxcpmMaxLen">Max Length</label>
                <input id="voxcpmMaxLen" type="number" value={settings.voxcpm.maxLength} onChange={(event) => updateSettings('voxcpm', { ...settings.voxcpm, maxLength: Number(event.currentTarget.value) })} />
              </div>
            </Panel>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const controller = useAppController();
  const [isPending, startTransition] = useTransition();

  const sidebarSurfaceButtons: Array<{ id: SurfaceId; label: string }> = [
    { id: 'generate', label: 'Generate' },
    { id: 'models', label: 'Models' },
    { id: 'voices', label: 'Voices' },
    { id: 'history', label: 'History' },
    { id: 'watermark-lab', label: 'Watermark Lab' },
    { id: 'system-status', label: 'System Status' },
    { id: 'advanced-settings', label: 'Advanced Settings' },
  ];

  const content = (() => {
    switch (controller.activeSurface) {
      case 'models':
        return <ModelsSurface controller={controller} />;
      case 'voices':
        return <VoicesSurface controller={controller} />;
      case 'history':
        return <HistorySurface controller={controller} />;
      case 'watermark-lab':
        return <WatermarkLabSurface controller={controller} />;
      case 'system-status':
        return <SystemStatusSurface controller={controller} />;
      case 'advanced-settings':
        return <AdvancedSettingsSurface controller={controller} />;
      default:
        return <GenerateSurface controller={controller} />;
    }
  })();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">T</span>
            <span>TTS Hub</span>
          </div>
          <div className="sidebar-subtitle">Atmospheric TTS workspace</div>
        </div>

        <div className="sidebar-label">Surfaces</div>
        <nav className="surface-nav" aria-label="Workspace sections">
          {sidebarSurfaceButtons.map((button) => (
            <SurfaceButton
              key={button.id}
              label={button.label}
              surface={button.id}
              active={controller.activeSurface === button.id}
              onClick={(surface) => {
                startTransition(() => controller.setActiveSurface(surface));
              }}
            />
          ))}
        </nav>

        <div className="sidebar-label">Models</div>
        <nav className="model-nav" id="modelNav">
          {controller.models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              status={controller.modelStatuses[model.id] || {}}
              active={controller.selectedModelId === model.id}
              onClick={(modelId) => controller.setSelectedModelId(modelId)}
            />
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="system-info">
            <span className={`status-dot${controller.infoAvailable === false ? ' error' : controller.infoAvailable === true ? '' : ' warning'}`} id="ffmpegDot" />
            <span id="ffmpegStatus">{controller.infoAvailable === null ? 'Checking ffmpeg...' : controller.infoAvailable ? 'ffmpeg: OK' : 'ffmpeg: Missing'}</span>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="top-bar">
          <div className="page-title">
            <div>
              <h1 id="selectedModelName">{controller.selectedModel ? normalizeModelName(controller.selectedModel.name) : 'Select a Model'}</h1>
              <div className="page-subtitle" id="modelDescription">{controller.selectedModel?.description || 'Pick a model from the sidebar'}</div>
            </div>
            <div className="status-badges" id="modelStatusBadges">
              {controller.selectedModelId ? (
                <>
                  <Badge tone={controller.selectedModelStatus.loaded ? 'success' : 'neutral'}>{controller.selectedModelStatus.loaded ? 'Loaded' : 'Ready'}</Badge>
                  {controller.selectedModelStatus.device ? <Badge tone="neutral">{controller.selectedModelStatus.device}</Badge> : null}
                  {typeof controller.selectedModelStatus.total_generations === 'number' ? <Badge tone="neutral">{controller.selectedModelStatus.total_generations} runs</Badge> : null}
                  {controller.supportsWatermark ? <Badge tone="info">Watermark</Badge> : null}
                </>
              ) : null}
            </div>
          </div>
          <div className="quick-actions">
            <button className="btn btn-secondary" id="clearSession" type="button" onClick={() => void controller.handleClearSession()}>Clear Session</button>
            <button className="btn btn-danger" id="unloadModel" type="button" onClick={() => void controller.handleUnloadModel()} disabled={!controller.selectedModelId || controller.isGenerating}>Unload</button>
          </div>
        </header>

        <div className="workspace-body">
          {content}
        </div>
      </main>

      <div id="systemClock" className="sr-only">{controller.systemClock}</div>
      <div id="surfaceStatus" className="sr-only">{controller.surfaceTitle}</div>
      <div id="ffmpegRuntimeStatus" className="sr-only">{controller.infoAvailable ? 'Available' : 'Unavailable'}</div>
    </div>
  );
}
