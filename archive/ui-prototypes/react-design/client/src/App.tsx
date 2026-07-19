import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  AppSettings,
  GenerationJobDetails,
  GenerationJobSummary,
  GenerationPhase,
  ModelSpec,
  ModelStatus,
  VoiceSummary,
  WatermarkRun,
  WatermarkRunDetails,
} from '../../frontend/src/types';
import {
  cancelGenerationJob,
  createGenerationJob,
  createVoice,
  deleteGenerationJob,
  deleteVoice,
  fetchGenerationJob,
  fetchGenerationJobs,
  fetchInfo,
  fetchModels,
  fetchStatus,
  fetchVoices,
  fetchWatermarkRunDetails,
  fetchWatermarkRuns,
  generationJobAudioUrl,
  renameVoice,
  unloadModel,
} from '../../frontend/src/api';
import {
  appendModelParams,
  buildDefaultSettings,
  isWatermarkSupported,
  requiresReferenceAudio,
  requiresTranscript,
} from '../../frontend/src/model-settings';

type Surface = 'generate' | 'voices' | 'jobs' | 'models' | 'watermark';
type VoiceSource = 'saved' | 'upload' | 'record' | 'none';
type NoticeKind = 'info' | 'success' | 'warning' | 'error' | 'loading';

type Notice = {
  kind: NoticeKind;
  text: string;
};

type PromptFile = {
  file: File | Blob;
  name: string;
  url: string;
};

type DetectResult = {
  detected: boolean;
  wm_prob: number;
  model?: {
    id?: number | string | null;
    name?: string | null;
    tts_model_id?: string | null;
  };
  run?: {
    id?: string | null;
  };
};

const SURFACES: Array<{ id: Surface; label: string; icon: string }> = [
  { id: 'generate', label: 'Generate', icon: 'wave' },
  { id: 'voices', label: 'Voices', icon: 'mic' },
  { id: 'jobs', label: 'Jobs', icon: 'list' },
  { id: 'models', label: 'Models & System', icon: 'grid' },
  { id: 'watermark', label: 'Watermark Lab', icon: 'shield' },
];

const STORAGE_PREFIX = 'tts-hub-react-prototype:';

function loadStored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(`${STORAGE_PREFIX}${key}`);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function saveStored<T>(key: string, value: T): void {
  try {
    localStorage.setItem(`${STORAGE_PREFIX}${key}`, JSON.stringify(value));
  } catch {
    // Local storage is optional.
  }
}

function formatModelName(name: string): string {
  return name.split(' (')[0] || name;
}

function normalizeTs(ts?: number | null): number | null {
  if (!ts || !Number.isFinite(ts)) return null;
  return Math.abs(ts) < 1e12 ? ts * 1000 : ts;
}

function formatRelative(ts?: number | null): string {
  const value = normalizeTs(ts);
  if (!value) return '--';
  const delta = Date.now() - value;
  const abs = Math.abs(delta);
  const mins = Math.round(abs / 60000);
  if (mins < 1) return 'now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 14) return `${days}d ago`;
  return new Date(value).toLocaleDateString();
}

function formatClock(ts?: number | null): string {
  const value = normalizeTs(ts);
  return value ? new Date(value).toLocaleTimeString() : new Date().toLocaleTimeString();
}

function formatDuration(seconds?: number | null): string {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return '--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

function formatMs(ms?: number | null): string {
  if (ms === undefined || ms === null || !Number.isFinite(ms)) return '--';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function phaseLabel(phase?: GenerationPhase | string | null): string {
  if (!phase) return 'Unknown';
  return phase.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
}

function isActivePhase(phase?: GenerationPhase | string | null): boolean {
  return !!phase && ['queued', 'preparing', 'generating', 'watermarking', 'converting'].includes(phase);
}

function statusTone(phase?: GenerationPhase | string | null): string {
  if (phase === 'completed') return 'good';
  if (phase === 'failed') return 'bad';
  if (phase === 'cancelled') return 'muted';
  if (isActivePhase(phase)) return 'warn';
  return 'muted';
}

function modelPurpose(modelId: string): string {
  const purposes: Record<string, string> = {
    'index-tts2': 'Voice cloning with emotion control',
    'qwen3-tts-mlx': 'MLX voice cloning with auto-transcribe',
    'chatterbox-multilingual': 'Multilingual TTS and cloning',
    'f5-hindi-urdu': 'Hindi / Urdu cloning with Roman mode',
    'cosyvoice3-mlx': 'MLX zero-shot, cross-lingual, instruct',
    'pocket-tts': 'Low-latency TTS with voice URL',
    'voxcpm-ane': 'ANE prompt-cache voice cloning',
  };
  return purposes[modelId] || 'Local TTS model';
}

function capabilityTags(modelId: string, settings: AppSettings): string[] {
  const tags: Record<string, string[]> = {
    'index-tts2': ['REF', 'EMOTION'],
    'qwen3-tts-mlx': ['REF', 'WATERMARK'],
    'chatterbox-multilingual': ['MULTILINGUAL'],
    'f5-hindi-urdu': ['REF', 'TRANSCRIPT'],
    'cosyvoice3-mlx': ['REF', settings.cosy.mode.toUpperCase()],
    'pocket-tts': ['VOICE URL'],
    'voxcpm-ane': ['TRANSCRIPT'],
  };
  const base = tags[modelId] || [];
  return isWatermarkSupported(modelId) && !base.includes('WATERMARK') ? [...base, 'WATERMARK'] : base;
}

function requirementIssues(
  modelId: string | null,
  text: string,
  promptText: string,
  voiceSource: VoiceSource,
  selectedVoice: VoiceSummary | null,
  promptFile: PromptFile | null,
  settings: AppSettings,
): string[] {
  const issues: string[] = [];
  if (!modelId) issues.push('Select a model');
  if (!text.trim()) issues.push('Add script text');
  const hasSavedVoice = voiceSource === 'saved' && !!selectedVoice;
  const hasPromptFile = (voiceSource === 'upload' || voiceSource === 'record') && !!promptFile;
  if (modelId && requiresReferenceAudio(modelId) && !hasSavedVoice && !hasPromptFile) {
    issues.push('Add reference voice');
  }
  if (modelId && requiresTranscript(modelId, settings)) {
    const savedHasTranscript = hasSavedVoice && selectedVoice?.has_transcript;
    if (!promptText.trim() && !savedHasTranscript) issues.push('Add transcript');
  }
  if (modelId === 'voxcpm-ane' && !settings.voxcpm.voice.trim() && !hasSavedVoice && !hasPromptFile) {
    issues.push('Add cached voice or reference voice');
  }
  return issues;
}

async function detectWatermark(form: FormData): Promise<DetectResult> {
  const response = await fetch('/api/watermark/detect', { method: 'POST', body: form });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error((data as { error?: string }).error || `HTTP ${response.status}`);
  return data as DetectResult;
}

function Icon({ name }: { name: string }) {
  if (name === 'wave') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 13h2l1.6-5 3.2 10L13 4l2.7 12 1.4-3H21" />
      </svg>
    );
  }
  if (name === 'mic') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="9" y="3" width="6" height="11" rx="3" />
        <path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" />
      </svg>
    );
  }
  if (name === 'list') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
      </svg>
    );
  }
  if (name === 'grid') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="4" width="6" height="6" rx="1" />
        <rect x="14" y="4" width="6" height="6" rx="1" />
        <rect x="4" y="14" width="6" height="6" rx="1" />
        <rect x="14" y="14" width="6" height="6" rx="1" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 5 6v5c0 4.4 2.8 8.3 7 10 4.2-1.7 7-5.6 7-10V6l-7-3Z" />
      <path d="m9 12 2 2 4-5" />
    </svg>
  );
}

function Waveform() {
  return (
    <div className="waveform" aria-hidden="true">
      {Array.from({ length: 76 }, (_, index) => (
        <span key={index} style={{ height: `${22 + ((index * 17) % 42)}px` }} />
      ))}
    </div>
  );
}

export function App() {
  const [surface, setSurface] = useState<Surface>(() => loadStored<Surface>('surface', 'generate'));
  const [models, setModels] = useState<ModelSpec[]>([]);
  const [statuses, setStatuses] = useState<Record<string, ModelStatus>>({});
  const [voices, setVoices] = useState<VoiceSummary[]>([]);
  const [jobs, setJobs] = useState<GenerationJobSummary[]>([]);
  const [watermarkRuns, setWatermarkRuns] = useState<WatermarkRun[]>([]);
  const [watermarkDefaultRun, setWatermarkDefaultRun] = useState<string | null>(null);
  const [watermarkDetails, setWatermarkDetails] = useState<WatermarkRunDetails | null>(null);
  const [selectedRun, setSelectedRun] = useState(() => loadStored('watermarkRun', ''));
  const [ffmpegReady, setFfmpegReady] = useState<boolean | null>(null);
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString());
  const [notice, setNotice] = useState<Notice>({ kind: 'loading', text: 'Loading archived prototype...' });

  const [selectedModelId, setSelectedModelId] = useState<string | null>(() => loadStored<string | null>('model', null));
  const [text, setText] = useState(() => loadStored('text', ''));
  const [promptText, setPromptText] = useState(() => loadStored('promptText', ''));
  const [voiceSource, setVoiceSource] = useState<VoiceSource>(() => loadStored<VoiceSource>('voiceSource', 'saved'));
  const [selectedVoiceId, setSelectedVoiceId] = useState(() => loadStored('voiceId', ''));
  const [promptFile, setPromptFile] = useState<PromptFile | null>(null);
  const [outputFormat, setOutputFormat] = useState<'wav' | 'mp3' | 'flac'>(() => loadStored('format', 'wav'));
  const [watermarkEnabled, setWatermarkEnabled] = useState(() => loadStored('watermarkEnabled', false));
  const [settings, setSettings] = useState<AppSettings>(() => ({ ...buildDefaultSettings(), ...loadStored<Partial<AppSettings>>('settings', {}) } as AppSettings));
  const [activeJob, setActiveJob] = useState<GenerationJobDetails | null>(null);
  const [jobFilter, setJobFilter] = useState<'all' | 'active' | 'completed' | 'failed'>('all');
  const [recording, setRecording] = useState(false);
  const [renameId, setRenameId] = useState('');
  const [voiceName, setVoiceName] = useState('');
  const [voicePrompt, setVoicePrompt] = useState('');
  const [voiceCreateFile, setVoiceCreateFile] = useState<File | null>(null);
  const [detectFile, setDetectFile] = useState<File | null>(null);
  const [detectThreshold, setDetectThreshold] = useState('0.35');
  const [detectResult, setDetectResult] = useState<DetectResult | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const promptInputRef = useRef<HTMLInputElement | null>(null);
  const createVoiceInputRef = useRef<HTMLInputElement | null>(null);
  const detectInputRef = useRef<HTMLInputElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordChunksRef = useRef<Blob[]>([]);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) || null,
    [models, selectedModelId],
  );
  const selectedStatus = selectedModelId ? statuses[selectedModelId] || {} : {};
  const selectedVoice = useMemo(
    () => voices.find((voice) => voice.id === selectedVoiceId) || null,
    [voices, selectedVoiceId],
  );
  const latestJob = activeJob || jobs.find((job) => job.status === 'completed' && job.output) || jobs[0] || null;
  const issues = requirementIssues(selectedModelId, text, promptText, voiceSource, selectedVoice, promptFile, settings);
  const canGenerate = issues.length === 0;

  const refreshCore = useCallback(async (quiet = false) => {
    try {
      const [modelData, infoData, statusData, voiceData, jobData, runData] = await Promise.all([
        fetchModels(),
        fetchInfo(),
        fetchStatus(),
        fetchVoices(),
        fetchGenerationJobs(),
        fetchWatermarkRuns(),
      ]);
      setModels(modelData);
      setFfmpegReady(infoData.ffmpeg.available);
      setClock(formatClock(infoData.time));
      setStatuses(statusData.models || {});
      setVoices(voiceData);
      setJobs(jobData);
      setWatermarkRuns(runData.runs || []);
      setWatermarkDefaultRun(runData.default_run_id);
      if (!selectedModelId && modelData[0]) setSelectedModelId(modelData[0].id);
      if (!selectedRun && runData.default_run_id) setSelectedRun(runData.default_run_id);
      if (!quiet) setNotice({ kind: 'success', text: 'Live backend data loaded.' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Failed to load backend data.' });
    }
  }, [selectedModelId, selectedRun]);

  useEffect(() => {
    void refreshCore();
    const timer = window.setInterval(() => void refreshCore(true), 5000);
    return () => window.clearInterval(timer);
  }, [refreshCore]);

  useEffect(() => {
    if (!selectedRun) return;
    fetchWatermarkRunDetails(selectedRun)
      .then(setWatermarkDetails)
      .catch((error) => setWatermarkDetails({ id: selectedRun, error: error instanceof Error ? error.message : String(error) }));
  }, [selectedRun]);

  useEffect(() => saveStored('surface', surface), [surface]);
  useEffect(() => saveStored('model', selectedModelId), [selectedModelId]);
  useEffect(() => saveStored('text', text), [text]);
  useEffect(() => saveStored('promptText', promptText), [promptText]);
  useEffect(() => saveStored('voiceSource', voiceSource), [voiceSource]);
  useEffect(() => saveStored('voiceId', selectedVoiceId), [selectedVoiceId]);
  useEffect(() => saveStored('format', outputFormat), [outputFormat]);
  useEffect(() => saveStored('watermarkEnabled', watermarkEnabled), [watermarkEnabled]);
  useEffect(() => saveStored('watermarkRun', selectedRun), [selectedRun]);
  useEffect(() => saveStored('settings', settings), [settings]);

  useEffect(() => {
    if (!activeJob || !isActivePhase(activeJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await fetchGenerationJob(activeJob.id);
        setActiveJob(next);
        setJobs((current) => [next, ...current.filter((job) => job.id !== next.id)]);
        if (!isActivePhase(next.status)) {
          setNotice({ kind: next.status === 'completed' ? 'success' : 'error', text: next.status === 'completed' ? 'Generation complete.' : next.error || phaseLabel(next.status) });
        }
      } catch (error) {
        setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Failed to poll generation job.' });
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [activeJob]);

  const patchSettings = <K extends keyof AppSettings>(group: K, patch: Partial<AppSettings[K]>) => {
    setSettings((current) => ({ ...current, [group]: { ...current[group], ...patch } }));
  };

  const handlePromptUpload = (file: File | null) => {
    if (!file) return;
    if (promptFile?.url) URL.revokeObjectURL(promptFile.url);
    setPromptFile({ file, name: file.name, url: URL.createObjectURL(file) });
    setVoiceSource('upload');
    setNotice({ kind: 'success', text: `Reference loaded: ${file.name}` });
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(recordChunksRef.current, { type: 'audio/webm' });
        if (promptFile?.url) URL.revokeObjectURL(promptFile.url);
        setPromptFile({ file: blob, name: 'recorded-reference.webm', url: URL.createObjectURL(blob) });
        setVoiceSource('record');
        setRecording(false);
        setNotice({ kind: 'success', text: 'Reference recording captured.' });
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setNotice({ kind: 'loading', text: 'Recording reference voice...' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Microphone recording failed.' });
    }
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
  };

  const submitGeneration = async () => {
    if (!selectedModelId || !canGenerate) {
      setNotice({ kind: 'warning', text: issues.join(', ') || 'Generation requirements are incomplete.' });
      return;
    }
    const form = new FormData();
    form.append('model_id', selectedModelId);
    form.append('text', text.trim());
    form.append('output_format', outputFormat);
    if (voiceSource === 'saved' && selectedVoiceId) form.append('voice_id', selectedVoiceId);
    if ((voiceSource === 'upload' || voiceSource === 'record') && promptFile) form.append('prompt_audio', promptFile.file, promptFile.name);
    if (promptText.trim()) form.append('prompt_text', promptText.trim());
    if (watermarkEnabled && isWatermarkSupported(selectedModelId)) {
      form.append('watermark', '1');
      if (selectedRun) form.append('watermark_run', selectedRun);
    }
    appendModelParams(form, selectedModelId, settings);
    form.append('request_snapshot', JSON.stringify({
      modelId: selectedModelId,
      text,
      promptText,
      voiceId: voiceSource === 'saved' ? selectedVoiceId || null : null,
      outputFormat,
      watermarkEnabled,
      watermarkRun: selectedRun || null,
      settings,
    }));

    setNotice({ kind: 'loading', text: 'Submitting generation job...' });
    try {
      const job = await createGenerationJob(form);
      setActiveJob(job);
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      setNotice({ kind: 'loading', text: `Job ${job.id.slice(0, 8)} queued.` });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Generation failed to start.' });
    }
  };

  const handleCreateVoice = async () => {
    if (!voiceName.trim() || !voiceCreateFile) {
      setNotice({ kind: 'warning', text: 'Voice name and audio file are required.' });
      return;
    }
    const form = new FormData();
    form.append('name', voiceName.trim());
    form.append('prompt_audio', voiceCreateFile, voiceCreateFile.name);
    if (voicePrompt.trim()) form.append('prompt_text', voicePrompt.trim());
    try {
      await createVoice(form);
      setVoiceName('');
      setVoicePrompt('');
      setVoiceCreateFile(null);
      if (createVoiceInputRef.current) createVoiceInputRef.current.value = '';
      await refreshCore(true);
      setNotice({ kind: 'success', text: 'Voice saved to library.' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Voice creation failed.' });
    }
  };

  const handleRenameVoice = async (voiceId: string) => {
    const nextName = window.prompt('Rename voice');
    if (!nextName?.trim()) return;
    try {
      await renameVoice(voiceId, nextName.trim());
      await refreshCore(true);
      setRenameId('');
      setNotice({ kind: 'success', text: 'Voice renamed.' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Voice rename failed.' });
    }
  };

  const handleDeleteVoice = async (voiceId: string) => {
    if (!window.confirm('Delete this saved voice?')) return;
    try {
      await deleteVoice(voiceId);
      if (selectedVoiceId === voiceId) setSelectedVoiceId('');
      await refreshCore(true);
      setNotice({ kind: 'success', text: 'Voice deleted.' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Voice delete failed.' });
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      const job = await cancelGenerationJob(jobId);
      setActiveJob(job);
      await refreshCore(true);
      setNotice({ kind: 'success', text: 'Job cancellation requested.' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Job cancellation failed.' });
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    try {
      await deleteGenerationJob(jobId);
      if (activeJob?.id === jobId) setActiveJob(null);
      await refreshCore(true);
      setNotice({ kind: 'success', text: 'Job deleted.' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Job delete failed.' });
    }
  };

  const handleUnload = async (modelId: string) => {
    try {
      await unloadModel(modelId);
      await refreshCore(true);
      setNotice({ kind: 'success', text: `${formatModelName(models.find((m) => m.id === modelId)?.name || modelId)} unloaded.` });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Unload failed.' });
    }
  };

  const handleDetect = async () => {
    if (!detectFile) {
      setNotice({ kind: 'warning', text: 'Choose an audio file to analyze.' });
      return;
    }
    const form = new FormData();
    form.append('audio', detectFile, detectFile.name);
    if (selectedRun) form.append('watermark_run', selectedRun);
    form.append('wm_threshold', detectThreshold || '0.35');
    setNotice({ kind: 'loading', text: 'Running watermark detection...' });
    try {
      const result = await detectWatermark(form);
      setDetectResult(result);
      setNotice({ kind: 'success', text: result.detected ? 'Watermark detected.' : 'No watermark detected.' });
    } catch (error) {
      setNotice({ kind: 'error', text: error instanceof Error ? error.message : 'Watermark detection failed.' });
    }
  };

  const filteredJobs = jobs.filter((job) => {
    if (jobFilter === 'all') return true;
    if (jobFilter === 'active') return isActivePhase(job.status);
    if (jobFilter === 'completed') return job.status === 'completed';
    return job.status === 'failed';
  });

  return (
    <div className="app-shell">
      <aside className="nav-rail" aria-label="Primary">
        <div className="brand">
          <div className="brand-mark"><Icon name="wave" /></div>
          <strong>TTS HUB</strong>
        </div>
        <nav className="nav-list">
          {SURFACES.map((item) => (
            <button key={item.id} className={surface === item.id ? 'nav-item active' : 'nav-item'} onClick={() => setSurface(item.id)}>
              <Icon name={item.icon} />
              <span>{item.label}</span>
              {item.id === 'voices' && <em>{voices.length}</em>}
            </button>
          ))}
        </nav>
        <div className="nav-footer">
          <div className="ffmpeg"><i className={ffmpegReady ? 'dot good' : 'dot bad'} />{ffmpegReady ? 'ffmpeg ready' : 'ffmpeg missing'}</div>
          <code>{clock}</code>
          <div className="theme-toggle" aria-label="Theme">
            <button className="active">Dark</button>
            <button disabled>Light</button>
          </div>
        </div>
      </aside>

      {surface === 'generate' && (
        <ModelRail
          models={models}
          statuses={statuses}
          selectedModelId={selectedModelId}
          settings={settings}
          onSelect={setSelectedModelId}
        />
      )}

      <main className={surface === 'generate' ? 'main-workspace' : 'main-workspace full'} aria-live="polite">
        {surface === 'generate' && selectedModel && (
          <GenerateWorkspace
            model={selectedModel}
            status={selectedStatus}
            text={text}
            setText={setText}
            promptText={promptText}
            setPromptText={setPromptText}
            voiceSource={voiceSource}
            setVoiceSource={setVoiceSource}
            voices={voices}
            selectedVoiceId={selectedVoiceId}
            setSelectedVoiceId={setSelectedVoiceId}
            promptFile={promptFile}
            promptInputRef={promptInputRef}
            onPromptUpload={handlePromptUpload}
            recording={recording}
            startRecording={startRecording}
            stopRecording={stopRecording}
            outputFormat={outputFormat}
            setOutputFormat={setOutputFormat}
            watermarkEnabled={watermarkEnabled}
            setWatermarkEnabled={setWatermarkEnabled}
            watermarkRuns={watermarkRuns}
            selectedRun={selectedRun}
            setSelectedRun={setSelectedRun}
            settings={settings}
            issues={issues}
            canGenerate={canGenerate}
            onGenerate={submitGeneration}
          />
        )}
        {surface === 'generate' && !selectedModel && <EmptyState title="No model selected" text="Backend models will appear here after the API responds." />}
        {surface === 'voices' && (
          <VoicesSurface
            voices={voices}
            selectedVoiceId={selectedVoiceId}
            setSelectedVoiceId={setSelectedVoiceId}
            setSurface={setSurface}
            voiceName={voiceName}
            setVoiceName={setVoiceName}
            voicePrompt={voicePrompt}
            setVoicePrompt={setVoicePrompt}
            voiceCreateFile={voiceCreateFile}
            setVoiceCreateFile={setVoiceCreateFile}
            inputRef={createVoiceInputRef}
            onCreate={handleCreateVoice}
            onRename={handleRenameVoice}
            onDelete={handleDeleteVoice}
            renameId={renameId}
            setRenameId={setRenameId}
          />
        )}
        {surface === 'jobs' && (
          <JobsSurface
            jobs={filteredJobs}
            filter={jobFilter}
            setFilter={setJobFilter}
            models={models}
            onSelectJob={(job) => setActiveJob(job as GenerationJobDetails)}
            onCancel={handleCancelJob}
            onDelete={handleDeleteJob}
          />
        )}
        {surface === 'models' && (
          <ModelsSurface models={models} statuses={statuses} settings={settings} onUnload={handleUnload} />
        )}
        {surface === 'watermark' && (
          <WatermarkSurface
            runs={watermarkRuns}
            defaultRun={watermarkDefaultRun}
            selectedRun={selectedRun}
            setSelectedRun={setSelectedRun}
            details={watermarkDetails}
            detectFile={detectFile}
            setDetectFile={setDetectFile}
            detectInputRef={detectInputRef}
            threshold={detectThreshold}
            setThreshold={setDetectThreshold}
            onDetect={handleDetect}
            result={detectResult}
          />
        )}
      </main>

      {surface === 'generate' && (
        <SettingsRail
          open={settingsOpen}
          setOpen={setSettingsOpen}
          modelId={selectedModelId}
          settings={settings}
          patchSettings={patchSettings}
        />
      )}

      <OutputDock
        job={latestJob}
        notice={notice}
        models={models}
        onCancel={latestJob && isActivePhase(latestJob.status) ? () => handleCancelJob(latestJob.id) : undefined}
      />
    </div>
  );
}

function ModelRail({
  models,
  statuses,
  selectedModelId,
  settings,
  onSelect,
}: {
  models: ModelSpec[];
  statuses: Record<string, ModelStatus>;
  selectedModelId: string | null;
  settings: AppSettings;
  onSelect: (modelId: string) => void;
}) {
  const loaded = models.filter((model) => statuses[model.id]?.loaded).length;
  return (
    <aside className="model-rail" aria-label="Models">
      <header>
        <span>MODELS</span>
        <code>{loaded}/{models.length}</code>
      </header>
      <div className="model-list">
        {models.map((model) => {
          const status = statuses[model.id] || {};
          return (
            <button
              key={model.id}
              className={selectedModelId === model.id ? 'model-card active' : 'model-card'}
              onClick={() => onSelect(model.id)}
            >
              <div>
                <i className={status.loaded ? 'dot good' : 'dot muted'} />
                <strong>{formatModelName(model.name)}</strong>
              </div>
              <code>{status.loaded ? 'loaded' : 'idle'} · {status.device || 'unknown'} · {status.total_generations || 0} runs</code>
              <span className="tag-row">
                {capabilityTags(model.id, settings).map((tag) => <em key={tag}>{tag}</em>)}
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

function GenerateWorkspace(props: {
  model: ModelSpec;
  status: ModelStatus;
  text: string;
  setText: (value: string) => void;
  promptText: string;
  setPromptText: (value: string) => void;
  voiceSource: VoiceSource;
  setVoiceSource: (value: VoiceSource) => void;
  voices: VoiceSummary[];
  selectedVoiceId: string;
  setSelectedVoiceId: (value: string) => void;
  promptFile: PromptFile | null;
  promptInputRef: React.RefObject<HTMLInputElement | null>;
  onPromptUpload: (file: File | null) => void;
  recording: boolean;
  startRecording: () => void;
  stopRecording: () => void;
  outputFormat: 'wav' | 'mp3' | 'flac';
  setOutputFormat: (value: 'wav' | 'mp3' | 'flac') => void;
  watermarkEnabled: boolean;
  setWatermarkEnabled: (value: boolean) => void;
  watermarkRuns: WatermarkRun[];
  selectedRun: string;
  setSelectedRun: (value: string) => void;
  settings: AppSettings;
  issues: string[];
  canGenerate: boolean;
  onGenerate: () => void;
}) {
  const needsRef = requiresReferenceAudio(props.model.id);
  const needsTranscript = requiresTranscript(props.model.id, props.settings);
  const supportsWm = isWatermarkSupported(props.model.id);
  const compatibleVoices = props.voices.filter((voice) => !voice.compatible_models?.length || voice.compatible_models.includes(props.model.id));

  return (
    <section className="generate-surface">
      <header className="workspace-head">
        <div>
          <h1>{formatModelName(props.model.name)}</h1>
          <p>{props.model.description}</p>
        </div>
        <div className="requirements">
          <span>Requirements</span>
          <em className={props.text.trim() ? 'ok' : ''}>Script</em>
          {needsRef && <em className={(props.voiceSource === 'saved' && props.selectedVoiceId) || props.promptFile ? 'ok' : ''}>Reference</em>}
          {needsTranscript && <em className={props.promptText.trim() ? 'ok' : ''}>Transcript</em>}
        </div>
      </header>

      <div className="script-panel">
        <label htmlFor="script">SCRIPT</label>
        <code>{props.text.length} chars</code>
        <textarea
          id="script"
          value={props.text}
          onChange={(event) => props.setText(event.target.value)}
          placeholder="Type or paste the text to synthesize..."
        />
      </div>

      {(needsTranscript || props.promptText) && (
        <div className="transcript-panel">
          <label htmlFor="promptText">REFERENCE TRANSCRIPT</label>
          <textarea
            id="promptText"
            value={props.promptText}
            onChange={(event) => props.setPromptText(event.target.value)}
            placeholder="Transcript for the reference voice when the selected model requires it..."
          />
        </div>
      )}

      <section className="voice-panel">
        <header>
          <div>
            <span>REFERENCE VOICE</span>
            {needsRef ? <em>Required</em> : <em className="soft">Optional</em>}
          </div>
          <div className="segmented">
            {(['saved', 'upload', 'record', 'none'] as VoiceSource[]).map((source) => (
              <button key={source} className={props.voiceSource === source ? 'active' : ''} onClick={() => props.setVoiceSource(source)}>
                {source[0].toUpperCase() + source.slice(1)}
              </button>
            ))}
          </div>
        </header>
        {props.voiceSource === 'saved' && (
          <div className="voice-list compact">
            {compatibleVoices.length === 0 && <EmptyState title="No compatible saved voices" text="Upload or record a reference for this model." />}
            {compatibleVoices.map((voice) => (
              <button
                key={voice.id}
                className={props.selectedVoiceId === voice.id ? 'voice-row active' : 'voice-row'}
                onClick={() => props.setSelectedVoiceId(voice.id)}
              >
                <i className={props.selectedVoiceId === voice.id ? 'dot good' : 'dot muted'} />
                <strong>{voice.name}</strong>
                <code>{formatDuration(voice.duration_s)}</code>
                {voice.has_transcript && <em>TRANSCRIPT</em>}
                <span>{formatRelative(voice.created_at)}</span>
              </button>
            ))}
          </div>
        )}
        {props.voiceSource === 'upload' && (
          <div className="upload-box">
            <input ref={props.promptInputRef} type="file" accept="audio/*" onChange={(event) => props.onPromptUpload(event.target.files?.[0] || null)} />
            <button onClick={() => props.promptInputRef.current?.click()}>Choose audio</button>
            <span>{props.promptFile?.name || 'No reference audio selected'}</span>
            {props.promptFile && <audio src={props.promptFile.url} controls />}
          </div>
        )}
        {props.voiceSource === 'record' && (
          <div className="upload-box">
            <button className={props.recording ? 'danger' : ''} onClick={props.recording ? props.stopRecording : props.startRecording}>
              {props.recording ? 'Stop recording' : 'Record reference'}
            </button>
            <span>{props.promptFile?.name || 'Microphone capture will be staged as a reference'}</span>
            {props.promptFile && <audio src={props.promptFile.url} controls />}
          </div>
        )}
        {props.voiceSource === 'none' && <EmptyState title="No reference voice" text="Only use this for models that support promptless generation or cached voices." />}
      </section>

      <footer className="generate-footer">
        <FieldGroup label="FORMAT">
          <div className="segmented strong">
            {(['wav', 'mp3', 'flac'] as const).map((format) => (
              <button key={format} className={props.outputFormat === format ? 'active' : ''} onClick={() => props.setOutputFormat(format)}>
                {format.toUpperCase()}
              </button>
            ))}
          </div>
        </FieldGroup>
        <FieldGroup label="WATERMARK">
          <label className={supportsWm ? 'switch' : 'switch disabled'}>
            <input
              type="checkbox"
              checked={supportsWm && props.watermarkEnabled}
              disabled={!supportsWm}
              onChange={(event) => props.setWatermarkEnabled(event.target.checked)}
            />
            <span>{supportsWm && props.watermarkEnabled ? 'On' : 'Off'}</span>
          </label>
        </FieldGroup>
        {supportsWm && props.watermarkEnabled && (
          <select value={props.selectedRun} onChange={(event) => props.setSelectedRun(event.target.value)} aria-label="Watermark run">
            {props.watermarkRuns.map((run) => <option key={run.id} value={run.id}>{run.label}</option>)}
          </select>
        )}
        <div className="issue-strip">
          {props.issues.map((issue) => <em key={issue}>{issue}</em>)}
        </div>
        <button className="generate-button" disabled={!props.canGenerate} onClick={props.onGenerate}>
          <span className="play-triangle" /> Run generation
        </button>
      </footer>
    </section>
  );
}

function FieldGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field-group">
      <span>{label}</span>
      {children}
    </div>
  );
}

function SettingsRail({
  open,
  setOpen,
  modelId,
  settings,
  patchSettings,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  modelId: string | null;
  settings: AppSettings;
  patchSettings: <K extends keyof AppSettings>(group: K, patch: Partial<AppSettings[K]>) => void;
}) {
  return (
    <aside className={open ? 'settings-rail open' : 'settings-rail'} aria-label="Model settings">
      <button className="rail-tab" onClick={() => setOpen(!open)}>MODEL SETTINGS</button>
      {open && (
        <div className="settings-body">
          <h2>Model Settings</h2>
          {!modelId && <EmptyState title="No model" text="Select a model to tune parameters." />}
          {modelId === 'index-tts2' && (
            <>
              <Select label="Emotion mode" value={settings.index.emoMode} onChange={(value) => patchSettings('index', { emoMode: value as AppSettings['index']['emoMode'] })} options={['speaker', 'emo_ref', 'emo_vector', 'emo_text']} />
              <Range label="Emotion alpha" value={settings.index.emoAlpha} min={0} max={1} step={0.05} onChange={(value) => patchSettings('index', { emoAlpha: value })} />
              <NumberInput label="Max text tokens" value={settings.index.maxTextTokens} onChange={(value) => patchSettings('index', { maxTextTokens: value })} />
              <NumberInput label="Temperature" value={settings.index.temperature} step={0.1} onChange={(value) => patchSettings('index', { temperature: value })} />
              <NumberInput label="Top P" value={settings.index.topP} step={0.05} onChange={(value) => patchSettings('index', { topP: value })} />
              <Toggle label="Fast mode" checked={settings.index.fastMode} onChange={(value) => patchSettings('index', { fastMode: value })} />
            </>
          )}
          {modelId === 'qwen3-tts-mlx' && (
            <>
              <TextInput label="Model" value={settings.qwen.model} onChange={(value) => patchSettings('qwen', { model: value })} />
              <Toggle label="Auto transcribe" checked={settings.qwen.autoTranscribe} onChange={(value) => patchSettings('qwen', { autoTranscribe: value })} />
              <TextInput label="Language" value={settings.qwen.language} onChange={(value) => patchSettings('qwen', { language: value })} />
              <Range label="Speed" value={settings.qwen.speed} min={0.5} max={2} step={0.05} onChange={(value) => patchSettings('qwen', { speed: value })} />
              <NumberInput label="Temperature" value={settings.qwen.temperature} step={0.1} onChange={(value) => patchSettings('qwen', { temperature: value })} />
              <NumberInput label="Max tokens" value={settings.qwen.maxTokens} onChange={(value) => patchSettings('qwen', { maxTokens: value })} />
            </>
          )}
          {modelId === 'chatterbox-multilingual' && (
            <>
              <TextInput label="Language" value={settings.chatterbox.language} onChange={(value) => patchSettings('chatterbox', { language: value })} />
              <Range label="CFG weight" value={settings.chatterbox.cfgWeight} min={0} max={1} step={0.05} onChange={(value) => patchSettings('chatterbox', { cfgWeight: value })} />
              <Range label="Temperature" value={settings.chatterbox.temperature} min={0.1} max={2} step={0.1} onChange={(value) => patchSettings('chatterbox', { temperature: value })} />
              <Range label="Exaggeration" value={settings.chatterbox.exaggeration} min={0} max={1.5} step={0.05} onChange={(value) => patchSettings('chatterbox', { exaggeration: value })} />
              <Toggle label="Chunking" checked={settings.chatterbox.enableChunking} onChange={(value) => patchSettings('chatterbox', { enableChunking: value })} />
              <NumberInput label="Max chunk chars" value={settings.chatterbox.maxChunkChars} onChange={(value) => patchSettings('chatterbox', { maxChunkChars: value })} />
              <NumberInput label="Crossfade ms" value={settings.chatterbox.crossfadeMs} onChange={(value) => patchSettings('chatterbox', { crossfadeMs: value })} />
              <Toggle label="DeepFilterNet" checked={settings.chatterbox.enableDf} onChange={(value) => patchSettings('chatterbox', { enableDf: value })} />
              <Toggle label="NovaSR" checked={settings.chatterbox.enableNovasr} onChange={(value) => patchSettings('chatterbox', { enableNovasr: value })} />
            </>
          )}
          {modelId === 'f5-hindi-urdu' && (
            <>
              <Toggle label="Roman mode" checked={settings.f5.romanMode} onChange={(value) => patchSettings('f5', { romanMode: value })} />
              <Toggle label="Overrides" checked={settings.f5.overridesEnabled} onChange={(value) => patchSettings('f5', { overridesEnabled: value })} />
              <TextArea label="Overrides text" value={settings.f5.overridesText} onChange={(value) => patchSettings('f5', { overridesText: value })} />
              <NumberInput label="NFE step" value={settings.f5.nfeStep} onChange={(value) => patchSettings('f5', { nfeStep: value })} />
              <Range label="Speed" value={settings.f5.speed} min={0.5} max={2} step={0.05} onChange={(value) => patchSettings('f5', { speed: value })} />
              <NumberInput label="Seed" value={settings.f5.seed} onChange={(value) => patchSettings('f5', { seed: value })} />
              <Toggle label="Remove silence" checked={settings.f5.removeSilence} onChange={(value) => patchSettings('f5', { removeSilence: value })} />
            </>
          )}
          {modelId === 'cosyvoice3-mlx' && (
            <>
              <Select label="Model" value={settings.cosy.model} onChange={(value) => patchSettings('cosy', { model: value })} options={['8bit', '4bit', 'fp16']} />
              <Select label="Mode" value={settings.cosy.mode} onChange={(value) => patchSettings('cosy', { mode: value as AppSettings['cosy']['mode'] })} options={['zero_shot', 'cross_lingual', 'instruct']} />
              <TextInput label="Language" value={settings.cosy.language} onChange={(value) => patchSettings('cosy', { language: value })} />
              <Range label="Speed" value={settings.cosy.speed} min={0.5} max={2} step={0.05} onChange={(value) => patchSettings('cosy', { speed: value })} />
              <TextArea label="Instruct text" value={settings.cosy.instructText} onChange={(value) => patchSettings('cosy', { instructText: value })} />
            </>
          )}
          {modelId === 'pocket-tts' && (
            <>
              <TextInput label="Voice URL" value={settings.pocket.voice} onChange={(value) => patchSettings('pocket', { voice: value })} />
              <Range label="Temperature" value={settings.pocket.temperature} min={0.1} max={2} step={0.1} onChange={(value) => patchSettings('pocket', { temperature: value })} />
              <NumberInput label="Decode steps" value={settings.pocket.lsdDecodeSteps} onChange={(value) => patchSettings('pocket', { lsdDecodeSteps: value })} />
              <Range label="EOS threshold" value={settings.pocket.eosThreshold} min={0} max={1} step={0.05} onChange={(value) => patchSettings('pocket', { eosThreshold: value })} />
              <TextInput label="Noise clamp" value={settings.pocket.noiseClamp} onChange={(value) => patchSettings('pocket', { noiseClamp: value })} />
              <Toggle label="Truncate prompt" checked={settings.pocket.truncatePrompt} onChange={(value) => patchSettings('pocket', { truncatePrompt: value })} />
            </>
          )}
          {modelId === 'voxcpm-ane' && (
            <>
              <TextInput label="Cached voice" value={settings.voxcpm.voice} onChange={(value) => patchSettings('voxcpm', { voice: value })} />
              <Range label="CFG value" value={settings.voxcpm.cfgValue} min={0} max={5} step={0.1} onChange={(value) => patchSettings('voxcpm', { cfgValue: value })} />
              <NumberInput label="Timesteps" value={settings.voxcpm.inferenceTimesteps} onChange={(value) => patchSettings('voxcpm', { inferenceTimesteps: value })} />
              <NumberInput label="Max length" value={settings.voxcpm.maxLength} onChange={(value) => patchSettings('voxcpm', { maxLength: value })} />
            </>
          )}
        </div>
      )}
    </aside>
  );
}

function TextInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label className="setting-field"><span>{label}</span><input value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}

function TextArea({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label className="setting-field"><span>{label}</span><textarea value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}

function NumberInput({ label, value, step = 1, onChange }: { label: string; value: number; step?: number; onChange: (value: number) => void }) {
  return <label className="setting-field"><span>{label}</span><input type="number" step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function Range({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) {
  return (
    <label className="setting-field">
      <span>{label}<code>{value}</code></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="setting-toggle"><span>{label}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /></label>;
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label className="setting-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

function VoicesSurface(props: {
  voices: VoiceSummary[];
  selectedVoiceId: string;
  setSelectedVoiceId: (id: string) => void;
  setSurface: (surface: Surface) => void;
  voiceName: string;
  setVoiceName: (value: string) => void;
  voicePrompt: string;
  setVoicePrompt: (value: string) => void;
  voiceCreateFile: File | null;
  setVoiceCreateFile: (file: File | null) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onCreate: () => void;
  onRename: (id: string) => void;
  onDelete: (id: string) => void;
  renameId: string;
  setRenameId: (id: string) => void;
}) {
  return (
    <section className="surface-page">
      <header className="page-head">
        <div><h1>Voice Library</h1><p>{props.voices.length} saved voices from the live backend.</p></div>
        <button className="primary-small" onClick={props.onCreate}>+ Add voice</button>
      </header>
      <div className="voice-create">
        <input placeholder="Voice name" value={props.voiceName} onChange={(event) => props.setVoiceName(event.target.value)} />
        <input ref={props.inputRef} type="file" accept="audio/*" onChange={(event) => props.setVoiceCreateFile(event.target.files?.[0] || null)} />
        <textarea placeholder="Optional reference transcript" value={props.voicePrompt} onChange={(event) => props.setVoicePrompt(event.target.value)} />
      </div>
      <div className="voice-grid">
        {props.voices.map((voice) => (
          <article key={voice.id} className={props.selectedVoiceId === voice.id ? 'library-card active' : 'library-card'}>
            <header>
              <strong>{voice.name}</strong>
              {voice.has_transcript && <em>TRANSCRIPT</em>}
            </header>
            <div className="metrics"><code>{formatDuration(voice.duration_s)}</code><code>{formatRelative(voice.created_at)}</code></div>
            <div className="chips">{voice.compatible_models.slice(0, 5).map((model) => <span key={model}>{model.replace('-tts', '')}</span>)}</div>
            <audio src={`/api/voices/${encodeURIComponent(voice.id)}/audio`} controls preload="none" />
            <footer>
              <button onClick={() => { props.setSelectedVoiceId(voice.id); props.setSurface('generate'); }}>Use</button>
              <button onClick={() => props.onRename(voice.id)}>Rename</button>
              <button className="danger-text" onClick={() => props.onDelete(voice.id)}>Delete</button>
            </footer>
          </article>
        ))}
      </div>
      {props.voices.length === 0 && <EmptyState title="No saved voices" text="Create one with an audio file and optional transcript." />}
    </section>
  );
}

function JobsSurface(props: {
  jobs: GenerationJobSummary[];
  filter: 'all' | 'active' | 'completed' | 'failed';
  setFilter: (filter: 'all' | 'active' | 'completed' | 'failed') => void;
  models: ModelSpec[];
  onSelectJob: (job: GenerationJobSummary) => void;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const modelName = (id: string) => formatModelName(props.models.find((model) => model.id === id)?.name || id);
  return (
    <section className="surface-page">
      <header className="page-head">
        <div><h1>Jobs</h1><p>Persistent generation history from the backend.</p></div>
        <div className="segmented strong">
          {(['all', 'active', 'completed', 'failed'] as const).map((filter) => <button key={filter} className={props.filter === filter ? 'active' : ''} onClick={() => props.setFilter(filter)}>{filter}</button>)}
        </div>
      </header>
      <div className="job-table">
        <div className="job-head"><span>Status</span><span>Text</span><span>Model</span><span>Format</span><span>When</span><span>Actions</span></div>
        {props.jobs.map((job) => (
          <div key={job.id} className="job-row">
            <span className={`status-pill ${statusTone(job.status)}`}>{phaseLabel(job.status)}</span>
            <button onClick={() => props.onSelectJob(job)}>{job.text || job.request?.text || 'Untitled generation'}</button>
            <span>{modelName(job.model_id)}</span>
            <code>{job.output_format?.toUpperCase()}</code>
            <span>{formatRelative(job.created_at)}</span>
            <span className="row-actions">
              {isActivePhase(job.status) && <button onClick={() => props.onCancel(job.id)}>Cancel</button>}
              {job.output && <a href={generationJobAudioUrl(job.id)} download>Audio</a>}
              <button onClick={() => props.onDelete(job.id)}>Delete</button>
            </span>
          </div>
        ))}
      </div>
      {props.jobs.length === 0 && <EmptyState title="No jobs in this filter" text="Generated audio will appear here with progress and output links." />}
    </section>
  );
}

function ModelsSurface({ models, statuses, settings, onUnload }: { models: ModelSpec[]; statuses: Record<string, ModelStatus>; settings: AppSettings; onUnload: (id: string) => void }) {
  const loaded = models.filter((model) => statuses[model.id]?.loaded).length;
  return (
    <section className="surface-page">
      <header className="page-head">
        <div><h1>Models & System</h1><p>{loaded} loaded · live status from hub worker manager.</p></div>
      </header>
      <div className="model-system-grid">
        {models.map((model) => {
          const status = statuses[model.id] || {};
          return (
            <article key={model.id} className="system-card">
              <header><strong>{formatModelName(model.name)}</strong><span className={`status-pill ${status.loaded ? 'good' : 'muted'}`}>{status.loaded ? 'LOADED' : 'UNLOADED'}</span></header>
              <p>{model.description}</p>
              <div className="system-metrics">
                <span><code>{status.device || 'unknown'}</code>DEVICE</span>
                <span><code>{status.total_generations || 0}</code>RUNS</span>
                <span><code>{formatMs(status.last_generation_duration_ms)}</code>LAST GEN</span>
              </div>
              <div className="chips">{capabilityTags(model.id, settings).map((tag) => <span key={tag}>{tag}</span>)}</div>
              <button disabled={!status.loaded} onClick={() => onUnload(model.id)}>{status.loaded ? 'Unload' : 'Load on generation'}</button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function WatermarkSurface(props: {
  runs: WatermarkRun[];
  defaultRun: string | null;
  selectedRun: string;
  setSelectedRun: (run: string) => void;
  details: WatermarkRunDetails | null;
  detectFile: File | null;
  setDetectFile: (file: File | null) => void;
  detectInputRef: React.RefObject<HTMLInputElement | null>;
  threshold: string;
  setThreshold: (value: string) => void;
  onDetect: () => void;
  result: DetectResult | null;
}) {
  return (
    <section className="surface-page watermark-page">
      <header className="page-head">
        <div><h1>Watermark Lab</h1><p>Inspect provenance runs and analyze audio against the selected detector.</p></div>
      </header>
      <div className="watermark-grid">
        <section className="lab-panel">
          <label>RUN</label>
          <select value={props.selectedRun} onChange={(event) => props.setSelectedRun(event.target.value)}>
            {props.runs.map((run) => <option key={run.id} value={run.id}>{run.label}{run.id === props.defaultRun ? ' (default)' : ''}</option>)}
          </select>
          <div className="run-list">
            {props.runs.slice(0, 20).map((run) => (
              <button key={run.id} className={props.selectedRun === run.id ? 'active' : ''} onClick={() => props.setSelectedRun(run.id)}>
                <strong>{run.label}</strong>
                <span>{run.status || 'unknown'} · {formatRelative(run.updated_at)}</span>
              </button>
            ))}
          </div>
        </section>
        <section className="lab-panel">
          <label>RUN DETAILS</label>
          <pre>{props.details ? JSON.stringify({
            id: props.details.id,
            status: props.details.status,
            metrics: props.details.metrics,
            error: props.details.error,
          }, null, 2) : 'No run selected.'}</pre>
        </section>
        <section className="lab-panel detect">
          <label>DETECT AUDIO</label>
          <input ref={props.detectInputRef} type="file" accept="audio/*" onChange={(event) => props.setDetectFile(event.target.files?.[0] || null)} />
          <div className="detect-controls">
            <input value={props.threshold} onChange={(event) => props.setThreshold(event.target.value)} aria-label="Watermark threshold" />
            <button onClick={props.onDetect}>Analyze</button>
          </div>
          <span>{props.detectFile?.name || 'No audio selected'}</span>
          {props.result && (
            <div className={props.result.detected ? 'detect-result detected' : 'detect-result'}>
              <strong>{props.result.detected ? 'Detected' : 'Not detected'}</strong>
              <code>p={props.result.wm_prob.toFixed(3)}</code>
              <span>{props.result.model?.name || props.result.model?.tts_model_id || 'Unknown model'}</span>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}

function OutputDock({ job, notice, models, onCancel }: { job: GenerationJobSummary | GenerationJobDetails | null; notice: Notice; models: ModelSpec[]; onCancel?: () => void }) {
  const model = job ? models.find((item) => item.id === job.model_id) : null;
  const audioUrl = job?.output ? generationJobAudioUrl(job.id) : '';
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    setIsPlaying(false);
    const el = audioRef.current;
    if (el) {
      el.pause();
      el.currentTime = 0;
    }
  }, [audioUrl]);

  const togglePlay = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      void el.play();
    } else {
      el.pause();
    }
  };

  const statusText = job ? (job.status === 'completed' ? 'DONE' : phaseLabel(job.status)) : '';

  return (
    <footer className="output-dock">
      <div className={`notice ${notice.kind}`}>
        <i className={notice.kind === 'error' ? 'dot bad' : notice.kind === 'success' ? 'dot good' : 'dot warn'} />
        <span>{notice.text}</span>
      </div>
      {job ? (
        <>
          <button
            className="transport"
            aria-label={job.output ? (isPlaying ? 'Pause output' : 'Play output') : 'Generating'}
            onClick={job.output ? togglePlay : undefined}
          >
            {job.output ? (isPlaying ? <span className="pause-bars" /> : <span className="play-triangle" />) : <span className="spinner" />}
          </button>
          <div className="output-meta">
            <strong>{job.output?.filename || `${formatModelName(model?.name || job.model_id)} job`}</strong>
            <span><em className={`status-pill ${statusTone(job.status)}`}>{statusText}</em>{job.watermark_enabled && <em className="status-pill watermark">WATERMARKED</em>}</span>
          </div>
          {job.output ? <Waveform /> : <div className="progress-line"><span /></div>}
          <div className="output-facts">{job.output ? `${formatDuration(job.output.duration_s)} · ${job.output.sample_rate || '--'}Hz · ${job.output.format.toUpperCase()}` : formatRelative(job.created_at)}</div>
          {audioUrl && (
            <audio
              ref={audioRef}
              src={audioUrl}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              hidden
            />
          )}
          {onCancel && <button className="dock-cancel" onClick={onCancel}>Cancel</button>}
          {audioUrl && <a className="download" href={audioUrl} download>Download</a>}
        </>
      ) : (
        <div className="output-empty">Generated audio will appear here.</div>
      )}
    </footer>
  );
}

function EmptyState({ title, text }: { title: string; text: string }) {
  return <div className="empty-state"><strong>{title}</strong><span>{text}</span></div>;
}
