export type SurfaceId =
  | 'generate'
  | 'models'
  | 'voices'
  | 'history'
  | 'watermark-lab'
  | 'system-status'
  | 'advanced-settings';

export interface ModelSpec {
  id: string;
  name: string;
  description: string;
  worker_entry?: string;
}

export interface ModelStatus {
  loaded?: boolean;
  device?: string;
  last_generation_time?: number | null;
  last_generation_duration_ms?: number | null;
  total_generations?: number;
}

export interface VoiceSummary {
  id: string;
  name: string;
  created_at: number;
  duration_s: number;
  has_caches: Record<string, boolean>;
  has_transcript: boolean;
  compatible_models: string[];
}

export type GenerationPhase =
  | 'queued'
  | 'preparing'
  | 'generating'
  | 'watermarking'
  | 'converting'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface GenerationRequestSnapshot {
  modelId: string;
  text: string;
  promptText: string;
  voiceId: string | null;
  outputFormat: 'wav' | 'mp3' | 'flac';
  watermarkEnabled: boolean;
  watermarkRun: string | null;
  settings: AppSettings;
}

export interface OutputAudioMetadata {
  path: string;
  filename: string;
  format: 'wav' | 'mp3' | 'flac';
  duration_s?: number | null;
  sample_rate?: number | null;
}

export interface GenerationJobSummary {
  id: string;
  status: GenerationPhase;
  phase: GenerationPhase;
  created_at: number;
  updated_at: number;
  started_at?: number | null;
  completed_at?: number | null;
  error?: string | null;
  model_id: string;
  voice_id?: string | null;
  text: string;
  output_format: 'wav' | 'mp3' | 'flac';
  watermark_enabled: boolean;
  watermark_run?: string | null;
  worker_duration_ms?: number | null;
  output?: OutputAudioMetadata | null;
  request?: Partial<GenerationRequestSnapshot>;
}

export interface GenerationJobDetails extends GenerationJobSummary {
  worker_meta?: Record<string, unknown>;
}

export interface WatermarkRun {
  id: string;
  label: string;
  status?: string | null;
  updated_at?: number | null;
}

export interface WatermarkRunDetails {
  id: string;
  label?: string | null;
  status?: string | null;
  updated_at?: number | null;
  report_excerpt?: string | null;
  metrics?: Record<string, unknown> | null;
  config?: Record<string, unknown> | null;
  error?: string;
}

export interface AppHistoryItem {
  modelId: string;
  timestamp: number;
  format: 'wav' | 'mp3' | 'flac';
  url?: string | null;
  watermarkEnabled?: boolean;
  watermarkRun?: string | null;
  voiceId?: string | null;
  settingsSummary?: string | null;
}

export interface AppSettings {
  index: {
    emoMode: 'speaker' | 'emo_ref' | 'emo_vector' | 'emo_text';
    emoAlpha: number;
    useRandom: boolean;
    emoVector: string;
    emoText: string;
    maxTextTokens: number;
    maxMelTokens: number;
    fastMode: boolean;
    doSample: boolean;
    temperature: number;
    topP: number;
    topK: number;
    numBeams: number;
    repetitionPenalty: number;
    lengthPenalty: number;
  };
  chatterbox: {
    language: string;
    usePrompt: boolean;
    cfgWeight: number;
    temperature: number;
    exaggeration: number;
    fastMode: boolean;
    enableChunking: boolean;
    maxChunkChars: number;
    crossfadeMs: number;
    enableDf: boolean;
    enableNovasr: boolean;
  };
  f5: {
    romanMode: boolean;
    overridesEnabled: boolean;
    overridesText: string;
    crossFade: number;
    nfeStep: number;
    speed: number;
    removeSilence: boolean;
    seed: number;
  };
  cosy: {
    model: string;
    mode: 'zero_shot' | 'cross_lingual' | 'instruct';
    language: string;
    speed: number;
    instructText: string;
  };
  qwen: {
    model: string;
    autoTranscribe: boolean;
    language: string;
    speed: number;
    temperature: number;
    maxTokens: number;
  };
  pocket: {
    voice: string;
    temperature: number;
    lsdDecodeSteps: number;
    eosThreshold: number;
    noiseClamp: string;
    truncatePrompt: boolean;
  };
  voxcpm: {
    voice: string;
    cfgValue: number;
    inferenceTimesteps: number;
    maxLength: number;
  };
}

export interface AppSnapshot {
  selectedModelId: string | null;
  activeSurface: SurfaceId;
  text: string;
  promptText: string;
  selectedVoiceId: string;
  outputFormat: 'wav' | 'mp3' | 'flac';
  watermarkEnabled: boolean;
  watermarkRun: string;
  watermarkThresholdAuto: boolean;
  watermarkThresholdManual: string;
  settings: AppSettings;
}
