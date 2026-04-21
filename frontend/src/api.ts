import type {
  AppHistoryItem,
  ModelSpec,
  ModelStatus,
  VoiceSummary,
  WatermarkRun,
  WatermarkRunDetails,
} from './types';

async function jsonRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((data as { error?: string }).error || `HTTP ${response.status}`);
  }
  return data as T;
}

export interface ModelsResponse {
  models: ModelSpec[];
}

export interface VoicesResponse {
  voices: VoiceSummary[];
}

export interface WatermarkRunsResponse {
  default_run_id: string | null;
  runs: WatermarkRun[];
}

export interface InfoResponse {
  ffmpeg: {
    available: boolean;
  };
  time: number;
}

export interface StatusResponse {
  models: Record<string, ModelStatus>;
}

export async function fetchModels(): Promise<ModelSpec[]> {
  const data = await jsonRequest<ModelsResponse>('/api/models');
  return data.models || [];
}

export async function fetchInfo(): Promise<InfoResponse> {
  return await jsonRequest<InfoResponse>('/api/info');
}

export async function fetchVoices(): Promise<VoiceSummary[]> {
  const data = await jsonRequest<VoicesResponse>('/api/voices');
  return data.voices || [];
}

export async function fetchWatermarkRuns(): Promise<WatermarkRunsResponse> {
  return await jsonRequest<WatermarkRunsResponse>('/api/watermark/runs');
}

export async function fetchWatermarkRunDetails(runId?: string | null): Promise<WatermarkRunDetails> {
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  return await jsonRequest<WatermarkRunDetails>(`/api/watermark/run_details${qs}`);
}

export async function fetchStatus(modelId?: string): Promise<StatusResponse> {
  const qs = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
  return await jsonRequest<StatusResponse>(`/api/status${qs}`);
}

export async function unloadModel(modelId: string): Promise<void> {
  const form = new FormData();
  form.append('model_id', modelId);
  await jsonRequest('/api/unload', { method: 'POST', body: form });
}

export async function createVoice(form: FormData): Promise<Record<string, unknown>> {
  return await jsonRequest<Record<string, unknown>>('/api/voices', { method: 'POST', body: form });
}

export async function deleteVoice(voiceId: string): Promise<void> {
  await jsonRequest(`/api/voices/${encodeURIComponent(voiceId)}`, { method: 'DELETE' });
}

export async function getVoiceMeta(voiceId: string): Promise<Record<string, unknown>> {
  return await jsonRequest<Record<string, unknown>>(`/api/voices/${encodeURIComponent(voiceId)}`);
}

export interface GenerateResult {
  blob: Blob;
  headers: Headers;
}

export async function generateAudio(form: FormData): Promise<GenerateResult> {
  const response = await fetch('/api/generate', { method: 'POST', body: form });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error((data as { error?: string }).error || `HTTP ${response.status}`);
  }
  return {
    blob: await response.blob(),
    headers: response.headers,
  };
}

export type { AppHistoryItem };
