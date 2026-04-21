import type { AppSettings } from './types';

export const MODEL_ICONS: Record<string, string> = {
  'index-tts2': '🎭',
  'qwen3-tts-mlx': '🗣️',
  'chatterbox-multilingual': '🌍',
  'f5-hindi-urdu': '🇮🇳',
  'cosyvoice3-mlx': '🍎',
  'pocket-tts': '⚡',
  'voxcpm-ane': '🧠',
};

export const WATERMARK_MODEL_MAP: Record<string, number> = {
  'index-tts2': 0,
  'qwen3-tts-mlx': 1,
};

export function isWatermarkSupported(modelId: string | null | undefined): boolean {
  return !!modelId && Object.prototype.hasOwnProperty.call(WATERMARK_MODEL_MAP, modelId);
}

export function requiresReferenceAudio(modelId: string | null | undefined): boolean {
  return ['index-tts2', 'f5-hindi-urdu', 'cosyvoice3-mlx', 'qwen3-tts-mlx'].includes(String(modelId || ''));
}

export function requiresTranscript(modelId: string | null | undefined, settings: AppSettings): boolean {
  const id = String(modelId || '');
  if (id === 'f5-hindi-urdu') return true;
  if (id === 'cosyvoice3-mlx') return settings.cosy.mode === 'zero_shot' || settings.cosy.mode === 'instruct';
  if (id === 'voxcpm-ane') return true;
  return false;
}

export function buildDefaultSettings(): AppSettings {
  return {
    index: {
      emoMode: 'speaker',
      emoAlpha: 0.65,
      useRandom: false,
      emoVector: '[0,0,0,0,0,0,0.45,0]',
      emoText: '',
      maxTextTokens: 120,
      maxMelTokens: 1500,
      fastMode: false,
      doSample: true,
      temperature: 0.8,
      topP: 0.8,
      topK: 30,
      numBeams: 3,
      repetitionPenalty: 10,
      lengthPenalty: 0,
    },
    chatterbox: {
      language: 'hi',
      usePrompt: true,
      cfgWeight: 0.5,
      temperature: 0.8,
      exaggeration: 0.5,
      fastMode: false,
      enableChunking: true,
      maxChunkChars: 150,
      crossfadeMs: 50,
      enableDf: false,
      enableNovasr: false,
    },
    f5: {
      romanMode: true,
      overridesEnabled: true,
      overridesText: '',
      crossFade: 0.15,
      nfeStep: 32,
      speed: 1,
      removeSilence: false,
      seed: -1,
    },
    cosy: {
      model: '8bit',
      mode: 'zero_shot',
      language: 'auto',
      speed: 1,
      instructText: '',
    },
    qwen: {
      model: 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit',
      autoTranscribe: true,
      language: 'auto',
      speed: 1,
      temperature: 0.7,
      maxTokens: 1200,
    },
    pocket: {
      voice: 'hf://kyutai/tts-voices/alba-mackenna/casual.wav',
      temperature: 0.8,
      lsdDecodeSteps: 8,
      eosThreshold: 0.4,
      noiseClamp: '',
      truncatePrompt: false,
    },
    voxcpm: {
      voice: '',
      cfgValue: 2,
      inferenceTimesteps: 10,
      maxLength: 2048,
    },
  };
}

export function appendModelParams(
  form: FormData,
  modelId: string,
  settings: AppSettings,
  extra?: { emotionAudio?: File | null },
): void {
  if (modelId === 'index-tts2') {
    const s = settings.index;
    form.append('emo_mode', s.emoMode);
    form.append('emo_alpha', String(s.emoAlpha));
    form.append('use_random', String(s.useRandom));
    form.append('max_text_tokens_per_segment', String(s.maxTextTokens));
    form.append('max_mel_tokens', String(s.maxMelTokens));
    form.append('fast_mode', String(s.fastMode));
    if (s.emoMode === 'emo_vector') form.append('emo_vector', s.emoVector.trim());
    if (s.emoMode === 'emo_text') form.append('emo_text', s.emoText.trim());
    form.append('do_sample', String(s.doSample));
    form.append('temperature', String(s.temperature));
    form.append('top_p', String(s.topP));
    form.append('top_k', String(s.topK));
    form.append('num_beams', String(s.numBeams));
    form.append('repetition_penalty', String(s.repetitionPenalty));
    form.append('length_penalty', String(s.lengthPenalty));
    if (s.emoMode === 'emo_ref' && extra?.emotionAudio) {
      form.append('emo_audio', extra.emotionAudio);
    }
    return;
  }

  if (modelId === 'chatterbox-multilingual') {
    const s = settings.chatterbox;
    form.append('language_id', s.language);
    form.append('use_prompt_audio', String(s.usePrompt));
    form.append('cfg_weight', String(s.cfgWeight));
    form.append('temperature', String(s.temperature));
    form.append('exaggeration', String(s.exaggeration));
    form.append('fast_mode', String(s.fastMode));
    form.append('enable_chunking', String(s.enableChunking));
    form.append('max_chunk_chars', String(s.maxChunkChars));
    form.append('crossfade_ms', String(s.crossfadeMs));
    form.append('enable_df', String(s.enableDf));
    form.append('enable_novasr', String(s.enableNovasr));
    return;
  }

  if (modelId === 'f5-hindi-urdu') {
    const s = settings.f5;
    form.append('roman_mode', String(s.romanMode));
    form.append('overrides_enabled', String(s.overridesEnabled));
    form.append('overrides_text', s.overridesText || '');
    form.append('cross_fade_duration', String(s.crossFade));
    form.append('nfe_step', String(s.nfeStep));
    form.append('speed', String(s.speed));
    form.append('remove_silence', String(s.removeSilence));
    form.append('seed', String(s.seed));
    return;
  }

  if (modelId === 'cosyvoice3-mlx') {
    const s = settings.cosy;
    form.append('cosy_model', s.model);
    form.append('mode', s.mode);
    form.append('language', s.language.trim() || 'auto');
    form.append('speed', String(s.speed));
    if (s.mode === 'instruct') form.append('instruct_text', s.instructText.trim());
    return;
  }

  if (modelId === 'qwen3-tts-mlx') {
    const s = settings.qwen;
    form.append('qwen_model', s.model);
    form.append('auto_transcribe', s.autoTranscribe ? '1' : '0');
    form.append('qwen_language', s.language.trim() || 'auto');
    form.append('qwen_speed', String(s.speed));
    form.append('qwen_temperature', String(s.temperature));
    form.append('qwen_max_tokens', String(s.maxTokens));
    return;
  }

  if (modelId === 'pocket-tts') {
    const s = settings.pocket;
    form.append('voice', s.voice.trim());
    form.append('temperature', String(s.temperature));
    form.append('lsd_decode_steps', String(s.lsdDecodeSteps));
    form.append('eos_threshold', String(s.eosThreshold));
    if (s.noiseClamp.trim()) form.append('noise_clamp', s.noiseClamp.trim());
    form.append('truncate_prompt', String(s.truncatePrompt));
    return;
  }

  if (modelId === 'voxcpm-ane') {
    const s = settings.voxcpm;
    if (s.voice.trim()) form.append('voice', s.voice.trim());
    form.append('cfg_value', String(s.cfgValue));
    form.append('inference_timesteps', String(s.inferenceTimesteps));
    form.append('max_length', String(s.maxLength));
  }
}
