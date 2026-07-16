import { expect, test } from '@playwright/test';

const models = [
  ['index-tts2', 'IndexTTS2', { emoMode: 'speaker', emoAlpha: 0.65, useRandom: false, emoVector: '[0,0,0,0,0,0,0.45,0]', emoText: '', maxTextTokens: 120, maxMelTokens: 1500, fastMode: false, doSample: true, temperature: 0.8, topP: 0.8, topK: 30, numBeams: 3, repetitionPenalty: 10, lengthPenalty: 0 }],
  ['qwen3-tts-mlx', 'Qwen3-TTS MLX', { model: 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit', autoTranscribe: true, language: 'auto', speed: 1, temperature: 0.7, maxTokens: 1200 }],
  ['chatterbox-multilingual', 'Chatterbox Multilingual', { language: 'hi', usePrompt: true, cfgWeight: 0.5, temperature: 0.8, exaggeration: 0.5, fastMode: false, enableChunking: true, maxChunkChars: 150, crossfadeMs: 50, enableDf: false, enableNovasr: false }],
  ['f5-hindi-urdu', 'F5 Hindi/Urdu', { romanMode: true, overridesEnabled: true, overridesText: '', crossFade: 0.15, nfeStep: 32, speed: 1, removeSilence: false, seed: -1 }],
  ['cosyvoice3-mlx', 'CosyVoice3-MLX', { model: '8bit', mode: 'zero_shot', language: 'auto', speed: 1, instructText: '' }],
  ['pocket-tts', 'Pocket TTS', { voice: 'hf://kyutai/tts-voices/alba-mackenna/casual.wav', temperature: 0.8, lsdDecodeSteps: 8, eosThreshold: 0.4, noiseClamp: '', truncatePrompt: false }],
  ['voxcpm-ane', 'VoxCPM-ANE', { voice: '', cfgValue: 2, inferenceTimesteps: 10, maxLength: 2048 }],
] as const;

const requiredKeys: Record<string, string[]> = {
  'index-tts2': ['emo_mode', 'emo_alpha', 'use_random', 'emo_vector', 'emo_text', 'max_text_tokens_per_segment', 'max_mel_tokens', 'fast_mode', 'do_sample', 'temperature', 'top_p', 'top_k', 'num_beams', 'repetition_penalty', 'length_penalty'],
  'qwen3-tts-mlx': ['qwen_model', 'auto_transcribe', 'qwen_language', 'qwen_speed', 'qwen_temperature', 'qwen_max_tokens'],
  'chatterbox-multilingual': ['language_id', 'use_prompt_audio', 'cfg_weight', 'temperature', 'exaggeration', 'fast_mode', 'enable_chunking', 'max_chunk_chars', 'crossfade_ms', 'enable_df', 'enable_novasr'],
  'f5-hindi-urdu': ['roman_mode', 'overrides_enabled', 'overrides_text', 'cross_fade_duration', 'nfe_step', 'speed', 'remove_silence', 'seed'],
  'cosyvoice3-mlx': ['cosy_model', 'mode', 'language', 'speed', 'instruct_text'],
  'pocket-tts': ['voice', 'temperature', 'lsd_decode_steps', 'eos_threshold', 'noise_clamp', 'truncate_prompt'],
  'voxcpm-ane': ['voice', 'cfg_value', 'inference_timesteps', 'max_length'],
};

async function mockMobileApi(page: import('@playwright/test').Page, jobs: unknown[] = []) {
  await page.route('**/api/models', route => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ models: models.map(([id, name, defaults]) => ({ id, name, description: name, capabilities: {}, defaults })) }),
  }));
  await page.route('**/api/status', route => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ models: Object.fromEntries(models.map(([id]) => [id, { loaded: false, device: 'test', total_generations: 0 }])) }),
  }));
  await page.route('**/api/voices', route => {
    if (route.request().method() !== 'GET') return route.continue();
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ voices: [{ id: 'a'.repeat(32), name: 'Test voice', duration_s: 2, created_at: 1, has_transcript: true, compatible_models: models.map(([id]) => id) }] }),
    });
  });
  await page.route('**/api/generation-jobs', route => {
    if (route.request().method() !== 'GET') return route.continue();
    return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ jobs }) });
  });
}

test.describe('mobile app', () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

  test('renders navigation and normalizes desktop job snapshots', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await mockMobileApi(page, [{
      id: 'b'.repeat(32), status: 'completed', created_at: Date.now() / 1000,
      model_id: 'qwen3-tts-mlx', voice_id: 'a'.repeat(32), text: 'Desktop snapshot text',
      output_format: 'mp3', watermark_enabled: false, worker_duration_ms: 1200,
      output: { format: 'mp3', filename: 'result.mp3', duration_s: 1.2 },
      request: { modelId: 'qwen3-tts-mlx', voiceId: 'a'.repeat(32), text: 'Desktop snapshot text', outputFormat: 'mp3', settings: { qwen: { speed: 1.25 } } },
    }]);
    await page.goto('/mobile/');
    await expect(page.locator('.tab')).toHaveCount(3);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    await page.locator('.tab[data-tab="jobs"]').click();
    await expect(page.locator('.job-card').first()).toContainText('Qwen3-TTS MLX');
    await expect(page.locator('.job-card').first()).toContainText('Desktop snapshot text');
    await page.locator('.job-card').first().click();
    await expect(page.locator('#sheet-body')).toContainText('Test voice');
    let confirmText = '';
    page.once('dialog', async dialog => { confirmText = dialog.message(); await dialog.dismiss(); });
    await page.getByRole('button', { name: 'Delete' }).click();
    expect(confirmText).toContain('cannot be undone');
    await page.getByRole('button', { name: 'Restore settings' }).click();
    await expect(page.locator('#script')).toHaveValue('Desktop snapshot text');
    await expect.poll(async () => page.evaluate(async () => (await navigator.serviceWorker.getRegistrations()).length)).toBeGreaterThan(0);
    expect(errors).toEqual([]);
  });

  test('submits exact worker fields and versioned snapshots for every model', async ({ page }) => {
    await mockMobileApi(page);
    const submissions: Array<{ model: string, body: string }> = [];
    await page.route('**/api/generation-jobs', async route => {
      if (route.request().method() === 'GET') {
        return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ jobs: [] }) });
      }
      const body = route.request().postDataBuffer()?.toString('utf8') || '';
      const model = body.match(/name="model_id"\r\n\r\n([^\r\n]+)/)?.[1] || '';
      submissions.push({ model, body });
      return route.fulfill({
        status: 202, contentType: 'application/json',
        body: JSON.stringify({ id: String(submissions.length).padStart(32, 'c'), status: 'completed', created_at: Date.now() / 1000, request: {} }),
      });
    });

    for (const [id, name] of models) {
      await page.goto('/mobile/');
      await page.locator('#model-row').click();
      await page.locator('.model-opt').filter({ hasText: name }).click();
      await page.locator('#script').fill(`Test ${id}`);
      await page.locator('#run-btn').click();
      await expect.poll(() => submissions.filter(item => item.model === id).length).toBe(1);
    }

    for (const { model, body } of submissions) {
      for (const key of requiredKeys[model]) expect(body, `${model} should submit ${key}`).toContain(`name="${key}"`);
      expect(body).toContain('"schemaVersion":2');
      expect(body).toContain('"source":"mobile"');
      expect(body).toContain('"settings"');
    }
  });

  test('keeps text while polling and persists theme', async ({ page }) => {
    await mockMobileApi(page);
    await page.goto('/mobile/');
    await page.locator('#script').fill('Do not erase this');
    await page.evaluate(async () => {
      await fetch('/api/generation-jobs');
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#script')).toHaveValue('Do not erase this');
    await page.locator('#theme-toggle').click();
    await expect(page.locator('#app')).toHaveAttribute('data-theme', 'light');
    await page.reload();
    await expect(page.locator('#app')).toHaveAttribute('data-theme', 'light');
  });

  test('blocks transcript-dependent models until a transcript is available', async ({ page }) => {
    await mockMobileApi(page);
    await page.route('**/api/voices', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ voices: [{ id: 'a'.repeat(32), name: 'No transcript', duration_s: 2, created_at: 1, has_transcript: false, compatible_models: models.map(([id]) => id) }] }),
    }));
    await page.goto('/mobile/');
    await page.locator('#model-row').click();
    await page.locator('.model-opt').filter({ hasText: 'F5 Hindi/Urdu' }).click();
    await page.locator('#script').fill('Transcript requirement');
    await expect(page.locator('#block-reason')).toContainText('transcript is required');
    await page.locator('#prompt-text').fill('This is the reference transcript.');
    await expect(page.locator('#block-reason')).toBeHidden();
  });
});
