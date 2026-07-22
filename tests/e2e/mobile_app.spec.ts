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

function silentWav(): Buffer {
  const sampleRate = 8000;
  const samples = 2000;
  const dataSize = samples * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVEfmt ', 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);
  return buffer;
}

async function mockMobileApi(page: import('@playwright/test').Page, jobs: unknown[] = []) {
  await page.route('**/api/models', route => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ models: models.map(([id, name, defaults]) => ({ id, name, description: name, capabilities: {}, defaults })) }),
  }));
  await page.route('**/api/status', route => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ models: Object.fromEntries(models.map(([id]) => [id, { loaded: false, device: 'test', total_generations: 0 }])) }),
  }));
  await page.route('**/api/watermark/runs', route => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({
      default_run_id: 'run-a',
      runs: [
        { id: 'run-a', label: 'Primary detector', status: 'completed', updated_at: 1 },
        { id: 'run-b', label: 'Robustness detector', status: 'completed', updated_at: 2 },
      ],
    }),
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
  await page.route('**/api/generation-jobs/*/audio', route => route.fulfill({
    contentType: 'audio/wav',
    body: silentWav(),
  }));
  await page.route('**/api/voices/*/audio', route => route.fulfill({
    contentType: 'audio/wav',
    body: silentWav(),
  }));
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
    await expect(page.locator('.tab')).toHaveCount(4);
    await expect(page.locator('#miniplayer')).toBeHidden();
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    await page.locator('.tab[data-tab="jobs"]').click();
    await expect(page.locator('.job-card').first()).toContainText('Qwen3-TTS MLX');
    await expect(page.locator('.job-card').first()).toContainText('Desktop snapshot text');
    await page.locator('.job-card').first().click();
    await expect(page.locator('#sheet-body')).toContainText('Test voice');
    await page.getByRole('button', { name: 'Delete' }).click();
    await expect(page.locator('#confirm-backdrop')).toBeVisible();
    await expect(page.locator('#confirm-backdrop')).toContainText('generated audio file');
    await expect(page.locator('#confirm-backdrop')).toContainText('cannot be undone');
    await page.getByRole('button', { name: 'Keep it' }).click();
    await page.getByRole('button', { name: 'Restore settings' }).click();
    await expect(page.locator('#script')).toHaveValue('Desktop snapshot text');
    await page.locator('.tab[data-tab="jobs"]').click();
    await page.locator('.job-card').first().click();
    await page.getByRole('button', { name: 'Play' }).click();
    await expect(page.locator('#miniplayer')).toBeVisible();
    await page.locator('.tab[data-tab="voices"]').click();
    await expect(page.locator('#miniplayer')).toBeHidden();
    await expect.poll(async () => page.evaluate(async () => (await navigator.serviceWorker.getRegistrations()).length)).toBeGreaterThan(0);
    expect(errors).toEqual([]);
  });

  test('provides touch-first tutorials for every mobile surface', async ({ page }) => {
    const now = Date.now() / 1000;
    await mockMobileApi(page, [{
      id: 'f'.repeat(32), status: 'completed', created_at: now, updated_at: now,
      model_id: 'qwen3-tts-mlx', voice_id: 'a'.repeat(32), text: 'Please wait a moment.',
      output_format: 'wav', watermark_enabled: false, favorite: true, label: 'Please wait', favorited_at: now,
      output: { format: 'wav', filename: 'phrase.wav', duration_s: 1.2 }, request: {},
    }]);
    await page.goto('/mobile/');
    await page.getByRole('button', { name: 'Play quick phrase: Please wait' }).click();
    await expect(page.locator('#miniplayer')).toBeVisible();

    await page.getByRole('button', { name: 'Show tutorial' }).click();
    const tutorial = page.locator('.mobile-tour');
    const card = page.locator('.mobile-tour-card');
    await expect(tutorial).toBeVisible();
    await expect(card).toContainText('Move around the mobile hub');
    await expect(card.locator('.mobile-tour-count')).toHaveText('1 / 9');
    expect(await card.getByRole('button', { name: 'Next' }).evaluate(button => button.getBoundingClientRect().height)).toBeGreaterThanOrEqual(44);
    await expect(card).toHaveClass(/top/);

    for (let step = 0; step < 6; step += 1) await card.getByRole('button', { name: 'Next' }).click();
    await expect(card).toContainText('Save a Quick Phrase');
    await expect.poll(async () => {
      const target = await page.locator('#mp-favorite').boundingBox();
      const spot = await page.locator('.mobile-tour-spotlight').boundingBox();
      if (!target || !spot) return false;
      return Math.abs((target.x + target.width / 2) - (spot.x + spot.width / 2)) < 3;
    }).toBe(true);

    await card.getByRole('button', { name: 'Next' }).click();
    await expect(card).toContainText('Play a Quick Phrase');
    await expect.poll(async () => {
      const target = await page.locator('#phrase-section').boundingBox();
      const spot = await page.locator('.mobile-tour-spotlight').boundingBox();
      if (!target || !spot) return false;
      return Math.abs((target.x + target.width / 2) - (spot.x + spot.width / 2)) < 3;
    }).toBe(true);
    await card.getByRole('button', { name: 'Exit tutorial' }).click();
    await expect(tutorial).toBeHidden();

    await page.locator('.tab[data-tab="voices"]').click();
    await page.getByRole('button', { name: 'Show tutorial' }).click();
    await expect(card).toContainText('Add a reference voice');
    await expect(card.locator('.mobile-tour-count')).toHaveText('1 / 2');
    await card.getByRole('button', { name: 'Exit tutorial' }).click();

    await page.locator('.tab[data-tab="jobs"]').click();
    await page.getByRole('button', { name: 'Show tutorial' }).click();
    await expect(card).toContainText('Filter generation history');
    await expect(card.locator('.mobile-tour-count')).toHaveText('1 / 2');
    await card.getByRole('button', { name: 'Exit tutorial' }).click();

    await page.locator('.tab[data-tab="verify"]').click();
    await page.getByRole('button', { name: 'Show tutorial' }).click();
    await expect(card).toContainText('Choose verification audio');
    await expect(card.locator('.mobile-tour-count')).toHaveText('1 / 3');
    await page.keyboard.press('Escape');
    await expect(tutorial).toBeHidden();
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
    await expect(page.locator('#model-name')).toHaveText('Qwen3-TTS MLX');
    await expect(page.locator('#surface-generate .sec-label').filter({ hasText: /^Model$/ })).toBeVisible();
    await expect(page.locator('#model-recommended')).toBeVisible();
    await expect(page.locator('#model-description')).toHaveText('Fast and reliable for daily use.');
    await page.locator('#model-row').click();
    await expect(page.locator('.model-opt').first()).toContainText('Qwen3-TTS MLX');
    await expect(page.locator('.model-opt').first()).toContainText('Recommended');
    await expect(page.locator('.model-opt').filter({ hasText: 'IndexTTS2' })).toContainText('Best quality and expressive control.');
    await expect(page.locator('.model-opt').filter({ hasText: 'Qwen3-TTS MLX' })).toContainText('Fast and reliable for daily use.');
    await expect(page.locator('.model-opt').filter({ hasText: 'Chatterbox Multilingual' })).toContainText('Multilingual and long-form speech.');
    await expect(page.locator('.mo-guidance').first()).toHaveCSS('white-space', 'nowrap');
    await expect(page.locator('#sheet-body')).not.toContainText(/(?:loaded|idle)\s*·/);
    await page.locator('.model-opt').filter({ hasText: 'IndexTTS2' }).click();
    await page.reload();
    await expect(page.locator('#model-name')).toHaveText('IndexTTS2');
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

  test('saves, renames, filters, and instantly replays quick phrases while gating active runs', async ({ page }) => {
    const id = 'b'.repeat(32);
    const now = Date.now() / 1000;
    let currentJob: any = {
      id, status: 'completed', created_at: now, model_id: 'qwen3-tts-mlx', voice_id: 'a'.repeat(32),
      text: 'Please give me a moment.', output_format: 'wav', favorite: false, label: null,
      output: { format: 'wav', filename: 'phrase.wav', duration_s: 1.2 }, request: {},
    };
    let generationPosts = 0;
    let audioGets = 0;
    const patches: any[] = [];
    page.on('request', request => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/generation-jobs') generationPosts += 1;
    });
    await mockMobileApi(page, [currentJob]);
    await page.route(`**/api/generation-jobs/${id}`, async route => {
      if (route.request().method() !== 'PATCH') return route.continue();
      const patch = route.request().postDataJSON();
      patches.push(patch);
      currentJob = {
        ...currentJob,
        ...patch,
        label: Object.prototype.hasOwnProperty.call(patch, 'label') ? patch.label : currentJob.label,
        favorited_at: patch.favorite === true ? now + patches.length : patch.favorite === false ? null : currentJob.favorited_at,
      };
      return route.fulfill({ contentType: 'application/json', body: JSON.stringify(currentJob) });
    });
    await page.route(`**/api/generation-jobs/${id}/audio`, route => {
      audioGets += 1;
      return route.fulfill({ contentType: 'audio/wav', body: silentWav() });
    });
    await page.route('**/api/generation-jobs', route => {
      if (route.request().method() !== 'GET') return route.continue();
      return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ jobs: [currentJob] }) });
    });
    await page.goto('/mobile/');
    await page.locator('.tab[data-tab="jobs"]').click();
    await page.locator('.job-card').first().click();
    await page.getByRole('button', { name: 'Save phrase' }).click();
    const phraseDialog = page.locator('#phrase-dialog');
    await expect(phraseDialog).toBeVisible();
    await expect(phraseDialog).toContainText('Save quick phrase');
    await phraseDialog.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.locator('#phrase-dialog-backdrop')).toBeHidden();
    await page.getByRole('button', { name: 'Save phrase' }).click();
    await page.locator('#phrase-dialog-input').fill('Need a moment');
    await page.locator('#phrase-dialog-input').press('Enter');
    await expect(page.locator('#sheet-body button').filter({ hasText: 'Saved' })).toBeVisible();
    await expect(page.locator('#sheet-title')).toContainText('Need a moment');

    await page.getByRole('button', { name: 'Rename' }).click();
    await expect(phraseDialog).toContainText('Rename run');
    await page.locator('#phrase-dialog-input').fill('Please wait');
    await phraseDialog.getByRole('button', { name: 'Rename' }).click();
    await expect(page.locator('#sheet-title')).toContainText('Please wait');
    await page.locator('#sheet-close').click();
    await page.locator('.filter-btn').filter({ hasText: 'Saved' }).click();
    await expect(page.locator('.job-card')).toHaveCount(1);
    await expect(page.locator('.job-card').first()).toContainText('Please wait');

    await page.locator('.job-card').first().click();
    await expect(page.locator('#sheet-backdrop')).toBeVisible();
    await page.locator('#sheet-body .dj-play').click();
    await expect(page.locator('#surface-jobs')).toBeVisible();
    await expect(page.locator('#miniplayer')).toBeVisible();
    await expect(page.locator('#sheet-backdrop')).toBeHidden();
    await expect(page.locator('#mp-favorite')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('#mp-favorite').click();
    await expect(page.locator('#mp-favorite')).toHaveAttribute('aria-pressed', 'false');
    await page.locator('#mp-favorite').click();
    await page.locator('#phrase-dialog-input').fill('Please wait');
    await phraseDialog.getByRole('button', { name: 'Save phrase' }).click();
    await expect(page.locator('#mp-favorite')).toHaveAttribute('aria-pressed', 'true');
    const playerLayout = await page.locator('#miniplayer').evaluate(player => {
      const playerBox = player.getBoundingClientRect();
      const waveBox = player.querySelector('#mp-wave')!.getBoundingClientRect();
      const downloadBox = player.querySelector('#mp-download')!.getBoundingClientRect();
      return {
        playerLeft: playerBox.left,
        playerRight: playerBox.right,
        waveWidth: waveBox.width,
        downloadRight: downloadBox.right,
        viewportWidth: window.innerWidth,
      };
    });
    expect(playerLayout.playerLeft).toBeGreaterThanOrEqual(0);
    expect(playerLayout.playerRight).toBeLessThanOrEqual(playerLayout.viewportWidth);
    expect(playerLayout.downloadRight).toBeLessThanOrEqual(playerLayout.viewportWidth - 12);
    expect(playerLayout.waveWidth).toBeGreaterThan(150);

    await page.locator('.tab[data-tab="generate"]').click();
    await expect(page.locator('#phrase-section')).toBeVisible();
    await page.locator('.phrase-chip').filter({ hasText: 'Please wait' }).click();
    await expect.poll(() => audioGets).toBeGreaterThan(0);
    expect(generationPosts).toBe(0);
    expect(patches).toEqual([
      { favorite: true, label: 'Need a moment' },
      { label: 'Please wait' },
      { favorite: false },
      { favorite: true, label: 'Please wait' },
    ]);

    const queued = { ...currentJob, id: 'c'.repeat(32), status: 'queued', output: null, favorite: false };
    await page.unroute('**/api/generation-jobs');
    await page.route('**/api/generation-jobs', route => {
      if (route.request().method() !== 'GET') return route.continue();
      return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ jobs: [queued, currentJob] }) });
    });
    await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
    await expect(page.locator('#run-label')).toHaveText('Generating…');
    await expect(page.locator('#run-btn')).toBeDisabled();

    const sw = await page.request.get('/mobile/sw.js');
    expect(await sw.text()).toContain('tts-hub-mobile-v19');
    expect(await sw.text()).toContain('/mobile/icon-maskable-512.png');
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
    await expect(page.locator('#block-reason')).toContainText('Edit it in Voices');
    await expect(page.locator('#prompt-text-wrap')).toBeHidden();
    await page.getByRole('button', { name: 'Upload' }).click();
    await page.locator('#ref-file-input').setInputFiles({
      name: 'reference.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.from('RIFF-test-audio'),
    });
    await expect(page.locator('#prompt-text-wrap')).toBeVisible();
    await page.locator('#prompt-text').fill('This is the reference transcript.');
    await expect(page.locator('#block-reason')).toBeHidden();
  });

  test('edits a saved voice with name, transcript, and replacement audio', async ({ page }) => {
    await mockMobileApi(page);
    let patchBody = '';
    await page.route(`**/api/voices/${'a'.repeat(32)}`, async route => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          contentType: 'application/json',
          body: JSON.stringify({ id: 'a'.repeat(32), name: 'Test voice', prompt_text: 'Old transcript' }),
        });
      }
      patchBody = route.request().postDataBuffer()?.toString('utf8') || '';
      return route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ id: 'a'.repeat(32), name: 'Edited voice' }),
      });
    });
    await page.goto('/mobile/');
    await page.locator('.tab[data-tab="voices"]').click();
    await page.getByRole('button', { name: 'Edit' }).click();
    await page.locator('#sheet-body input[placeholder="Voice name"]').fill('Edited voice');
    await page.locator('#sheet-body textarea').fill('Updated transcript');
    await page.locator('#edit-file-input').setInputFiles({
      name: 'replacement.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.from('RIFF-replacement'),
    });
    await page.getByRole('button', { name: 'Save changes' }).click();
    await expect.poll(() => patchBody).toContain('Edited voice');
    expect(patchBody).toContain('Updated transcript');
    expect(patchBody).toContain('replacement.wav');
  });

  test('shows honest queue progress and polished priority settings', async ({ page }) => {
    const now = Date.now() / 1000;
    await mockMobileApi(page, [
      { id: 'c'.repeat(32), status: 'queued', phase: 'queued', created_at: now - 3, model_id: 'index-tts2', text: 'Queued run', request: {} },
      { id: 'd'.repeat(32), status: 'queued', phase: 'queued', created_at: now - 1, model_id: 'qwen3-tts-mlx', text: 'Second run', request: {} },
    ]);
    await page.goto('/mobile/');
    await expect(page.locator('#active-phase')).toHaveText('Waiting in queue');
    await expect(page.locator('#active-meta')).toContainText('position 1 of 2');
    await expect(page.locator('#active-detail')).toContainText('earlier jobs');

    await page.locator('#model-row').click();
    await page.locator('.model-opt').filter({ hasText: 'IndexTTS2' }).click();
    await page.locator('#options-row').click();
    await page.locator('#sheet-body select').first().selectOption('emo_vector');
    await expect(page.locator('.emotion-item')).toHaveCount(8);
    const happy = page.locator('.emotion-item').filter({ hasText: 'Happy' });
    await expect(happy.locator('input[type="range"]')).toHaveValue('0');
    await happy.locator('input[type="range"]').fill('0.75');
    await expect(happy).toContainText('0.75');

    await page.locator('#sheet-close').click();
    await page.locator('#model-row').click();
    await page.locator('.model-opt').filter({ hasText: 'Qwen3-TTS MLX' }).click();
    await page.locator('#options-row').click();
    await expect(page.locator('#sheet-body input[type="range"]')).toHaveCount(2);
    await expect(page.locator('#sheet-body')).toContainText('Auto-transcribe reference');
    await expect(page.locator('#sheet-body')).toContainText('Qwen derives the transcript');

    await page.locator('#sheet-close').click();
    await page.locator('#model-row').click();
    await page.locator('.model-opt').filter({ hasText: 'Chatterbox Multilingual' }).click();
    await page.locator('#options-row').click();
    await expect(page.locator('#sheet-body select').first()).toHaveValue('hi');
    await expect(page.locator('#sheet-body input[type="range"]')).toHaveCount(3);
    await expect(page.locator('#sheet-body')).toContainText('Exaggeration');
  });

  test('verifies uploaded audio with advanced controls and invalidates stale results', async ({ page }) => {
    await mockMobileApi(page);
    let detectBody = '';
    let detectResult: any = {
      detected: true,
      wm_prob: 0.913,
      model: { id: 0, name: 'IndexTTS2', tts_model_id: 'index-tts2' },
      run: { id: 'run-b' },
    };
    await page.route('**/api/watermark/detect', async route => {
      detectBody = route.request().postDataBuffer()?.toString('utf8') || '';
      await new Promise(resolve => setTimeout(resolve, 80));
      return route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify(detectResult),
      });
    });
    await page.goto('/mobile/');
    await page.locator('.tab[data-tab="verify"]').click();
    await expect(page.locator('#surface-verify')).toBeVisible();
    await expect(page.locator('#miniplayer')).toBeHidden();
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    await expect(page.locator('#verify-advanced-body')).toBeHidden();
    await page.locator('#verify-file-input').setInputFiles({ name: 'external.wav', mimeType: 'audio/wav', buffer: silentWav() });
    await expect(page.locator('#verify-file-staged')).toBeVisible();
    await page.locator('#verify-advanced-toggle').click();
    await page.locator('#verify-run-select').selectOption('run-b');
    await page.locator('#verify-threshold').fill('0.42');
    await page.locator('#verify-btn').click();
    await expect(page.locator('#verify-btn-label')).toHaveText('Checking audio…');
    await expect(page.locator('#verify-result')).toContainText('Watermark detected');
    await expect(page.locator('#verify-result')).toContainText('Likely generated by IndexTTS2');
    await expect(page.locator('#verify-result')).not.toContainText('Clear result');
    await expect(page.locator('#verify-result')).toContainText('IndexTTS2');
    await expect(page.locator('.verify-result-facts')).toBeHidden();
    await page.getByText('View technical details', { exact: true }).click();
    await expect(page.locator('.verify-result-facts')).toBeVisible();
    await expect(page.locator('.verify-result-facts')).toContainText('0.913');
    expect(detectBody).toContain('name="audio"');
    expect(detectBody).toContain('external.wav');
    expect(detectBody).toContain('name="watermark_run"');
    expect(detectBody).toContain('run-b');
    expect(detectBody).toContain('name="wm_threshold"');
    expect(detectBody).toContain('0.42');

    detectResult = { detected: false, wm_prob: 0.05, model: null, run: { id: 'run-b' } };
    await page.locator('#verify-file-input').setInputFiles({ name: 'clean.wav', mimeType: 'audio/wav', buffer: silentWav() });
    await page.locator('#verify-btn').click();
    await expect(page.locator('#verify-result')).toContainText('No watermark detected');
    await expect(page.locator('#verify-result')).not.toContainText('unknown source model');
    await page.getByText('View technical details', { exact: true }).click();
    await expect(page.locator('.verify-result-facts')).not.toContainText('Source model');

    await page.locator('#verify-threshold').fill('0.5');
    await expect(page.locator('#verify-result')).toBeHidden();
    await page.locator('#verify-reset').click();
    await expect(page.locator('#verify-threshold')).toHaveValue('0.35');
    await expect(page.locator('#verify-run-select')).toHaveValue('');
    await page.locator('#remove-verify-file').click();
    await expect(page.locator('#verify-file-staged')).toBeHidden();
    await expect(page.locator('#verify-block-reason')).toContainText('Choose or record');
  });

  test('records verification audio, cleans microphone tracks, and handles detector errors', async ({ page }) => {
    await page.addInitScript(() => {
      (window as any).__trackStops = 0;
      const track = { stop: () => { (window as any).__trackStops += 1; } };
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: { getUserMedia: async () => ({ getTracks: () => [track] }) },
      });
      class FakeMediaRecorder {
        state = 'inactive';
        mimeType = 'audio/webm';
        ondataavailable: ((event: { data: Blob }) => void) | null = null;
        onstop: (() => void) | null = null;
        constructor(_stream: unknown) {}
        start() { this.state = 'recording'; }
        stop() {
          this.state = 'inactive';
          this.ondataavailable?.({ data: new Blob(['recorded'], { type: this.mimeType }) });
          this.onstop?.();
        }
      }
      Object.defineProperty(window, 'MediaRecorder', { configurable: true, value: FakeMediaRecorder });
    });
    await mockMobileApi(page);
    await page.route('**/api/watermark/detect', route => route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ error: 'Detector could not decode this clip.' }),
    }));
    await page.goto('/mobile/');
    await page.locator('.tab[data-tab="verify"]').click();
    await page.getByRole('button', { name: 'Record' }).click();
    await page.locator('#record-verify-file').click();
    await expect(page.locator('#record-verify-file')).toContainText('Stop recording');
    await page.locator('#record-verify-file').click();
    await expect(page.locator('#verify-file-staged')).toBeVisible();
    expect(await page.evaluate(() => (window as any).__trackStops)).toBeGreaterThan(0);
    await page.locator('#verify-btn').click();
    await expect(page.locator('#verify-error')).toContainText('Detector could not decode');
    await expect(page.locator('#verify-file-staged')).toBeVisible();
  });

  test('disables verification when no detector run is available', async ({ page }) => {
    await mockMobileApi(page);
    await page.route('**/api/watermark/runs', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ default_run_id: null, runs: [] }),
    }));
    await page.goto('/mobile/');
    await page.locator('.tab[data-tab="verify"]').click();
    await expect(page.locator('#verify-run-status')).toHaveText('Detector unavailable');
    await expect(page.locator('#verify-block-reason')).toContainText('No trained detector');
    await expect(page.locator('#verify-btn')).toBeDisabled();
  });
});
