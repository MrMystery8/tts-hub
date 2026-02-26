import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

function resolveExistingFile(candidates: string[]): string | null {
  for (const p of candidates) {
    if (p && fs.existsSync(p)) return p;
  }
  return null;
}

function repoRoot(): string {
  // tests/e2e/... => repo root is 3 levels up
  return path.resolve(__dirname, '..', '..');
}

test.describe('qwen3-tts-mlx voice cloning', () => {
  test('Save voice + transcript persists + generate', async ({ page }) => {
    const qwenPythonCandidates = [
      '/Users/ayaanminhas/Desktop/Personal_Work/Qwen3-TTS/.venv/bin/python3',
      '/Users/ayaanminhas/Desktop/Personal_Work/Qwen3-TTS/.venv/bin/python',
    ];
    if (!resolveExistingFile(qwenPythonCandidates)) {
      test.skip(true, 'Qwen3-TTS venv missing; expected /Users/ayaanminhas/Desktop/Personal_Work/Qwen3-TTS/.venv');
    }

    const defaultAudio = '/Users/ayaanminhas/Desktop/Personal_Work/tts-hub/mini_benchmark_data/174/50561/174-50561-0012.flac';
    const audioPath =
      process.env.E2E_REF_AUDIO_PATH ||
      resolveExistingFile([
        defaultAudio,
        path.join(repoRoot(), 'mini_benchmark_data/174/50561/174-50561-0012.flac'),
        path.join(repoRoot(), 'medium_benchmark_data/1069/133709/1069-133709-0000.flac'),
      ]);

    if (!audioPath || !fs.existsSync(audioPath)) {
      test.skip(true, 'No prerecorded voice samples found; set E2E_REF_AUDIO_PATH');
    }

    const refText = (process.env.E2E_REF_TEXT || 'THE WANDERING SINGER').trim();
    const genText = 'THIS IS AN AUTOMATED QWEN VOICE CLONING TEST.';
    const qwenModelId = (process.env.E2E_QWEN_MODEL_ID || 'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit').trim();

    await page.goto('/');

    const qwenCard = page.locator('.model-card').filter({ hasText: 'Qwen3-TTS MLX' });
    if ((await qwenCard.count()) === 0) {
      test.skip(true, 'Qwen3-TTS MLX model not present in UI (backend not integrated)');
    }
    await qwenCard.first().click();

    await page.setInputFiles('#promptFile', audioPath);
    await page.fill('#promptText', refText);

    const voiceName = `E2E Voice ${Date.now()}`;
    page.once('dialog', (dialog) => dialog.accept(voiceName));
    await page.click('#saveVoiceBtn');

    const voiceSelect = page.locator('#savedVoiceSelect');
    await expect(voiceSelect).toHaveValue(/^[0-9a-f]{32}$/);

    await page.fill('#text', genText);

    // Model-specific Qwen settings
    await page.selectOption('#qwenModel', qwenModelId);
    await page.check('#qwenAutoTranscribe');

    const [resp] = await Promise.all([
      page.waitForResponse((r) => r.url().includes('/api/generate') && r.status() === 200),
      page.click('#generate'),
    ]);

    const headers = resp.headers();
    expect(headers['x-ref-transcript-status']).toBe('provided');
    expect(headers['x-model-id']).toBeTruthy();

    const output = page.locator('#output');
    await expect(output).toBeVisible();
    const src = await output.evaluate((el) => (el as HTMLAudioElement).src);
    expect(src.startsWith('blob:')).toBeTruthy();

    // Cleanup saved voice
    page.once('dialog', (dialog) => dialog.accept());
    await page.click('#deleteVoiceBtn');
    await expect(voiceSelect).toHaveValue('');
  });

  test('Auto-transcribe path (optional)', async ({ page }) => {
    if (process.env.E2E_QWEN_ENABLE_AUTOTRANSCRIBE !== '1') {
      test.skip(true, 'Set E2E_QWEN_ENABLE_AUTOTRANSCRIBE=1 to run this test');
    }

    const qwenPythonCandidates = [
      '/Users/ayaanminhas/Desktop/Personal_Work/Qwen3-TTS/.venv/bin/python3',
      '/Users/ayaanminhas/Desktop/Personal_Work/Qwen3-TTS/.venv/bin/python',
    ];
    if (!resolveExistingFile(qwenPythonCandidates)) {
      test.skip(true, 'Qwen3-TTS venv missing; expected /Users/ayaanminhas/Desktop/Personal_Work/Qwen3-TTS/.venv');
    }

    const defaultAudio = '/Users/ayaanminhas/Desktop/Personal_Work/tts-hub/mini_benchmark_data/174/50561/174-50561-0012.flac';
    const audioPath =
      process.env.E2E_REF_AUDIO_PATH ||
      resolveExistingFile([
        defaultAudio,
        path.join(repoRoot(), 'mini_benchmark_data/174/50561/174-50561-0012.flac'),
      ]);

    if (!audioPath || !fs.existsSync(audioPath)) {
      test.skip(true, 'No prerecorded voice samples found; set E2E_REF_AUDIO_PATH');
    }

    const genText = 'THIS IS AN AUTOMATED QWEN AUTO-TRANSCRIBE TEST.';
    const qwenModelId = (process.env.E2E_QWEN_MODEL_ID || 'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit').trim();

    await page.goto('/');
    const qwenCard = page.locator('.model-card').filter({ hasText: 'Qwen3-TTS MLX' });
    if ((await qwenCard.count()) === 0) {
      test.skip(true, 'Qwen3-TTS MLX model not present in UI (backend not integrated)');
    }
    await qwenCard.first().click();

    await page.setInputFiles('#promptFile', audioPath);
    await page.fill('#promptText', '');

    const voiceName = `E2E Voice Auto ${Date.now()}`;
    page.once('dialog', (dialog) => dialog.accept(voiceName));
    await page.click('#saveVoiceBtn');

    const voiceSelect = page.locator('#savedVoiceSelect');
    await expect(voiceSelect).toHaveValue(/^[0-9a-f]{32}$/);

    await page.fill('#text', genText);
    await page.selectOption('#qwenModel', qwenModelId);
    await page.check('#qwenAutoTranscribe');

    const [resp] = await Promise.all([
      page.waitForResponse((r) => r.url().includes('/api/generate') && r.status() === 200),
      page.click('#generate'),
    ]);

    const headers = resp.headers();
    expect(headers['x-ref-transcript-status']).toBe('auto_transcribed');
    expect(headers['x-model-id']).toBeTruthy();

    // Cleanup
    page.once('dialog', (dialog) => dialog.accept());
    await page.click('#deleteVoiceBtn');
    await expect(voiceSelect).toHaveValue('');
  });
});

