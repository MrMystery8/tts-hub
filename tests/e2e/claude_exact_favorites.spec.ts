import { expect, test } from '@playwright/test';

function silentWav(): Buffer {
  const samples = 800;
  const dataSize = samples * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVEfmt ', 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(8000, 24);
  buffer.writeUInt32LE(16000, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);
  return buffer;
}

test.describe('claude_exact saved phrases', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('supports star, rename, filter, instant replay, run gating, tour, dialogs, and long ids', async ({ page }) => {
    const id = 'b'.repeat(32);
    const voiceId = 'a'.repeat(32);
    const now = Date.now() / 1000;
    let jobs: any[] = [{
      id,
      status: 'completed',
      phase: 'completed',
      created_at: now,
      model_id: 'qwen3-tts-mlx',
      voice_id: voiceId,
      text: 'Please give me a moment while I prepare my response.',
      output_format: 'wav',
      watermark_enabled: true,
      worker_duration_ms: 1200,
      favorite: false,
      label: null,
      favorited_at: null,
      output: { path: 'output.wav', format: 'wav', filename: 'phrase.wav', duration_s: 1.2 },
      request: {},
    }];
    const models = [
      { id: 'index-tts2', name: 'IndexTTS2', description: 'Index' },
      { id: 'qwen3-tts-mlx', name: 'Qwen3-TTS MLX', description: 'Qwen' },
      { id: 'chatterbox-multilingual', name: 'Chatterbox Multilingual', description: 'Chatterbox' },
    ];
    const patchBodies: any[] = [];
    let audioRequests = 0;
    let generationPosts = 0;

    await page.route('**/api/models', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify({ models }) }));
    await page.route('**/api/info', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify({ time: now, ffmpeg: { available: true } }) }));
    await page.route('**/api/status', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ models: Object.fromEntries(models.map(model => [model.id, { loaded: false, device: 'test', total_generations: 1 }])) }),
    }));
    await page.route('**/api/voices', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ voices: [{ id: voiceId, name: 'Test voice', created_at: now, duration_s: 2, has_transcript: true, compatible_models: models.map(model => model.id) }] }),
    }));
    await page.route('**/api/watermark/runs', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify({ default_run_id: null, runs: [] }) }));
    await page.route('**/api/generation-jobs', route => {
      if (route.request().method() === 'POST') {
        generationPosts += 1;
        return route.fulfill({ status: 202, contentType: 'application/json', body: JSON.stringify(jobs[0]) });
      }
      return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ jobs }) });
    });
    await page.route(`**/api/generation-jobs/${id}`, async route => {
      if (route.request().method() !== 'PATCH') return route.continue();
      const body = route.request().postDataJSON();
      patchBodies.push(body);
      const current = jobs.find(job => job.id === id);
      const updated = {
        ...current,
        ...body,
        label: Object.prototype.hasOwnProperty.call(body, 'label') ? body.label : current.label,
        favorited_at: body.favorite === true ? now + patchBodies.length : body.favorite === false ? null : current.favorited_at,
      };
      jobs = jobs.map(job => job.id === id ? updated : job);
      return route.fulfill({ contentType: 'application/json', body: JSON.stringify(updated) });
    });
    await page.route('**/api/generation-jobs/*/audio', route => {
      audioRequests += 1;
      return route.fulfill({ contentType: 'audio/wav', body: silentWav() });
    });

    await page.goto('/');
    await page.locator('[data-tour="nav"] button').filter({ hasText: 'Jobs' }).click();
    await expect(page.getByRole('button', { name: 'Save as a quick phrase' })).toBeVisible();

    page.once('dialog', dialog => dialog.accept('Need a moment'));
    await page.getByRole('button', { name: 'Save as a quick phrase' }).click();
    await expect(page.getByRole('button', { name: 'Remove from quick phrases' })).toBeVisible();
    await expect(page.getByText('Need a moment', { exact: true })).toBeVisible();

    await page.getByText('Need a moment', { exact: true }).first().click();
    page.once('dialog', dialog => dialog.accept('Please wait'));
    await page.getByRole('button', { name: 'Rename', exact: true }).click();
    await expect(page.getByText('Please wait', { exact: true }).first()).toBeVisible();
    await page.getByRole('button', { name: '★ Saved', exact: true }).first().click();
    await expect(page.getByText('Please wait', { exact: true }).first()).toBeVisible();

    const detail = page.locator('[data-scroll-keep="jobs-detail"]');
    await expect(detail).toBeVisible();
    const overflow = await detail.evaluate(node => ({ client: node.clientWidth, scroll: node.scrollWidth }));
    expect(overflow.scroll - overflow.client).toBeLessThanOrEqual(1);

    await page.getByRole('button', { name: 'Delete', exact: true }).click();
    await expect(page.getByRole('dialog')).toContainText('generated audio file');
    await expect(page.getByRole('dialog')).toContainText('Quick Phrases');
    await page.getByRole('button', { name: 'Keep it' }).click();

    await page.locator('[data-tour="nav"] button').filter({ hasText: 'Generate' }).click();
    await expect(page.getByText('Quick phrases', { exact: true })).toBeVisible();
    const beforeReplay = audioRequests;
    await page.getByRole('button', { name: 'Play quick phrase: Please wait' }).click();
    await expect.poll(() => audioRequests).toBeGreaterThan(beforeReplay);
    expect(generationPosts).toBe(0);
    await expect(page.getByRole('button', { name: 'Remove playing audio from quick phrases' })).toBeVisible();
    await page.getByRole('button', { name: 'Remove playing audio from quick phrases' }).click();
    await expect(page.getByRole('button', { name: 'Save playing audio as a quick phrase' })).toBeVisible();
    page.once('dialog', dialog => dialog.accept('Please wait'));
    await page.getByRole('button', { name: 'Save playing audio as a quick phrase' }).click();
    await expect(page.getByRole('button', { name: 'Remove playing audio from quick phrases' })).toBeVisible();

    await page.locator('[data-tour-launch]').click();
    await expect(page.locator('.tts-tour-dot')).toHaveCount(9);
    await page.getByRole('button', { name: 'Exit tour' }).click();

    jobs = [{ ...jobs[0], id: 'c'.repeat(32), status: 'queued', phase: 'queued', output: null, favorite: false }, jobs[0]];
    await page.reload();
    await expect(page.getByRole('button', { name: 'Generating…' })).toHaveAttribute('aria-disabled', 'true');
    expect(patchBodies).toEqual([
      { favorite: true, label: 'Need a moment' },
      { label: 'Please wait' },
      { favorite: false },
      { favorite: true, label: 'Please wait' },
    ]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  });
});
