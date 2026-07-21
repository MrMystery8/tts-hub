import { expect, test } from '@playwright/test';

test.describe('desktop cross-client generation sync', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('clears the running transport when a mobile-started job completes', async ({ page }) => {
    const now = Date.now() / 1000;
    const model = { id: 'pocket-tts', name: 'Pocket TTS', description: 'Pocket' };
    const job = {
      id: 'd'.repeat(32),
      status: 'generating',
      phase: 'generating',
      created_at: now,
      started_at: now,
      updated_at: now,
      model_id: model.id,
      text: 'Started from mobile',
      output_format: 'wav',
      watermark_enabled: false,
      output: null,
      request: {},
    };
    let jobs: any[] = [];

    await page.route('**/api/models', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ models: [model] }),
    }));
    await page.route('**/api/info', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ time: now, ffmpeg: { available: true } }),
    }));
    await page.route('**/api/status', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ models: { [model.id]: { loaded: true, device: 'test', total_generations: 1 } } }),
    }));
    await page.route('**/api/voices', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ voices: [] }),
    }));
    await page.route('**/api/watermark/runs', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ default_run_id: null, runs: [] }),
    }));
    await page.route('**/api/generation-jobs', route => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ jobs }),
    }));

    await page.goto('/');
    await expect(page.getByText('Output appears here. Run a generation to begin.')).toBeVisible();

    // Simulate another client creating the job. Desktop discovers it on its
    // periodic core refresh rather than through its own per-job poller.
    jobs = [job];
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeVisible();

    jobs = [{
      ...job,
      status: 'completed',
      phase: 'completed',
      updated_at: now + 2,
      completed_at: now + 2,
      worker_duration_ms: 2000,
      output: {
        path: 'mobile-output.wav',
        filename: 'mobile-output.wav',
        format: 'wav',
        duration_s: 1.5,
        sample_rate: 24000,
      },
    }];

    await expect(page.getByText('DONE', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeHidden();
    await expect(page.getByText('mobile-output.wav', { exact: true })).toBeVisible();
  });
});
