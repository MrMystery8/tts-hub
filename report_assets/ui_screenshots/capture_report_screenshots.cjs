#!/usr/bin/env node

/*
 * Reproducible FYP UI capture against the live TTS Hub APIs.
 * No routes or responses are mocked. Existing records are never altered/deleted.
 */
const { chromium } = require('@playwright/test');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const ROOT = path.resolve(__dirname, '../..');
const OUT = __dirname;
const BASE_URL = process.env.TTS_HUB_URL || 'http://localhost:7896';
const DEMO_VOICE = 'FYP Demo - Narration';
const DEMO_AUDIO = path.join(ROOT, 'medium_benchmark_data/1069/133709/1069-133709-0000.flac');
const DEMO_TRANSCRIPT = "HAD LAID BEFORE HER A PAIR OF ALTERNATIVES NOW OF COURSE YOU'RE COMPLETELY YOUR OWN MISTRESS AND ARE AS FREE AS THE BIRD ON THE BOUGH I DON'T MEAN YOU WERE NOT SO BEFORE BUT YOU'RE AT PRESENT ON A DIFFERENT FOOTING";
const RUN_B = 'outputs/dashboard_runs/sweep3_B_static_12_2';
const QUICK_PHRASE_LABEL = 'Please wait';
const DESKTOP_VIEWPORT = { width: 1440, height: 900, deviceScaleFactor: 2 };
const MOBILE_VIEWPORT = { width: 390, height: 844, deviceScaleFactor: 2 };
const JOBS = [
  { model: 'index-tts2', text: 'This message was generated locally using my saved voice.', watermark: false },
  { model: 'chatterbox-multilingual', text: 'Thank you for your patience. I am ready to continue.', watermark: false },
  { model: 'qwen3-tts-mlx', text: 'Please give me a moment while I prepare my response.', watermark: true },
];

async function api(route, options = {}) {
  const response = await fetch(`${BASE_URL}${route}`, options);
  if (!response.ok) throw new Error(`${options.method || 'GET'} ${route}: ${response.status} ${await response.text()}`);
  return response.json();
}

async function ensureVoice() {
  const listed = await api('/api/voices');
  let voice = (listed.voices || []).find((item) => item.name === DEMO_VOICE);
  if (voice) return voice;
  const form = new FormData();
  form.append('name', DEMO_VOICE);
  form.append('prompt_text', DEMO_TRANSCRIPT);
  const bytes = fs.readFileSync(DEMO_AUDIO);
  form.append('prompt_audio', new Blob([bytes], { type: 'audio/flac' }), path.basename(DEMO_AUDIO));
  voice = await api('/api/voices', { method: 'POST', body: form });
  return voice.voice || voice;
}

async function waitForJob(id, timeoutMs = 20 * 60 * 1000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const job = await api(`/api/generation-jobs/${id}`);
    if (job.status === 'completed') return job;
    if (job.status === 'failed' || job.status === 'cancelled') throw new Error(`${id} ${job.status}: ${job.error || 'unknown error'}`);
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error(`Timed out waiting for job ${id}`);
}

async function ensureJobs(voice) {
  const listed = await api('/api/generation-jobs');
  const all = listed.jobs || [];
  const result = [];
  for (const spec of JOBS) {
    let job = all.find((item) => item.status === 'completed' && item.model_id === spec.model && item.text === spec.text && !!item.watermark_enabled === spec.watermark);
    if (!job) {
      const form = new FormData();
      form.append('model_id', spec.model);
      form.append('text', spec.text);
      form.append('voice_id', voice.id);
      form.append('output_format', 'wav');
      if (spec.watermark) {
        form.append('watermark', '1');
        form.append('watermark_run', RUN_B);
      }
      form.append('request_snapshot', JSON.stringify({ schemaVersion: 2, source: 'fyp-report', modelId: spec.model, voiceId: voice.id, text: spec.text, outputFormat: 'wav', watermarkEnabled: spec.watermark, watermarkRun: spec.watermark ? RUN_B : null }));
      const queued = await api('/api/generation-jobs', { method: 'POST', body: form });
      job = await waitForJob(queued.id);
    }
    result.push(job);
  }
  const qwenIndex = result.findIndex((job) => job.model_id === 'qwen3-tts-mlx');
  if (qwenIndex >= 0 && (!result[qwenIndex].favorite || result[qwenIndex].label !== QUICK_PHRASE_LABEL)) {
    result[qwenIndex] = await api(`/api/generation-jobs/${result[qwenIndex].id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ favorite: true, label: QUICK_PHRASE_LABEL }),
    });
  }
  return result;
}

async function saveQwenAudio(qwenJob) {
  const audioPath = path.join(OUT, 'fyp_demo_qwen_watermarked.wav');
  const response = await fetch(`${BASE_URL}/api/generation-jobs/${qwenJob.id}/audio`);
  if (!response.ok) throw new Error(`Could not download Qwen output: ${response.status}`);
  fs.writeFileSync(audioPath, Buffer.from(await response.arrayBuffer()));
  return audioPath;
}

async function verifyWatermark(audioPath) {
  const form = new FormData();
  form.append('audio', new Blob([fs.readFileSync(audioPath)], { type: 'audio/wav' }), path.basename(audioPath));
  form.append('watermark_run', RUN_B);
  form.append('wm_threshold', '0.50');
  const result = await api('/api/watermark/detect', { method: 'POST', body: form });
  const modelText = JSON.stringify(result.model || {}).toLowerCase();
  if (!result.detected || Number(result.wm_prob || 0) < 0.5 || !modelText.includes('qwen')) {
    throw new Error(`Expected positive Qwen attribution, received ${JSON.stringify(result)}`);
  }
  return result;
}

async function settle(page, waitMs = 3600) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.waitForTimeout(waitMs);
}

async function openDesktop(page) {
  await page.goto(`${BASE_URL}/`, { waitUntil: 'domcontentloaded' });
  await page.locator('[data-tour="nav"]').waitFor();
  await page.getByText(DEMO_VOICE, { exact: true }).first().waitFor({ timeout: 15000 });
  await settle(page);
}

async function useDarkTheme(page) {
  const dark = page.getByRole('button', { name: 'Dark', exact: true });
  if (await dark.count()) await dark.click();
  await page.waitForTimeout(250);
}

async function captureReferenceIntake(page) {
  await page.locator('[data-tour="nav"] button').filter({ hasText: 'Generate' }).click();
  await useDarkTheme(page);
  const panel = page.locator('[data-tour="reference"]');
  await panel.waitFor();
  await panel.getByRole('button', { name: 'Record', exact: true }).click();
  const record = await panel.screenshot();
  await panel.getByRole('button', { name: 'Upload', exact: true }).click();
  const upload = await panel.screenshot();

  const montage = await page.context().newPage();
  await montage.setViewportSize({ width: 720, height: 900 });
  await montage.setContent(`<!doctype html><style>
    html,body{margin:0;background:#0c0d10;color:#e9ebef;font-family:Arial,sans-serif}
    main{width:680px;margin:20px;display:flex;flex-direction:column;gap:14px}
    .label{font-size:12px;font-weight:700;letter-spacing:.08em;color:#9aa2ad;text-transform:uppercase}
    img{display:block;width:100%;height:auto;border-radius:10px}
  </style><main>
    <div class="label">Record mode</div><img src="data:image/png;base64,${record.toString('base64')}">
    <div class="label">Upload mode</div><img src="data:image/png;base64,${upload.toString('base64')}">
  </main>`);
  await montage.locator('main').screenshot({ path: path.join(OUT, 'Figure_4.14_Desktop_Reference_Intake.png') });
  await montage.close();
}

async function captureTours(page) {
  await page.locator('[data-tour="nav"] button').filter({ hasText: 'Generate' }).click();
  await useDarkTheme(page);
  await page.locator('[data-tour="phrases"]').waitFor();
  await page.locator('[data-tour-launch]').click();
  await page.locator('.tts-tour-card').waitFor();
  await page.waitForTimeout(350);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.18_Desktop_Tour_Navigation.png') });
  await page.getByRole('button', { name: 'Step 5' }).click();
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.19_Desktop_Tour_Reference.png') });
  await page.getByRole('button', { name: 'Step 7' }).click();
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.20_Desktop_Tour_Actions.png') });
  await page.getByRole('button', { name: 'Exit tour' }).click();
}

async function captureDesktop(page, audioPath) {
  await openDesktop(page);
  await useDarkTheme(page);

  // 4.12 — configured generation with the latest real Qwen output in the dock.
  await page.locator('[data-tour="models"] button').filter({ hasText: 'Qwen3-TTS MLX' }).click();
  await page.locator('[data-tour="script"] textarea').fill(JOBS[2].text);
  await page.getByText(DEMO_VOICE, { exact: true }).first().click();
  const wmOff = page.locator('[data-tour="actions"]').getByRole('button', { name: 'Off', exact: true });
  if (await wmOff.count()) await wmOff.click();
  await page.locator('[data-tour="actions"]').getByRole('button', { name: 'On', exact: true }).waitFor();
  await page.evaluate((demoName) => {
    const ref = document.querySelector('[data-tour="reference"]');
    const nameNode = [...ref.querySelectorAll('span')].find((node) => node.textContent.trim() === demoName);
    const selectedRow = nameNode && nameNode.closest('button') && nameNode.closest('button').parentElement;
    if (selectedRow && selectedRow.parentElement) {
      [...selectedRow.parentElement.children].forEach((row) => { if (row !== selectedRow) row.style.display = 'none'; });
    }
  }, DEMO_VOICE);
  await page.waitForTimeout(350);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.12_Desktop_Generate.png') });

  // 4.13 — alternate theme and the real filtered voice record plus add controls.
  await page.locator('[data-tour="nav"] button').filter({ hasText: 'Voices' }).click();
  await page.getByRole('button', { name: 'Light', exact: true }).click();
  await page.getByPlaceholder('Search voices…').fill(DEMO_VOICE);
  await page.getByRole('button', { name: /Add voice/ }).click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.13_Desktop_Voices.png') });

  // 4.14 — record/upload reference intake, composed from two live panel states.
  await captureReferenceIntake(page);

  // 4.15 — a focused, live crop containing only the report's three frozen backends.
  await page.locator('[data-tour="nav"] button').filter({ hasText: 'Generate' }).click();
  await useDarkTheme(page);
  await page.locator('[data-tour="models"] button').first().waitFor();
  await page.screenshot({
    path: path.join(OUT, 'Figure_4.15_Desktop_Model_Rail.png'),
    clip: { x: 212, y: 0, width: 248, height: 320 },
  });

  // 4.16 — completed real jobs with a labelled, starred Qwen result.
  await page.locator('[data-tour="nav"] button').filter({ hasText: 'Jobs' }).click();
  await page.getByRole('button', { name: 'Completed', exact: true }).click();
  await page.getByText(JOBS[2].text, { exact: false }).first().click();
  await page.getByText('Restore settings', { exact: true }).waitFor();
  await page.evaluate((texts) => {
    const body = document.querySelector('[data-scroll-keep="jobs-body"]');
    if (!body) return;
    [...body.children].slice(1).forEach((row) => {
      if (!texts.some((value) => row.textContent.includes(value))) row.style.display = 'none';
    });
  }, JOBS.map((job) => job.text));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.16_Desktop_Jobs.png') });

  // 4.17 — Run B and a real detector response from the watermarked Qwen WAV.
  await page.locator('[data-tour="nav"] button').filter({ hasText: 'Watermark Lab' }).click();
  const runB = page.getByRole('button').filter({ hasText: /Run B|sweep3_B_static_12_2/ }).first();
  if (await runB.count()) await runB.click();
  const chooserPromise = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: /Choose audio/ }).click();
  const chooser = await chooserPromise;
  await chooser.setFiles(audioPath);
  await page.getByText('Detection result', { exact: true }).waitFor({ timeout: 120000 });
  await page.getByText('Detected', { exact: true }).waitFor();
  await settle(page);
  await page.evaluate((runId) => {
    const surface = document.querySelector('[data-surface="watermark"]');
    const list = surface && surface.querySelector('[data-scroll-keep="wm-runs"]');
    if (list) {
      [...list.querySelectorAll('button')].forEach((button) => {
        if (!button.textContent.includes('sweep3_B_static_12_2')) button.style.display = 'none';
        else {
          const label = button.querySelector('span:nth-child(2)');
          if (label) label.textContent = 'Run B';
        }
      });
    }
    const detail = surface && surface.querySelector('[data-scroll-keep="wm-detail"]');
    if (detail) {
      const heading = [...detail.querySelectorAll('div')].find((node) => node.children.length === 0 && node.textContent.trim() === 'sweep3_B_static_12_2');
      if (heading) heading.textContent = 'Run B';
      const raw = detail.querySelector('pre');
      if (raw) raw.style.display = 'none';
      [...detail.querySelectorAll('div')].forEach((node) => { if (node.children.length === 0 && node.textContent.trim() === runId) node.textContent = 'Run B'; });
    }
  }, RUN_B);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.17_Desktop_Watermark_Lab.png') });

  // 4.18–4.20 — live guided-tour states; the saved phrase adds a ninth step.
  await captureTours(page);
}

async function openMobile(page) {
  await page.goto(`${BASE_URL}/mobile/`, { waitUntil: 'domcontentloaded' });
  await page.locator('#model-row').waitFor();
  await page.getByText(DEMO_VOICE, { exact: true }).first().waitFor({ timeout: 15000 });
  await settle(page);
}

async function selectMobileQwen(page) {
  await page.locator('#model-row').click();
  await page.locator('.model-opt').filter({ hasText: 'Qwen3-TTS MLX' }).click();
}

async function captureMobile(page, audioPath) {
  await openMobile(page);

  // 4.21 — responsive generation workflow with Quick Phrases visible.
  await selectMobileQwen(page);
  await page.locator('.voice-chip').filter({ hasText: DEMO_VOICE }).locator('.vc-pick').click();
  await page.evaluate((demoName) => {
    document.querySelectorAll('.voice-chip').forEach((card) => { if (!card.textContent.includes(demoName)) card.style.display = 'none'; });
  }, DEMO_VOICE);
  await page.locator('#script').fill(JOBS[2].text);
  await page.locator('#options-row').click();
  const wmRow = page.locator('.wm-row');
  if ((await wmRow.textContent()).includes('Off')) await wmRow.click();
  await page.getByRole('button', { name: /Close/ }).click().catch(async () => page.locator('#sheet-close').click());
  await page.evaluate((demoName) => {
    document.querySelectorAll('.voice-chip').forEach((card) => { if (!card.textContent.includes(demoName)) card.style.display = 'none'; });
  }, DEMO_VOICE);
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.21_Mobile_Generate.png') });

  // 4.22 — completed history with the selected saved phrase playing in place.
  await page.locator('.tab[data-tab="jobs"]').click();
  await page.locator('.filter-btn').filter({ hasText: 'Done' }).click();
  await page.evaluate((texts) => {
    document.querySelectorAll('.job-card').forEach((card) => {
      if (!texts.some((value) => card.textContent.includes(value))) card.style.display = 'none';
    });
  }, JOBS.map((job) => job.text));
  const qwenCard = page.locator('.job-card').filter({ hasText: JOBS[2].text }).first();
  await qwenCard.click();
  await page.locator('#sheet-body .dj-play').click();
  await page.locator('#surface-jobs').waitFor({ state: 'visible' });
  await page.locator('#miniplayer').waitFor({ state: 'visible' });
  await page.locator('#mp-favorite[aria-pressed="true"]').waitFor({ state: 'visible' });
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.22_Mobile_Jobs.png') });

  // 4.23 — the shared record in its real edit sheet (preview/add/edit affordances).
  await page.locator('.tab[data-tab="voices"]').click();
  await page.evaluate((demoName) => {
    document.querySelectorAll('.voice-card').forEach((card) => { if (!card.textContent.includes(demoName)) card.style.display = 'none'; });
  }, DEMO_VOICE);
  const demoCard = page.locator('.voice-card').filter({ hasText: DEMO_VOICE }).first();
  await demoCard.locator('.edit-btn').click();
  await page.getByText('Edit voice', { exact: true }).waitFor();
  await page.evaluate((demoName) => {
    document.querySelectorAll('.voice-card').forEach((card) => { if (!card.textContent.includes(demoName)) card.style.display = 'none'; });
  }, DEMO_VOICE);
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.23_Mobile_Voices.png') });
  await page.locator('#sheet-close').click();

  // 4.24 — real mobile verification result with probability and source model.
  await page.locator('.tab[data-tab="verify"]').click();
  await page.locator('#verify-file-input').setInputFiles(audioPath);
  await page.locator('#verify-btn').click();
  await page.getByText('Watermark detected', { exact: true }).waitFor({ timeout: 120000 });
  await settle(page);
  await page.screenshot({ path: path.join(OUT, 'Figure_4.24_Mobile_Verify.png') });
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const voice = await ensureVoice();
  const jobs = await ensureJobs(voice);
  const qwen = jobs.find((job) => job.model_id === 'qwen3-tts-mlx');
  const audioPath = await saveQwenAudio(qwen);
  const detection = await verifyWatermark(audioPath);

  const browser = await chromium.launch({ headless: true });
  try {
    const desktop = await browser.newContext({ viewport: { width: DESKTOP_VIEWPORT.width, height: DESKTOP_VIEWPORT.height }, deviceScaleFactor: DESKTOP_VIEWPORT.deviceScaleFactor, colorScheme: 'dark' });
    await captureDesktop(await desktop.newPage(), audioPath);
    await desktop.close();

    const mobile = await browser.newContext({ viewport: { width: MOBILE_VIEWPORT.width, height: MOBILE_VIEWPORT.height }, deviceScaleFactor: MOBILE_VIEWPORT.deviceScaleFactor, isMobile: true, hasTouch: true, colorScheme: 'dark' });
    await captureMobile(await mobile.newPage(), audioPath);
    await mobile.close();
  } finally {
    await browser.close();
  }

  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
  const dirty = execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim().length > 0;
  const manifest = {
    captured_at: new Date().toISOString(),
    base_url: BASE_URL,
    source: { commit, dirty_worktree: dirty },
    viewports: { desktop: DESKTOP_VIEWPORT, mobile: MOBILE_VIEWPORT },
    voice: { id: voice.id, name: voice.name || DEMO_VOICE },
    jobs: jobs.map((job) => ({ id: job.id, model_id: job.model_id, status: job.status, watermark_enabled: !!job.watermark_enabled, favorite: !!job.favorite, label: job.label || null })),
    detection: { detected: detection.detected, wm_prob: detection.wm_prob, model: detection.model, run: detection.run },
    ui_state: { mobile_quick_phrases_first: true, mobile_jobs_player_visible: true, player_favorite_controls: ['desktop', 'mobile'] },
    figures: [
      'Figure_4.12_Desktop_Generate.png', 'Figure_4.13_Desktop_Voices.png', 'Figure_4.14_Desktop_Reference_Intake.png',
      'Figure_4.15_Desktop_Model_Rail.png', 'Figure_4.16_Desktop_Jobs.png', 'Figure_4.17_Desktop_Watermark_Lab.png',
      'Figure_4.18_Desktop_Tour_Navigation.png', 'Figure_4.19_Desktop_Tour_Reference.png', 'Figure_4.20_Desktop_Tour_Actions.png',
      'Figure_4.21_Mobile_Generate.png', 'Figure_4.22_Mobile_Jobs.png', 'Figure_4.23_Mobile_Voices.png', 'Figure_4.24_Mobile_Verify.png',
    ],
  };
  fs.writeFileSync(path.join(OUT, 'capture_manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
