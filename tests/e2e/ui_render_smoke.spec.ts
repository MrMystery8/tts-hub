import { expect, test } from '@playwright/test';

const desktopSurfaces = [
  { label: 'Generate', selector: '[data-tour="script"] textarea' },
  { label: 'Voices', selector: '[data-surface="voices"]' },
  { label: 'Jobs', selector: '[data-surface="jobs"]' },
  { label: 'Models & System', selector: '[data-surface="models"]' },
  { label: 'Watermark', selector: '[data-surface="watermark"]' },
];

test.describe('supported client render smoke', () => {
  test('all desktop surfaces render without page errors or horizontal overflow', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    const navigation = page.locator('[data-tour="nav"]');
    await expect(navigation).toBeVisible();

    for (const surface of desktopSurfaces) {
      await navigation.locator('button').filter({ hasText: surface.label }).click();
      await expect(page.locator(surface.selector)).toBeVisible();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflow, `${surface.label} should not overflow horizontally`).toBeLessThanOrEqual(1);
    }

    expect(errors).toEqual([]);
  });

  test('formal navigation and primary desktop controls render', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');

    const navigation = page.locator('[data-tour="nav"]');
    for (const label of ['Generate', 'Voices', 'Jobs', 'Models & System', 'Watermark']) {
      await expect(navigation.locator('button').filter({ hasText: label })).toBeVisible();
    }
    await expect(page.locator('[data-tour="models"]')).toBeVisible();
    await expect(page.locator('[data-tour="script"] textarea')).toBeVisible();
    await expect(page.locator('[data-tour="actions"] button').last()).toBeVisible();
    await expect(page.getByRole('img', { name: 'TTS Hub' })).toBeVisible();
  });

  test('generation controls are reachable in both supported clients', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    await expect(page.locator('[data-tour="script"] textarea')).toBeVisible();
    await expect(page.locator('[data-tour="actions"] button').last()).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/mobile/');
    await expect(page.locator('#script')).toBeVisible();
    await expect(page.locator('#run-btn')).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });

  test('recommends Qwen for daily use and remembers the desktop model choice', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');

    const modelRail = page.locator('[data-tour="models"]');
    const qwen = modelRail.getByRole('button').filter({ hasText: 'Qwen3-TTS MLX' });
    await expect(qwen).toHaveAttribute('aria-pressed', 'true');
    await expect(qwen).toContainText('Recommended');
    await expect(qwen).toContainText('Fast and reliable for daily use.');
    await expect(modelRail.getByRole('button').filter({ hasText: 'IndexTTS2' })).toContainText('Best quality and expressive control.');
    await expect(modelRail.getByRole('button').filter({ hasText: 'Chatterbox Multilingual' })).toContainText('Multilingual and long-form speech.');
    for (const description of [
      'Best quality and expressive control.',
      'Fast and reliable for daily use.',
      'Multilingual and long-form speech.',
    ]) {
      const line = modelRail.getByText(description, { exact: true });
      await expect(line).toBeVisible();
      expect(await line.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
    }
    await expect(modelRail).not.toContainText(/idle\s*·|unknown\s*·|\d+ runs/);

    await modelRail.getByRole('button').filter({ hasText: 'IndexTTS2' }).click();
    await page.reload();
    await expect(modelRail.getByRole('button').filter({ hasText: 'IndexTTS2' })).toHaveAttribute('aria-pressed', 'true');
  });
});
