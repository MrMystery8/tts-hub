import { expect, test } from '@playwright/test';

const surfaces = [
  'generate',
  'models',
  'voices',
  'history',
  'watermark-lab',
  'system-status',
  'advanced-settings',
];

test.describe('React UI render smoke', () => {
  test('all surfaces render without console errors or horizontal overflow', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(message.text());
    });

    for (const surface of surfaces) {
      await page.goto(`/#${surface}`);
      await expect(page.locator('.workspace-body')).toBeVisible();
      await expect(page.locator('.workspace-body > .surface')).toBeVisible();

      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflow, `${surface} should not overflow horizontally`).toBeLessThanOrEqual(1);
    }

    expect(errors).toEqual([]);
  });

  test('academic navigation labels and product-polish controls render', async ({ page }) => {
    await page.goto('/#generate');
    await expect(page.locator('[data-surface-link="generate"]').first()).toContainText('Generate Speech');
    await expect(page.locator('[data-surface-link="models"]').first()).toContainText('Model Lab');
    await expect(page.locator('[data-surface-link="voices"]').first()).toContainText('Voice Library');
    await expect(page.locator('[data-surface-link="history"]').first()).toContainText('Runs');
    await expect(page.locator('[data-surface-link="watermark-lab"]').first()).toContainText('Provenance Lab');
    await expect(page.locator('.generation-state')).toBeVisible();
    await expect(page.locator('#generate')).toBeVisible();
    await expect(page.locator('#text')).toBeVisible();

    await page.goto('/#models');
    await expect(page.locator('.comparison-row')).toHaveCount(7);

    await page.goto('/#voices');
    await expect(page.getByLabel('Search saved voices')).toBeVisible();
  });

  test('generate action is reachable early on desktop and mobile', async ({ page }) => {
    const assertGeneratePlacement = async (width: number, height: number, maxTop: number) => {
      await page.setViewportSize({ width, height });
      await page.goto('/#generate');
      await expect(page.locator('#generate')).toBeVisible();
      await expect(page.locator('#text')).toBeVisible();

      const metrics = await page.evaluate(() => {
        const generate = document.getElementById('generate')?.getBoundingClientRect();
        const text = document.getElementById('text')?.getBoundingClientRect();
        return {
          generateTop: Math.round(generate?.top ?? 99999),
          textTop: Math.round(text?.top ?? 99999),
          overflow: document.documentElement.scrollWidth - window.innerWidth,
        };
      });

      expect(metrics.overflow).toBeLessThanOrEqual(1);
      expect(metrics.textTop).toBeLessThan(maxTop);
      expect(metrics.generateTop).toBeLessThan(maxTop);
    };

    await assertGeneratePlacement(1440, 900, 760);
    await assertGeneratePlacement(390, 844, 820);
  });
});
