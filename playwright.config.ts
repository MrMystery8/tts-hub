import { defineConfig } from '@playwright/test';

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:7896';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 10 * 60 * 1000,
  expect: { timeout: 60 * 1000 },
  use: {
    baseURL,
    browserName: 'chromium',
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
  },
  reporter: [['list']],
  webServer: {
    command: 'bash -lc "./run.sh"',
    url: `${baseURL}/api/info`,
    reuseExistingServer: true,
    timeout: 180 * 1000,
  },
});
