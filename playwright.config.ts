import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 10 * 60 * 1000,
  expect: { timeout: 60 * 1000 },
  use: {
    baseURL: 'http://localhost:7891',
    browserName: 'chromium',
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
  },
  reporter: [['list']],
  webServer: {
    command: 'bash -lc "./run.sh"',
    url: 'http://localhost:7891/api/info',
    reuseExistingServer: true,
    timeout: 180 * 1000,
  },
});

