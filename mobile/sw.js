/* Minimal service worker: cache the app shell, never cache API or audio. */
const CACHE = "tts-hub-mobile-v11";
const SHELL = [
  "/mobile/",
  "/mobile/app.css",
  "/mobile/app.js",
  "/mobile/manifest.webmanifest",
  "/mobile/icon-192.png",
  "/mobile/icon-512.png",
  "/mobile/icon-maskable-512.png",
  "/mobile/favicon-32.png",
  "/mobile/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.pathname.startsWith("/api/")) return;
  if (!url.pathname.startsWith("/mobile")) return;
  // Network-first for the shell so UI updates land immediately; cache is the offline fallback.
  event.respondWith(
    fetch(event.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(event.request, copy));
        return res;
      })
      .catch(() => caches.match(event.request, { ignoreSearch: true }))
  );
});
