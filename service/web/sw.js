/* Service worker: lets the app be installed on a phone's home screen and
 * opens the app shell when the network drops for a moment. API calls are
 * never cached, so plate data always comes from the server. */
const CACHE = "platevision-v1";
const SHELL = ["/", "/static/style.css", "/static/app.js", "/static/icon.svg",
               "/static/icon-192.png", "/manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  const isShell = e.request.method === "GET" && url.origin === location.origin &&
    (url.pathname === "/" || url.pathname.startsWith("/static/") || url.pathname === "/manifest.webmanifest");
  if (!isShell) return;  // API: straight to the network
  // Network first (so updates show immediately), cache as fallback.
  e.respondWith(fetch(e.request).then((res) => {
    const copy = res.clone();
    caches.open(CACHE).then((c) => c.put(e.request, copy));
    return res;
  }).catch(() => caches.match(e.request)));
});
