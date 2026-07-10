/*! Cross-origin isolation via a service worker (ADR-0018).
 *
 * GitHub Pages is static hosting with no control over response headers, but
 * SharedArrayBuffer / wasm threads (the multi-threaded decode pool) require the
 * page to be cross-origin-isolated. This script — loaded as the first thing in
 * <head> — registers a service worker that re-serves every response with
 * `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
 * credentialless`, then reloads once so the SW controls the page.
 *
 * `credentialless` (not `require-corp`) is deliberate: it lets cross-origin
 * subresources — the HuggingFace model files and their LFS CDN redirect — load
 * WITHOUT a `Cross-Origin-Resource-Policy` header (which HF does not send),
 * fetched without credentials. `require-corp` would break model downloads.
 *
 * Adapted from github.com/gzuidhof/coi-serviceworker (MIT).
 */
/* eslint-disable */
const coepCredentialless = true;

if (typeof window === "undefined") {
  // ---- Service worker context ----
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener("message", (ev) => {
    if (ev.data && ev.data.type === "deregister") {
      self.registration
        .unregister()
        .then(() => self.clients.matchAll())
        .then((clients) => clients.forEach((client) => client.navigate(client.url)));
    }
  });

  self.addEventListener("fetch", function (event) {
    const r = event.request;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;

    // Under credentialless, cross-origin `no-cors` subresources are fetched
    // without credentials — mirror that so caches/headers line up.
    const request =
      coepCredentialless && r.mode === "no-cors" ? new Request(r, { credentials: "omit" }) : r;

    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.status === 0) return response; // opaque — leave as-is
          const headers = new Headers(response.headers);
          headers.set("Cross-Origin-Embedder-Policy", "credentialless");
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers,
          });
        })
        .catch((e) => console.error(e)),
    );
  });
} else {
  // ---- Page context: register + reload once to gain isolation ----
  (() => {
    // Guard against reload loops: only self-reload once per session.
    const reloadedBySelf = window.sessionStorage.getItem("coiReloadedBySelf");
    window.sessionStorage.removeItem("coiReloadedBySelf");

    const n = navigator;
    if (window.crossOriginIsolated) return; // already isolated (headers present)
    if (reloadedBySelf) return; // just reloaded; the SW should be controlling now
    if (!window.isSecureContext || !n.serviceWorker) return;

    n.serviceWorker.register(window.document.currentScript.src).then(
      (registration) => {
        registration.addEventListener("updatefound", () => {
          window.sessionStorage.setItem("coiReloadedBySelf", "updatefound");
          window.location.reload();
        });
        if (registration.active && !n.serviceWorker.controller) {
          window.sessionStorage.setItem("coiReloadedBySelf", "notcontrolling");
          window.location.reload();
        }
      },
      (err) => console.error("COOP/COEP service worker failed to register:", err),
    );
  })();
}
