// Verify the DEPLOYED instance is cross-origin-isolated — i.e. the multi-threaded
// decode pool (ADR-0018) can actually engage in production. GitHub Pages sends no
// COOP/COEP headers, so isolation rests ENTIRELY on the coi-serviceworker
// (public/coi-serviceworker.js) injecting `credentialless` + `same-origin` and
// forcing one reload to gain control. Nothing else in CI exercises the SW-on-real-
// Pages path (the pre-deploy real-model gate runs against a vite preview whose
// headers bypass the SW). This closes that dark gate: load the actual URL, let the
// SW take control, and assert `crossOriginIsolated === true`.
//
// Chromium supports `credentialless`; Safari does not, and there degrades to
// single-threaded BY DESIGN (ADR-0018) — so this gate is Chromium-only and a
// Safari failure would not be a regression.
import { chromium } from "playwright";

const URL = process.env.HAI_URL || "https://hologram-technologies.github.io/hologram-ai/";
const browser = await chromium.launch({
  // Codespaces/CI can classify the public URL as a private-network target.
  args: ["--disable-features=BlockInsecurePrivateNetworkRequests"],
});
let failed = false;
const fail = (m) => {
  console.error(`  ✗ ${m}`);
  failed = true;
};
const ok = (m) => console.log(`  ✓ ${m}`);
try {
  console.log(`Cross-origin-isolation probe: ${URL}`);
  const page = await (await browser.newContext()).newPage();
  await page.goto(URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(4000); // let the SW register + auto-reload

  let isolated = await page.evaluate(() => globalThis.crossOriginIsolated === true);
  if (!isolated) {
    // First load can precede the SW gaining control; the SW reloads once — give
    // it an explicit second chance before failing.
    await page.reload({ waitUntil: "networkidle" });
    await page.waitForTimeout(3000);
    isolated = await page.evaluate(() => globalThis.crossOriginIsolated === true);
  }
  const sab = await page.evaluate(() => typeof SharedArrayBuffer !== "undefined");
  const controlling = await page.evaluate(
    () => !!(navigator.serviceWorker && navigator.serviceWorker.controller),
  );
  const cores = await page.evaluate(() => navigator.hardwareConcurrency);

  console.log(`  crossOriginIsolated=${isolated}  SharedArrayBuffer=${sab}  swControlling=${controlling}  cores=${cores}`);
  if (isolated) ok("deployed instance is cross-origin-isolated — the decode pool CAN engage");
  else fail("deployed instance is NOT cross-origin-isolated — the pool cannot engage, decode is single-threaded");
  if (sab) ok("SharedArrayBuffer available");
  else fail("SharedArrayBuffer unavailable — shared-memory pool impossible");
  if (controlling) ok("coi-serviceworker is controlling the page");
  else fail("no service worker controlling — isolation on Pages depends on it");
} catch (e) {
  fail(`probe error: ${String(e).slice(0, 400)}`);
} finally {
  await browser.close();
}
console.log(failed ? "\nISOLATION PROBE: FAILED" : "\nISOLATION PROBE: PASS");
process.exitCode = failed ? 1 : 0;
