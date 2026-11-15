//! Embedded local dashboard and registry metadata route.

use std::sync::Arc;

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde_json::json;

use crate::state::AppState;

/// Serve a dependency-free local dashboard for model registry and inference
/// smoke tests. The page deliberately uses the public HTTP endpoints so it is
/// also useful as a small integration example for clients.
pub(crate) async fn studio() -> Html<&'static str> {
    Html(STUDIO_HTML)
}

pub(crate) async fn studio_registry(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({"models": state.models.descriptors()}))
}

pub(crate) const STUDIO_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuantInfer Studio</title>
  <style>
    :root { color-scheme: light dark; font: 16px/1.45 system-ui, sans-serif; }
    body { margin: 0 auto; max-width: 1050px; padding: 2rem; }
    header { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
    h1 { margin: 0; }
    .status { color: #087f5b; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
    section { border: 1px solid #8886; border-radius: .7rem; padding: 1rem; }
    label { display: block; margin: .6rem 0 .2rem; }
    textarea, input, select, button { box-sizing: border-box; font: inherit; width: 100%; padding: .5rem; }
    textarea { min-height: 8rem; resize: vertical; }
    button { cursor: pointer; margin-top: .8rem; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #8882; padding: .7rem; border-radius: .4rem; }
    code { font-size: .9em; }
    .sample { display: grid; grid-template-columns: 7rem 1fr 6rem; gap: .5rem; align-items: center; margin: .4rem 0; }
    .bar { height: .7rem; border-radius: .35rem; background: #4c6ef5; min-width: 2px; }
  </style>
</head>
<body>
  <header>
    <h1>QuantInfer Studio</h1>
    <span id="health" class="status">checking service…</span>
  </header>
  <p>Inspect the live model registry, run a generation smoke test, and view
     runtime counters without installing a separate frontend.</p>
  <div class="grid">
    <section>
      <h2>Model registry</h2>
      <p id="models">loading…</p>
      <pre id="registry-details">loading package and profile metadata…</pre>
      <button id="refresh" type="button">Refresh</button>
    </section>
    <section>
      <h2>Generation</h2>
      <label for="model">Model</label>
      <select id="model"></select>
      <label for="prompt">Prompt</label>
      <textarea id="prompt" placeholder="Write a prompt or use token IDs below"></textarea>
      <label for="max-tokens">Max new tokens</label>
      <input id="max-tokens" type="number" min="1" max="4096" value="32">
      <button id="generate" type="button">Generate</button>
      <pre id="result">No request yet.</pre>
    </section>
    <section>
      <h2>Benchmark counters</h2>
      <pre id="metrics">loading…</pre>
      <h3>Live generation samples</h3>
      <div id="benchmarks">Run a generation request to collect a sample.</div>
      <p>For reproducible latency and memory measurements, use the pinned
        <code>quantinfer-xtask bench</code> command.</p>
    </section>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    const samples = [];
    function renderBenchmarks() {
      if (!samples.length) {
        $('benchmarks').textContent = 'Run a generation request to collect a sample.';
        return;
      }
      const peak = Math.max(...samples.map((sample) => sample.elapsed));
      $('benchmarks').innerHTML = samples.map((sample) => {
        const width = Math.max(2, Math.round(sample.elapsed / peak * 100));
        return `<div class="sample"><span>${sample.elapsed.toFixed(1)} ms</span>` +
          `<span class="bar" style="width:${width}%" title="${sample.rate.toFixed(2)} tokens/s"></span>` +
          `<span>${sample.rate.toFixed(2)} tok/s</span></div>`;
      }).join('');
    }
    async function refresh() {
      const [health, models, metrics, registry] = await Promise.all([
        fetch('/healthz'), fetch('/v1/models'), fetch('/metrics'), fetch('/studio/api/registry')
      ]);
      $('health').textContent = health.ok ? 'service healthy' : 'service unavailable';
      const payload = await models.json();
      const entries = payload.data || [];
      $('models').textContent = entries.length ? entries.map((m) => m.id).join(', ') : 'No models loaded';
      $('model').replaceChildren(...entries.map((m) => new Option(m.id, m.id)));
      $('metrics').textContent = await metrics.text();
      $('registry-details').textContent = JSON.stringify((await registry.json()).models || [], null, 2);
    }
    $('refresh').addEventListener('click', () => refresh().catch((e) => $('health').textContent = e));
    $('generate').addEventListener('click', async () => {
      const model = $('model').value;
      if (!model) { $('result').textContent = 'Load a model before generating.'; return; }
      $('result').textContent = 'running…';
      const started = performance.now();
      const response = await fetch('/v1/generate', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({model, prompt: $('prompt').value,
          max_new_tokens: Number($('max-tokens').value) || 32})
      });
      const body = await response.json();
      const elapsed = performance.now() - started;
      if (response.ok) {
        const tokenCount = Array.isArray(body.tokens) ? body.tokens.length : 0;
        samples.unshift({elapsed, rate: tokenCount / Math.max(elapsed / 1000, 0.001)});
        samples.splice(5);
        renderBenchmarks();
      }
      $('result').textContent = response.ok ? (body.text || JSON.stringify(body, null, 2))
        : (body.error || JSON.stringify(body, null, 2));
    });
    refresh().catch((e) => $('health').textContent = e);
  </script>
</body>
</html>"##;
