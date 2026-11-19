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
  const registryModels = (await registry.json()).models || [];
  $('registry-details').textContent = JSON.stringify(registryModels, null, 2);
  selectOptions('registered-model', registryModels.filter((model) => model.managed).map((model) => model.id));
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
function syncJobForm() {
  const kind = $('job-kind').value;
  const calibrated = kind === 'quantize' || kind === 'calibrate';
  $('job-context-fields').hidden = !calibrated && kind !== 'benchmark';
  $('job-context').required = calibrated || kind === 'benchmark';
  $('job-data-fields').hidden = !calibrated;
  $('job-quantize-fields').hidden = kind !== 'quantize';
  $('job-codec-fields').hidden = kind !== 'quantize' && kind !== 'package';
  $('job-package-fields').hidden = kind !== 'package';
  $('job-benchmark-fields').hidden = kind !== 'benchmark';
  $('job-dataset').required = calibrated;
  $('benchmark-prompt').required = kind === 'benchmark';
}
function selectOptions(id, values, emptyLabel = '') {
  const element = $(id), previous = element.value;
  const signature = JSON.stringify([values, emptyLabel]);
  if (element.dataset.options === signature) return;
  element.dataset.options = signature;
  const options = values.map((name) => new Option(name, name));
  if (emptyLabel) options.unshift(new Option(emptyLabel, ''));
  element.replaceChildren(...options);
  if (values.includes(previous) || (emptyLabel && !previous)) element.value = previous;
}
async function inventory() {
  const response = await fetch('/studio/api/inventory');
  if (!response.ok) return false;
  const {sources, datasets, adapters, profiles} = await response.json();
  selectOptions('job-source', sources);
  selectOptions('job-dataset', datasets);
  selectOptions('job-validation', datasets, 'Skip validation');
  selectOptions('register-source', sources);
  selectOptions('register-lora', adapters, 'No adapter');
  selectOptions('register-profile', profiles, 'Default execution settings');
  selectOptions('profile-existing', profiles);
  selectOptions('profile-source', sources);
  $('operations').hidden = false;
  return true;
}
async function change(path, method, body) {
  const response = await fetch(path, {method,
    headers: {'content-type': 'application/json', 'x-quantinfer-studio': '1'},
    body: body === undefined ? undefined : JSON.stringify(body)});
  if (!response.ok) {
    const content = await response.text();
    const message = response.headers.get('content-type')?.includes('application/json')
      ? JSON.parse(content).error : content;
    throw new Error(message || response.statusText);
  }
}
function showError(error) { $('job-status').textContent = error.message; }
$('job-kind').addEventListener('change', syncJobForm);
$('dataset-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const name = $('dataset-name').value;
    await change('/studio/api/datasets/' + encodeURIComponent(name), 'PUT', {text: $('dataset-text').value});
    await inventory();
    $('job-dataset').value = name;
    $('job-status').textContent = 'Dataset saved.';
  } catch (error) { showError(error); }
});
$('registration-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    $('job-status').textContent = 'Loading model…';
    await change('/studio/api/models/' + encodeURIComponent($('register-name').value), 'PUT', {
      source: $('register-source').value, lora: $('register-lora').value || null,
      profile: $('register-profile').value || null
    });
    await refresh();
    $('job-status').textContent = 'Model registered.';
  } catch (error) { showError(error); }
});
$('remove-model').addEventListener('click', async () => {
  const name = $('registered-model').value;
  if (!name) return;
  try {
    await change('/studio/api/models/' + encodeURIComponent(name), 'DELETE');
    await refresh();
    $('job-status').textContent = 'Registration removed.';
  } catch (error) { showError(error); }
});
const profilePath = (name) => '/studio/api/profiles/' + encodeURIComponent(name);
const numberOrNull = (id) => $(id).value === '' ? null : Number($(id).value);
$('profile-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await change(profilePath($('profile-name').value), 'PUT', {
      source: $('profile-source').value, context_length: Number($('profile-context').value),
      thread_count: Number($('profile-threads').value), memory_budget_bytes: numberOrNull('profile-budget'),
      model_variant: $('profile-variant').value || null, quality_threshold: numberOrNull('profile-quality'),
      performance_baseline_tokens_per_second: numberOrNull('profile-performance')
    });
    await inventory();
    $('job-status').textContent = 'Profile saved.';
  } catch (error) { showError(error); }
});
$('load-profile').addEventListener('click', async () => {
  const name = $('profile-existing').value;
  if (!name) return;
  try {
    const response = await fetch(profilePath(name));
    const profile = await response.json();
    if (!response.ok) throw new Error(profile.error);
    $('profile-name').value = name;
    $('profile-source').value = profile.source;
    $('profile-context').value = profile.settings.context_length;
    $('profile-threads').value = profile.settings.thread_count;
      $('profile-budget').value = profile.settings.memory_budget_bytes ?? '';
      $('profile-variant').value = profile.settings.model_variant;
      $('profile-quality').value = profile.settings.quality_threshold ?? '';
      $('profile-performance').value = profile.settings.performance_baseline_tokens_per_second ?? '';
  } catch (error) { showError(error); }
});
$('delete-profile').addEventListener('click', async () => {
  const name = $('profile-existing').value;
  if (!name) return;
  try {
    await change(profilePath(name), 'DELETE');
    await inventory();
    $('job-status').textContent = 'Stored profile removed.';
  } catch (error) { showError(error); }
});
$('job-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const kind = $('job-kind').value;
  const body = {kind, source: $('job-source').value};
  if (kind !== 'verify' && kind !== 'package') body.context = Number($('job-context').value);
  if (kind === 'benchmark') {
    body.prompt = $('benchmark-prompt').value;
    body.tokens = Number($('benchmark-tokens').value);
  } else if (kind === 'quantize' || kind === 'calibrate') {
    body.dataset = $('job-dataset').value;
  }
  if (kind === 'package') {
    body.codec = $('job-codec').value;
    body.max_shard_bytes = Number($('job-shard-mib').value) * 1024 * 1024 || null;
  }
  if (kind === 'quantize') Object.assign(body, {
    validation_dataset: $('job-validation').value || null,
    algorithm: $('job-algorithm').value, bits: Number($('job-bits').value),
    group_size: Number($('job-group').value), package: $('job-profile').value === 'package', codec: $('job-codec').value
  });
  try {
    await change('/studio/api/jobs', 'POST', body);
    $('job-status').textContent = 'Job queued.';
    await jobs();
  } catch (error) { showError(error); }
});
let shownJobs = '';
async function jobs() {
  const response = await fetch('/studio/api/jobs');
  if (!response.ok) return;
  const {jobs: entries} = await response.json();
  const signature = JSON.stringify(entries);
  if (shownJobs === signature) return;
  shownJobs = signature;
  const fragment = document.createDocumentFragment();
  for (const job of entries) {
    const row = document.createElement('article');
    row.className = 'job';
    const label = document.createElement('strong');
    label.textContent = `${job.request.algorithm || job.request.kind} · ${job.request.source} · ${job.phase}`;
    row.append(label);
    if (job.phase === 'queued' || job.phase === 'running') {
      const button = document.createElement('button');
      button.textContent = 'Cancel';
      button.addEventListener('click', () => change('/studio/api/jobs/' + encodeURIComponent(job.id), 'DELETE').then(jobs).catch(showError));
      row.append(document.createTextNode(' '), button);
    }
    if (job.detail || job.result?.artifact) {
      const details = document.createElement('pre');
      details.textContent = job.detail || JSON.stringify(job.result, null, 2);
      row.append(details);
    }
    const metrics = job.result?.metrics;
    if (metrics) {
      const times = [['Load', metrics.model_load_ms], ['Prefill', metrics.prefill_ms], ['First token', metrics.ttft_ms], ['Decode', metrics.profile_decode_ms]];
      const max = Math.max(1, ...times.map(([, value]) => value || 0));
      for (const [name, value] of times) {
        if (!Number.isFinite(value)) continue;
        const metric = document.createElement('div');
        metric.className = 'metric';
        const scale = document.createElement('meter');
        scale.max = max;
        scale.value = value;
        scale.setAttribute('aria-label', name);
        metric.append(document.createTextNode(name), scale, document.createTextNode(`${value.toFixed(2)} ms`));
        row.append(metric);
      }
      const details = document.createElement('pre');
      details.textContent = JSON.stringify(metrics, null, 2);
      row.append(details);
    }
    fragment.append(row);
  }
  $('job-history').replaceChildren(fragment);
}
syncJobForm();
inventory().then((enabled) => {
  if (!enabled) return;
  async function poll() {
    try { await Promise.all([inventory(), jobs()]); } catch (error) { showError(error); }
    setTimeout(poll, 2000);
  }
  poll();
}).catch(showError);
refresh().catch((e) => $('health').textContent = e);
