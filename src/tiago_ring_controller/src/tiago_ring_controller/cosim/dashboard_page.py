"""The dashboard page: one HTML file, no external libraries."""

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ring co-simulation</title>
<style>
  :root { --bg:#111418; --panel:#1b2027; --fg:#e6e9ee; --muted:#8a94a3; --accent:#4c9be8; --left:#4c9be8; --right:#f28c38; --goal:#e0524f; --centroid:#4fc26b; --start:#9aa4b2; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; padding:10px 16px; background:var(--panel); border-bottom:1px solid #2a3140; }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  .badge { padding:2px 10px; border-radius:12px; background:#2a3140; color:var(--muted); font-weight:600; }
  .badge.running { background:#1f4d2b; color:#8fe0a4; } .badge.resetting { background:#4d3f1f; color:#f0c674; } .badge.error { background:#4d1f1f; color:#f08080; }
  .stat { color:var(--muted); } .stat b { color:var(--fg); font-variant-numeric: tabular-nums; }
  .controls { display:flex; gap:8px; align-items:center; margin-left:auto; }
  input[type=number] { width:90px; padding:4px 6px; background:#0e1116; color:var(--fg); border:1px solid #2a3140; border-radius:4px; }
  button { padding:5px 12px; border:0; border-radius:4px; background:var(--accent); color:#fff; font-weight:600; cursor:pointer; }
  button.stop { background:var(--goal); } button.reset { background:#5b6572; } button:disabled { opacity:.4; cursor:default; }
  select { padding:4px 6px; background:#0e1116; color:var(--fg); border:1px solid #2a3140; border-radius:4px; }
  main { display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px 16px; }
  .panel { background:var(--panel); border-radius:8px; padding:8px 10px; }
  .panel h2 { font-size:13px; margin:0 0 6px; color:var(--muted); font-weight:600; }
  canvas { width:100%; height:auto; display:block; background:#0e1116; border-radius:4px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric: tabular-nums; }
  th, td { text-align:right; padding:3px 6px; border-bottom:1px solid #2a3140; } th { color:var(--muted); font-weight:600; }
  td:first-child, th:first-child { text-align:left; }
  #message { color:var(--muted); padding:0 16px 12px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Ring co-simulation</h1>
  <span id="status" class="badge">connecting</span>
  <span class="stat">trial <b id="trial">-</b></span>
  <span class="stat">t <b id="time">0.00</b> s</span>
  <span class="stat">NEST step <b id="step">-</b></span>
  <span class="stat">L <b id="left">-</b> R <b id="right">-</b></span>
  <span class="stat">error <b id="error">-</b> rad</span>
  <div class="controls">
    <label class="stat">goal (rad) <input id="goal" type="number" step="0.05" value="0.5"></label>
    <label class="stat">max steps <input id="maxsteps" type="number" step="10" placeholder="profile"></label>
    <label class="stat">before trial <select id="resetmode"><option value="rebuild">rebuild network + home robot</option><option value="continue">continue from current state</option></select></label>
    <button id="start">Start trial</button>
    <button id="stop" class="stop" disabled>Stop</button>
    <button id="reset" class="reset" title="rebuild the NEST network and home the robot now">Reset now</button>
  </div>
</header>
<div id="message"></div>
<main>
  <section class="panel"><h2>state ring r1 (spikes per tick)</h2><canvas id="ring" width="440" height="440"></canvas></section>
  <section class="panel"><h2>r1 activity, last <span id="histlen">120</span> NEST steps</h2><canvas id="raster" width="640" height="440"></canvas></section>
  <section class="panel"><h2>gain populations and filtered drive</h2><canvas id="gain" width="640" height="300"></canvas></section>
  <section class="panel"><h2>joint angle vs goal, decoded velocity</h2><canvas id="joint" width="640" height="300"></canvas></section>
  <section class="panel" style="grid-column: 1 / -1"><h2>trials</h2>
    <table><thead><tr><th>#</th><th>goal</th><th>start</th><th>final</th><th>|error|</th><th>steps</th><th>stop</th><th>reset</th><th>NEST reset</th><th>robot reset</th><th>wall</th></tr></thead>
    <tbody id="trials"></tbody></table>
  </section>
</main>
<script>
(function () {
  const HISTORY = 120, SERIES = 600;
  const css = getComputedStyle(document.documentElement);
  const color = name => css.getPropertyValue('--' + name).trim();
  const $ = id => document.getElementById(id);
  const state = { status: 'connecting', trial: 0, N: 100, raster: [], steps: [], left: [], right: [], drive: [],
                  t: [], pos: [], vel: [], latest: null, goalIndex: null, initIndex: null, goalRad: null, centroid: null, dirty: true };

  function resetBuffers(N) {
    state.N = N || state.N; state.raster = []; state.steps = []; state.left = []; state.right = []; state.drive = [];
    state.t = []; state.pos = []; state.vel = []; state.latest = null; state.goalIndex = null; state.initIndex = null;
    state.goalRad = null; state.centroid = null; state.dirty = true;
  }
  function push(arr, v, max) { arr.push(v); if (arr.length > max) arr.shift(); }

  function onTick(m) {
    if (m.trial !== state.trial) { resetBuffers(); state.trial = m.trial; }
    state.latest = m;
    if (m.goal_index !== null) state.goalIndex = m.goal_index;
    if (m.goal_rad !== null) state.goalRad = m.goal_rad;
    if (m.init_index !== null) state.initIndex = m.init_index;
    if (m.new_sample && m.r1) {
      state.N = m.r1.length;
      push(state.raster, m.r1, HISTORY);
      push(state.steps, m.nest_step, SERIES); push(state.left, m.left, SERIES); push(state.right, m.right, SERIES);
      push(state.drive, m.drive === null ? (state.drive.length ? state.drive[state.drive.length - 1] : 0) : m.drive, SERIES);
      state.centroid = m.centroid;
    }
    if (m.phase === 'main' && m.joint_position !== null) {
      push(state.t, m.t_s, SERIES); push(state.pos, m.joint_position, SERIES);
      push(state.vel, m.decoded_velocity === null ? 0 : m.decoded_velocity, SERIES);
    }
    $('trial').textContent = m.trial; $('time').textContent = m.t_s.toFixed(2);
    $('step').textContent = m.nest_step === null ? '-' : m.nest_step;
    $('left').textContent = m.left === null ? '-' : m.left; $('right').textContent = m.right === null ? '-' : m.right;
    $('error').textContent = m.error === null ? '-' : m.error.toFixed(4);
    state.dirty = true;
  }

  function onStatus(s) {
    state.status = s.status;
    const badge = $('status'); badge.textContent = s.status; badge.className = 'badge ' + s.status;
    $('message').textContent = s.message || '';
    $('start').disabled = !(s.status === 'idle' || s.status === 'error');
    $('reset').disabled = !(s.status === 'idle' || s.status === 'error');
    $('stop').disabled = !(s.status === 'running' || s.status === 'resetting');
    if (s.reset_mode && !state.modeTouched) $('resetmode').value = s.reset_mode;
    if (s.next_goal !== null && s.next_goal !== undefined && document.activeElement !== $('goal')) $('goal').value = s.next_goal;
    const rows = (s.trials || []).map(t => {
      const tm = t.timing || {}; const r = tm.reset_wall_s || {};
      return '<tr><td>' + t.trial + '</td><td>' + t.goal.toFixed(4) + '</td><td>' + t.q_start.toFixed(4) + '</td><td>' + t.q_final.toFixed(4) +
        '</td><td>' + t.abs_error.toFixed(4) + '</td><td>' + t.n_steps + '</td><td>' + t.stop_reason + '</td><td>' + (t.reset_mode || '-') + '</td><td>' +
        (r.nest === undefined ? '-' : r.nest.toFixed(1) + ' s') + '</td><td>' + (r.robot === undefined ? '-' : r.robot.toFixed(1) + ' s') +
        '</td><td>' + t.wall_s.toFixed(1) + ' s</td></tr>';
    });
    $('trials').innerHTML = rows.join('');
  }

  // ---- drawing --------------------------------------------------------
  const viridis = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  function cmap(v) {
    const x = Math.max(0, Math.min(1, v)) * (viridis.length - 1), i = Math.floor(x), f = x - i;
    const a = viridis[i], b = viridis[Math.min(i + 1, viridis.length - 1)];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function angleOf(index) { return 2 * Math.PI * index / state.N - Math.PI / 2; }

  function drawRing() {
    const c = $('ring'), ctx = c.getContext('2d'), W = c.width, H = c.height, cx = W / 2, cy = H / 2;
    const r0 = 70, r1 = Math.min(W, H) / 2 - 20;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#2a3140'; ctx.lineWidth = 1;
    for (const r of [r0, (r0 + r1) / 2, r1]) { ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke(); }
    ctx.fillStyle = color('muted'); ctx.font = '11px system-ui'; ctx.textAlign = 'center';
    for (let k = 0; k < 4; k++) { const a = angleOf(state.N * k / 4); ctx.fillText(Math.round(state.N * k / 4), cx + (r1 + 12) * Math.cos(a), cy + (r1 + 12) * Math.sin(a) + 4); }
    const current = state.raster.length ? state.raster[state.raster.length - 1] : null;
    if (current) {
      const top = Math.max(1, Math.max.apply(null, current));
      ctx.strokeStyle = color('left'); ctx.lineWidth = Math.max(1.5, 2 * Math.PI * r0 / state.N);
      for (let i = 0; i < state.N; i++) {
        if (!current[i]) continue;
        const a = angleOf(i), len = (r1 - r0) * current[i] / top;
        ctx.beginPath(); ctx.moveTo(cx + r0 * Math.cos(a), cy + r0 * Math.sin(a)); ctx.lineTo(cx + (r0 + len) * Math.cos(a), cy + (r0 + len) * Math.sin(a)); ctx.stroke();
      }
    }
    const marker = (index, col, dash) => {
      if (index === null || index === undefined) return;
      const a = angleOf(index); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash || []);
      ctx.beginPath(); ctx.moveTo(cx + (r0 - 8) * Math.cos(a), cy + (r0 - 8) * Math.sin(a)); ctx.lineTo(cx + (r1 + 6) * Math.cos(a), cy + (r1 + 6) * Math.sin(a)); ctx.stroke(); ctx.setLineDash([]);
    };
    marker(state.initIndex, color('start'), [6, 4]); marker(state.centroid, color('centroid')); marker(state.goalIndex, color('goal'));
    ctx.textAlign = 'left'; let y = 16;
    for (const [label, col] of [['goal', color('goal')], ['start', color('start')], ['centroid', color('centroid')]]) {
      ctx.fillStyle = col; ctx.fillRect(10, y - 9, 14, 3); ctx.fillStyle = color('fg'); ctx.fillText(label, 30, y); y += 16;
    }
  }

  const rasterBuf = document.createElement('canvas');
  function drawRaster() {
    const c = $('raster'), ctx = c.getContext('2d'), W = c.width, H = c.height, L = 40, B = 24;
    ctx.clearRect(0, 0, W, H);
    if (!state.raster.length) return;
    const N = state.N, cols = state.raster.length;
    rasterBuf.width = cols; rasterBuf.height = N;
    const img = rasterBuf.getContext('2d').createImageData(cols, N);
    let top = 1; for (const col of state.raster) for (const v of col) if (v > top) top = v;
    for (let x = 0; x < cols; x++) for (let i = 0; i < N; i++) {
      const [r, g, b] = cmap(state.raster[x][i] / top), o = 4 * ((N - 1 - i) * cols + x);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
    rasterBuf.getContext('2d').putImageData(img, 0, 0);
    const pw = (W - L - 10) * cols / HISTORY, ph = H - B - 10;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(rasterBuf, L, 10, pw, ph);
    if (state.goalIndex !== null) { const y = 10 + ph * (1 - (state.goalIndex + 0.5) / N); ctx.strokeStyle = color('goal'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke(); }
    ctx.fillStyle = color('muted'); ctx.font = '11px system-ui'; ctx.textAlign = 'right';
    for (const f of [0, 0.5, 1]) ctx.fillText(Math.round(f * (N - 1)), L - 4, 10 + ph * (1 - f) + 4);
    ctx.textAlign = 'center'; const s0 = state.steps[state.steps.length - cols];
    ctx.fillText('NEST step ' + s0 + ' … ' + state.steps[state.steps.length - 1], L + (W - L - 10) / 2, H - 6);
  }

  function lineChart(canvas, xs, series, opts) {
    const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height, L = 48, R = series.some(s => s.right) ? 52 : 10, T = 10, B = 26;
    ctx.clearRect(0, 0, W, H);
    if (xs.length < 2) return;
    const x0 = xs[0], x1 = xs[xs.length - 1], sx = x => L + (W - L - R) * (x - x0) / Math.max(x1 - x0, 1e-9);
    const axis = (values, pad) => { let lo = Math.min.apply(null, values), hi = Math.max.apply(null, values); if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); } const p = Math.max(pad, 0.1 * (hi - lo)); return [lo - p, hi + p]; };
    const leftSeries = series.filter(s => !s.right), rightSeries = series.filter(s => s.right);
    const [lo, hi] = axis(leftSeries.flatMap(s => s.values.slice(-xs.length)).concat(opts.hline !== null && opts.hline !== undefined ? [opts.hline] : []), opts.pad || 0.5);
    const sy = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    ctx.strokeStyle = '#2a3140'; ctx.lineWidth = 1; ctx.fillStyle = color('muted'); ctx.font = '11px system-ui'; ctx.textAlign = 'right';
    for (const f of [0, 0.5, 1]) { const v = lo + f * (hi - lo), y = sy(v); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke(); ctx.fillText(v.toFixed(opts.digits || 2), L - 4, y + 4); }
    ctx.textAlign = 'center'; ctx.fillText(opts.xlabel + ' ' + x0.toFixed(opts.xdigits || 0) + ' … ' + x1.toFixed(opts.xdigits || 0), L + (W - L - R) / 2, H - 6);
    if (opts.hline !== null && opts.hline !== undefined) { ctx.strokeStyle = color('goal'); ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(L, sy(opts.hline)); ctx.lineTo(W - R, sy(opts.hline)); ctx.stroke(); ctx.setLineDash([]); }
    const plot = (values, col, yfun, dash) => { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash(dash || []); ctx.beginPath(); const off = values.length - xs.length; for (let i = 0; i < xs.length; i++) { const y = yfun(values[i + off]); if (i === 0) ctx.moveTo(sx(xs[i]), y); else ctx.lineTo(sx(xs[i]), y); } ctx.stroke(); ctx.setLineDash([]); };
    for (const s of leftSeries) plot(s.values, s.color, sy, s.dash);
    if (rightSeries.length) {
      const vals = rightSeries.flatMap(s => s.values.slice(-xs.length)); const vmax = Math.max(1e-3, Math.max.apply(null, vals.map(Math.abs)));
      const sy2 = v => T + (H - T - B) * (1 - (v + vmax) / (2 * vmax));
      ctx.textAlign = 'left'; ctx.fillStyle = color('muted'); for (const f of [-1, 0, 1]) ctx.fillText((f * vmax).toFixed(3), W - R + 4, sy2(f * vmax) + 4);
      for (const s of rightSeries) plot(s.values, s.color, sy2, s.dash);
    }
    let lx = L + 6; ctx.textAlign = 'left';
    for (const s of series) { ctx.fillStyle = s.color; ctx.fillRect(lx, T + 4, 14, 3); ctx.fillStyle = color('fg'); ctx.fillText(s.label, lx + 18, T + 10); lx += 18 + ctx.measureText(s.label).width + 16; }
  }

  function draw() {
    if (state.dirty) {
      state.dirty = false;
      drawRing(); drawRaster();
      lineChart($('gain'), state.steps, [
        { label: 'left', values: state.left, color: color('left') },
        { label: 'right', values: state.right, color: color('right') },
        { label: 'filtered drive', values: state.drive, color: '#e6e9ee', dash: [4, 3] }], { xlabel: 'NEST step', zero: true, digits: 0 });
      lineChart($('joint'), state.t, [
        { label: 'joint angle', values: state.pos, color: color('centroid') },
        { label: 'decoded velocity', values: state.vel, color: '#b07cf0', right: true }],
        { xlabel: 't [s]', hline: state.goalRad, pad: 0.05, digits: 3, xdigits: 1 });
    }
    requestAnimationFrame(draw);
  }

  // ---- transport ------------------------------------------------------
  function connect() {
    const source = new EventSource('/events');
    source.onmessage = ev => {
      const m = JSON.parse(ev.data);
      if (m.type === 'tick') onTick(m);
      else if (m.type === 'status') onStatus(m);
      else if (m.type === 'trial_start') { resetBuffers(m.population_size); state.trial = m.trial; }
    };
    source.onerror = () => { $('status').textContent = 'disconnected'; $('status').className = 'badge error'; };
  }
  const post = (path, body) => fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) }).then(r => r.json());
  $('start').onclick = () => { const body = { goal: parseFloat($('goal').value), reset_mode: $('resetmode').value }; const ms = parseInt($('maxsteps').value, 10); if (!isNaN(ms)) body.max_steps = ms; post('/api/start', body).then(r => { if (r.error) $('message').textContent = r.error; }); };
  $('stop').onclick = () => post('/api/stop');
  $('reset').onclick = () => post('/api/reset');
  $('resetmode').onchange = () => { state.modeTouched = true; };
  $('goal').onchange = () => post('/api/goal', { goal: parseFloat($('goal').value) });
  $('histlen').textContent = HISTORY;
  connect(); requestAnimationFrame(draw);
})();
</script>
</body>
</html>
"""

__all__ = ["PAGE_HTML"]
