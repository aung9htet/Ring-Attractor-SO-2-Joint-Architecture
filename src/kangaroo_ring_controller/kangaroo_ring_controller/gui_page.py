"""The ring GUI page: plain HTML + canvas, no libraries (served by gui.make_server)."""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Kangaroo ring GUI</title>
<style>
body{font:14px sans-serif;margin:16px;color:#111;background:#fff}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
#goal{width:min(520px,90vw)}
canvas{border:1px solid #ccc;max-width:100%}
#status{font-family:monospace;white-space:pre-wrap;margin-bottom:8px}
</style></head><body>
<h3>Kangaroo ring GUI</h3>
<div class="row">
 <label>arm <select id="side"><option>left</option><option>right</option></select></label>
 <label>joint <select id="joint"><option>0</option><option>1</option><option>2</option><option>3</option></select></label>
 <label>seed <input id="seed" type="number" value="12345" style="width:7em"></label>
 <button id="reset" title="return the arm to this joint's start pose">Reset</button> <button id="stop">Stop</button>
</div>
<div class="row">
 <span id="lo"></span><input id="goal" type="range" step="0.01"><span id="hi"></span>
 <span>goal <b id="goalv">-</b> rad &nbsp; joint at <b id="pos">-</b> rad</span>
</div>
<div id="status">connecting...</div>
<div class="row"><canvas id="ring" width="300" height="300"></canvas><canvas id="plot" width="600" height="300"></canvas></div>
<script>
const $ = id => document.getElementById(id);
let ticks = [], st = {}, joints = [], pending = false;
const f = x => typeof x === 'number' ? x.toFixed(3) : x;
const sel = () => ({side: $('side').value, joint: +$('joint').value, seed: $('seed').value});
async function post(path, body) {
  const r = await fetch(path, {method: 'POST', body: JSON.stringify(body)});
  if (!r.ok) $('status').textContent = 'error ' + r.status + ': ' + await r.json();
}
async function loadJoints(moveSlider) {
  joints = await (await fetch('/api/joints')).json();
  const j = joints.find(j => j.side === $('side').value && j.joint === +$('joint').value);
  $('goal').min = j.limits[0]; $('goal').max = j.limits[1];
  $('lo').textContent = j.limits[0].toFixed(2); $('hi').textContent = j.limits[1].toFixed(2);
  if (j.position !== null) $('pos').textContent = f(j.position);
  if (moveSlider && j.position !== null) { $('goal').value = j.position; $('goalv').textContent = f(j.position); }
}
async function select() { await post('/api/select', sel()); await loadJoints(true); }
$('side').onchange = $('joint').onchange = $('seed').onchange = select;
$('goal').oninput = () => $('goalv').textContent = f(+$('goal').value);
$('goal').onchange = () => post('/api/start', {...sel(), goal: $('goal').value});
$('reset').onclick = () => post('/api/reset', sel());
$('stop').onclick = () => post('/api/stop', {});
function showStatus() {
  const s = st, what = s.side ? '  ' + s.side + ' j' + s.joint + (s.joint_name ? ' (' + s.joint_name + ')' : '') : '';
  $('status').textContent = s.status + (s.message ? ': ' + s.message : '') + what +
    (s.goal !== undefined ? '  goal ' + f(s.goal) : '') +
    (s.status === 'building' ? '\\nbuilding the ring (22-30 s)...' : '') +
    (s.status === 'running' && s.building && !ticks.length ? '\\nbuilding the ring (22-30 s), then reaching...' : '') +
    ('calibration' in s ? '\\ndecoder: ' + (s.calibration || 'uncalibrated') : '') +
    ('q_final' in s ? '\\n' + f(s.q_start) + ' -> ' + f(s.q_final) + ' rad (' + s.stop_reason + ', ' + s.ticks + ' ticks)' : '');
}
function drawRing() {
  const c = $('ring').getContext('2d'), W = 300, R = 120, t = ticks[ticks.length - 1];
  c.clearRect(0, 0, W, W); c.strokeStyle = '#999'; c.beginPath(); c.arc(W / 2, W / 2, R, 0, 2 * Math.PI); c.stroke();
  if (!t) return;
  const dot = (i, color, r) => { const a = 2 * Math.PI * i / t.neurons - Math.PI / 2;
    c.fillStyle = color; c.beginPath(); c.arc(W / 2 + R * Math.cos(a), W / 2 + R * Math.sin(a), r, 0, 2 * Math.PI); c.fill(); };
  dot(ticks[0].state_index, '#aaa', 6); dot(t.goal_index, '#2a2', 8); dot(t.state_index, '#c22', 6);
  c.fillStyle = '#111'; c.fillText('grey start, green goal, red joint', 8, W - 8);
}
function panel(c, y0, h, ys, line, label) {
  const W = 600; c.strokeStyle = '#ccc'; c.strokeRect(40, y0 + 5, W - 50, h);
  c.fillStyle = '#111'; c.fillText(label, 44, y0 + 18);
  if (!ys.length) return;
  const all = line === undefined ? ys : ys.concat([line]);
  let lo = Math.min(...all), hi = Math.max(...all); if (!(hi > lo)) { lo -= 1; hi += 1; }
  const X = i => 40 + (W - 50) * i / Math.max(ys.length - 1, 1), Y = v => y0 + 5 + h * (hi - v) / (hi - lo);
  c.fillText(hi.toFixed(2), 2, y0 + 14); c.fillText(lo.toFixed(2), 2, y0 + h + 4);
  if (line !== undefined) { c.strokeStyle = '#2a2'; c.beginPath(); c.moveTo(40, Y(line)); c.lineTo(W - 10, Y(line)); c.stroke(); }
  c.strokeStyle = '#c22'; c.beginPath(); ys.forEach((v, i) => i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v))); c.stroke();
}
function drawPlot() {
  const c = $('plot').getContext('2d'); c.clearRect(0, 0, 600, 300);
  panel(c, 0, 140, ticks.map(t => t.joint_position), st.goal, 'joint [rad] (green: goal)');
  panel(c, 150, 140, ticks.map(t => t.signed_drive), 0, 'signed drive');
}
function draw() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; drawRing(); drawPlot(); }); }
const es = new EventSource('/events');
es.addEventListener('status', e => {
  st = JSON.parse(e.data); if (st.status === 'running') ticks = [];
  if (['done', 'reset', 'ready', 'error'].includes(st.status)) loadJoints(false);
  showStatus(); draw();
});
es.addEventListener('tick', e => { const t = JSON.parse(e.data); ticks.push(t); $('pos').textContent = f(t.joint_position); showStatus(); draw(); });
select();
</script></body></html>
"""
