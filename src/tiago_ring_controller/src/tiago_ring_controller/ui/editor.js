/* Ring blocks editor: plain JavaScript + SVG, no dependencies.
 *
 * The document in memory has the graph file's shape (schema, name, simulation,
 * blocks[{id,type,params,ui}], edges[{from,to,params}], robot).  The server is
 * the authority: Validate / Save / Run post the document and get back the
 * canonical one (defaults filled, problems listed), so a saved file is
 * byte-identical to what Graph.save writes for the same content.
 */
(function () {
  "use strict";
  const SVG = "http://www.w3.org/2000/svg";
  const BLOCK_W = 150, ROW_H = 18, HEAD_H = 34, PORT_R = 5;
  const state = {
    types: {},                       // type name -> description (params, ports, neural)
    doc: emptyDocument(),
    selected: null,                  // {kind: "block"|"edge", id|index}
    drag: null,                      // block move or edge creation in progress
    primary: null,                   // ids of the blocks the live overlay annotates
    live: null,
  };

  function emptyDocument() {
    return {
      schema: "ring-blocks/1", name: "untitled",
      simulation: { dt_ms: 50.0, nest_lead_steps: 4, max_steps: 400, rng_seed: 13579, local_num_threads: 1, step_mode: "run", reset_mode: "rebuild" },
      blocks: [], edges: [], robot: { engine: "gazebo", stepper: "clock_wait" },
    };
  }

  // -- helpers -----------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  function el(tag, attrs, parent) {
    const node = document.createElementNS(SVG, tag);
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function html(tag, attrs, text, parent) {
    const node = document.createElement(tag);
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function setStatus(text, error) {
    const node = $("status");
    node.textContent = text;
    node.className = "status" + (error ? " error" : "");
  }
  async function api(path, payload) {
    const options = payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok && data.error) throw new Error(data.error);
    return data;
  }
  function blockById(id) { return state.doc.blocks.find((b) => b.id === id); }
  function typeOf(block) { return state.types[block.type]; }
  function ports(block, direction) {
    const type = typeOf(block);
    return type ? type.ports.filter((p) => p.direction === direction) : [];
  }
  function blockHeight(block) {
    const rows = Math.max(ports(block, "in").length, ports(block, "out").length, 1);
    return HEAD_H + rows * ROW_H + 8;
  }
  function portPosition(block, portName) {
    const ins = ports(block, "in"), outs = ports(block, "out");
    let index = ins.findIndex((p) => p.name === portName);
    if (index >= 0) return { x: block.ui.x, y: block.ui.y + HEAD_H + index * ROW_H + ROW_H / 2, port: ins[index] };
    index = outs.findIndex((p) => p.name === portName);
    if (index >= 0) return { x: block.ui.x + BLOCK_W, y: block.ui.y + HEAD_H + index * ROW_H + ROW_H / 2, port: outs[index] };
    return null;
  }
  function freshId(type) {
    const base = type.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
    let n = 1;
    while (blockById(base + n)) n += 1;
    return base + n;
  }
  function defaults(type) {
    const values = {};
    for (const spec of state.types[type].params) values[spec.name] = JSON.parse(JSON.stringify(spec.default));
    return values;
  }

  // -- rendering --------------------------------------------------------------
  function render() {
    renderBlocks();
    renderEdges();
    renderInspector();
    $("graph-name").value = state.doc.name;
  }

  function renderBlocks() {
    const layer = $("blocks");
    layer.innerHTML = "";
    for (const block of state.doc.blocks) {
      if (!block.ui) block.ui = { x: 40, y: 40 };
      const type = typeOf(block);
      const group = el("g", { class: "block " + (type && type.neural ? "neural" : "signal") + (isSelected("block", block.id) ? " selected" : ""), "data-id": block.id, transform: "translate(" + block.ui.x + "," + block.ui.y + ")" }, layer);
      const height = blockHeight(block);
      el("rect", { class: "body", width: BLOCK_W, height: height }, group);
      el("text", { class: "title", x: 8, y: 16 }, group).textContent = block.id;
      el("text", { class: "type", x: 8, y: 29 }, group).textContent = block.type;
      ports(block, "in").forEach((port, index) => {
        const y = HEAD_H + index * ROW_H + ROW_H / 2;
        el("circle", { class: "port " + port.kind, cx: 0, cy: y, r: PORT_R, "data-block": block.id, "data-port": port.name, "data-direction": "in", "data-kind": port.kind }, group);
        el("text", { class: "port-label", x: 9, y: y + 4 }, group).textContent = port.name;
      });
      ports(block, "out").forEach((port, index) => {
        const y = HEAD_H + index * ROW_H + ROW_H / 2;
        el("circle", { class: "port " + port.kind, cx: BLOCK_W, cy: y, r: PORT_R, "data-block": block.id, "data-port": port.name, "data-direction": "out", "data-kind": port.kind }, group);
        el("text", { class: "port-label", x: BLOCK_W - 9, y: y + 4, "text-anchor": "end" }, group).textContent = port.name;
      });
      renderOverlay(block, group, height);
      group.addEventListener("mousedown", (event) => onBlockMouseDown(event, block));
    }
  }

  function renderOverlay(block, group, height) {
    const live = state.live, primary = state.primary;
    if (!live || !primary) return;
    if (block.id === primary.state_ring && Array.isArray(live.r1)) {
      const n = live.r1.length, max = Math.max(1, ...live.r1);
      const width = (BLOCK_W - 16) / n;
      for (let i = 0; i < n; i += 1) {
        if (live.r1[i] <= 0) continue;
        el("rect", { class: "bump", x: 8 + i * width, y: height - 4 - 10 * live.r1[i] / max, width: Math.max(width, 1), height: 10 * live.r1[i] / max }, group);
      }
      if (live.centroid != null) el("text", { class: "overlay", x: BLOCK_W - 8, y: 16, "text-anchor": "end" }, group).textContent = "centroid " + live.centroid.toFixed(1);
    } else if (block.id === primary.gain && live.left != null) {
      el("text", { class: "overlay", x: BLOCK_W - 8, y: 16, "text-anchor": "end" }, group).textContent = "L " + live.left + " R " + live.right;
    } else if (block.id === primary.decoder && live.drive != null) {
      el("text", { class: "overlay", x: BLOCK_W - 8, y: 16, "text-anchor": "end" }, group).textContent = "drive " + live.drive.toFixed(2);
    } else if (block.id === primary.joint && live.joint_position != null) {
      el("text", { class: "overlay", x: BLOCK_W - 8, y: 16, "text-anchor": "end" }, group).textContent = live.joint_position.toFixed(3) + " rad";
    }
  }

  function edgePath(a, b) {
    const dx = Math.max(40, Math.abs(b.x - a.x) / 2);
    return "M " + a.x + " " + a.y + " C " + (a.x + dx) + " " + a.y + ", " + (b.x - dx) + " " + b.y + ", " + b.x + " " + b.y;
  }

  function renderEdges() {
    const layer = $("edges");
    layer.innerHTML = "";
    state.doc.edges.forEach((edge, index) => {
      const [fromId, fromPort] = splitRef(edge.from), [toId, toPort] = splitRef(edge.to);
      const source = blockById(fromId), target = blockById(toId);
      if (!source || !target) return;
      const a = portPosition(source, fromPort), b = portPosition(target, toPort);
      if (!a || !b) return;
      const path = el("path", { class: "edge " + a.port.kind + (isSelected("edge", index) ? " selected" : ""), d: edgePath(a, b), "marker-end": "url(#arrow)", "data-index": index }, layer);
      path.addEventListener("mousedown", (event) => { event.stopPropagation(); select("edge", index); });
      if (edge.params && Object.keys(edge.params).length) {
        el("text", { class: "port-label", x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 4, "text-anchor": "middle" }, layer).textContent = JSON.stringify(edge.params);
      }
    });
  }

  function splitRef(ref) { const k = ref.lastIndexOf("."); return [ref.slice(0, k), ref.slice(k + 1)]; }
  function isSelected(kind, key) { return state.selected && state.selected.kind === kind && state.selected.key === key; }
  function select(kind, key) { state.selected = kind == null ? null : { kind, key }; render(); }

  // -- inspector ----------------------------------------------------------------
  function field(parent, spec, value, onChange) {
    const wrap = html("div", { class: "field" }, null, parent);
    html("label", {}, spec.name + (spec.unit ? " [" + spec.unit + "]" : ""), wrap);
    let input;
    if (spec.choices) {
      input = html("select", {}, null, wrap);
      for (const choice of spec.choices) { const option = html("option", { value: choice }, String(choice), input); if (choice === value) option.selected = true; }
      input.addEventListener("change", () => onChange(input.value, wrap));
    } else if (spec.type === "bool") {
      input = html("input", { type: "checkbox" }, null, wrap);
      input.checked = !!value;
      input.addEventListener("change", () => onChange(input.checked, wrap));
    } else if (spec.type === "dict" || spec.type === "list") {
      input = html("textarea", { rows: 2 }, JSON.stringify(value), wrap);
      input.addEventListener("change", () => { try { onChange(JSON.parse(input.value), wrap); wrap.classList.remove("invalid"); } catch (e) { wrap.classList.add("invalid"); } });
    } else {
      input = html("input", { type: spec.type === "str" ? "text" : "number" }, null, wrap);
      if (spec.type !== "str") { input.step = spec.type === "int" ? "1" : "any"; if (spec.minimum != null) input.min = spec.minimum; if (spec.maximum != null) input.max = spec.maximum; }
      input.value = value;
      input.addEventListener("change", () => {
        let parsed = input.value;
        if (spec.type === "int") parsed = parseInt(input.value, 10);
        if (spec.type === "float") parsed = parseFloat(input.value);
        const bad = (spec.type !== "str" && !Number.isFinite(parsed)) || (spec.minimum != null && parsed < spec.minimum) || (spec.maximum != null && parsed > spec.maximum);
        wrap.classList.toggle("invalid", !!bad);
        if (!bad) onChange(parsed, wrap);
      });
    }
    if (spec.doc) html("div", { class: "doc" }, spec.doc, wrap);
    return input;
  }

  function renderInspector() {
    const body = $("inspector-body");
    body.innerHTML = "";
    if (!state.selected) { html("p", { class: "hint" }, "Select a block to edit its parameters; drag from an output port (right) to an input port (left) to connect; Delete removes the selection.", body); return; }
    if (state.selected.kind === "edge") {
      const edge = state.doc.edges[state.selected.key];
      if (!edge) return;
      html("p", {}, edge.from + " → " + edge.to, body);
      field(body, { name: "params", type: "dict", doc: "edge parameters (weight, margin)" }, edge.params || {}, (value) => { edge.params = value; renderEdges(); });
      html("button", {}, "Delete edge", body).addEventListener("click", () => { state.doc.edges.splice(state.selected.key, 1); select(null); });
      return;
    }
    const block = blockById(state.selected.key);
    if (!block) return;
    const type = typeOf(block);
    const idInput = field(body, { name: "id", type: "str", doc: "unique, no dots" }, block.id, (value) => renameBlock(block, value));
    idInput.pattern = "[^.\\s]+";
    html("p", { class: "hint" }, type.type + (type.doc ? " — " + type.doc : ""), body);
    for (const spec of type.params) field(body, spec, block.params[spec.name], (value) => { block.params[spec.name] = value; });
    html("button", {}, "Delete block", body).addEventListener("click", () => removeBlock(block.id));
  }

  function renameBlock(block, newId) {
    if (!newId || newId.indexOf(".") >= 0 || (newId !== block.id && blockById(newId))) return;
    for (const edge of state.doc.edges) {
      const [f, fp] = splitRef(edge.from), [t, tp] = splitRef(edge.to);
      if (f === block.id) edge.from = newId + "." + fp;
      if (t === block.id) edge.to = newId + "." + tp;
    }
    block.id = newId;
    state.selected = { kind: "block", key: newId };
    render();
  }
  function removeBlock(id) {
    state.doc.blocks = state.doc.blocks.filter((b) => b.id !== id);
    state.doc.edges = state.doc.edges.filter((e) => splitRef(e.from)[0] !== id && splitRef(e.to)[0] !== id);
    select(null);
  }

  function renderSimulationForms() {
    const simulation = $("simulation-form");
    simulation.innerHTML = "";
    const specs = [
      { name: "dt_ms", type: "float", minimum: 1 }, { name: "nest_lead_steps", type: "int", minimum: 0 }, { name: "max_steps", type: "int", minimum: 1 },
      { name: "rng_seed", type: "int" }, { name: "local_num_threads", type: "int", minimum: 1 },
      { name: "step_mode", type: "str", choices: ["run", "simulate"] }, { name: "reset_mode", type: "str", choices: ["rebuild", "continue"] },
    ];
    for (const spec of specs) field(simulation, spec, state.doc.simulation[spec.name], (value) => { state.doc.simulation[spec.name] = value; });
    const robot = $("robot-form");
    robot.innerHTML = "";
    field(robot, { name: "engine", type: "str", choices: ["fake", "gazebo"] }, state.doc.robot.engine, (value) => { state.doc.robot.engine = value; });
    field(robot, { name: "stepper", type: "str", choices: ["clock_wait", "plugin"] }, state.doc.robot.stepper, (value) => { state.doc.robot.stepper = value; });
  }

  // -- interaction ----------------------------------------------------------------
  function canvasPoint(event) {
    const svg = $("canvas"), rect = svg.getBoundingClientRect();
    return { x: event.clientX - rect.left + $("canvas-wrap").scrollLeft, y: event.clientY - rect.top + $("canvas-wrap").scrollTop };
  }
  function onBlockMouseDown(event, block) {
    event.stopPropagation();
    const target = event.target;
    if (target.classList && target.classList.contains("port")) {
      if (target.getAttribute("data-direction") !== "out") return;
      state.drag = { kind: "edge", from: block.id + "." + target.getAttribute("data-port"), portKind: target.getAttribute("data-kind"), start: canvasPoint(event) };
      return;
    }
    const point = canvasPoint(event);
    state.drag = { kind: "move", id: block.id, dx: point.x - block.ui.x, dy: point.y - block.ui.y };
    select("block", block.id);
  }
  function onMouseMove(event) {
    if (!state.drag) return;
    const point = canvasPoint(event);
    if (state.drag.kind === "move") {
      const block = blockById(state.drag.id);
      block.ui.x = Math.max(0, Math.round(point.x - state.drag.dx));
      block.ui.y = Math.max(0, Math.round(point.y - state.drag.dy));
      renderBlocks();
      renderEdges();
    } else {
      $("drag-edge").setAttribute("d", edgePath(state.drag.start, point));
    }
  }
  function onMouseUp(event) {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;
    $("drag-edge").setAttribute("d", "");
    if (drag.kind !== "edge") return;
    const target = document.elementFromPoint(event.clientX, event.clientY);
    if (!target || !target.classList || !target.classList.contains("port") || target.getAttribute("data-direction") !== "in") return;
    const to = target.getAttribute("data-block") + "." + target.getAttribute("data-port");
    const message = connect(drag.from, to);
    setStatus(message.error || "connected " + drag.from + " → " + to, !!message.error);
    render();
  }

  /* Connect two ports with the browser-side checks (kind, direction, one producer per
   * non-multi input); the server validates again.  Used by mouse-up and the self-test. */
  function connect(from, to, params) {
    const [fromId, fromPort] = splitRef(from), [toId, toPort] = splitRef(to);
    const source = blockById(fromId), target = blockById(toId);
    if (!source || !target) return { error: "unknown block in " + from + " → " + to };
    const out = typeOf(source).ports.find((p) => p.name === fromPort && p.direction === "out");
    const port = typeOf(target).ports.find((p) => p.name === toPort && p.direction === "in");
    if (!out || !port) return { error: "not an output → input pair: " + from + " → " + to };
    if (out.kind !== port.kind) return { error: "cannot connect " + out.kind + " to " + port.kind };
    if (!port.multi) state.doc.edges = state.doc.edges.filter((e) => e.to !== to);
    if (!state.doc.edges.some((e) => e.from === from && e.to === to)) {
      const edge = { from: from, to: to };
      if (params && Object.keys(params).length) edge.params = params;
      state.doc.edges.push(edge);
    }
    return {};
  }
  function onKeyDown(event) {
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    if (document.activeElement && ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (!state.selected) return;
    if (state.selected.kind === "block") removeBlock(state.selected.key);
    else { state.doc.edges.splice(state.selected.key, 1); select(null); }
  }

  function addBlock(type) {
    const block = { id: freshId(type), type: type, params: defaults(type), ui: { x: 40 + 30 * (state.doc.blocks.length % 8), y: 40 + 60 * (state.doc.blocks.length % 10) } };
    state.doc.blocks.push(block);
    select("block", block.id);
  }

  // -- documents ----------------------------------------------------------------
  function loadDocument(doc, path) {
    state.doc = JSON.parse(JSON.stringify(doc));
    if (!state.doc.robot) state.doc.robot = { engine: "gazebo", stepper: "clock_wait" };
    state.doc.blocks.forEach((block, index) => {
      if (!block.params) block.params = {};
      if (!block.ui) block.ui = { x: 40 + 190 * (index % 6), y: 40 + 130 * Math.floor(index / 6) };
    });
    state.selected = null;
    if (path) $("file-path").value = path;
    renderSimulationForms();
    render();
    showProblems([]);
  }
  function currentDocument() {
    state.doc.name = $("graph-name").value;
    return state.doc;
  }
  function showProblems(problems) {
    const list = $("problems");
    list.innerHTML = "";
    for (const problem of problems || []) html("li", {}, problem, list);
  }
  async function validate() {
    const result = await api("/api/graph/validate", { graph: currentDocument() });
    showProblems(result.problems);
    if (result.valid) { setStatus("valid: " + result.graph.blocks.length + " blocks, " + result.graph.edges.length + " edges"); return result; }
    setStatus(result.problems.length + " problem(s)", true);
    return result;
  }
  async function save() {
    const path = $("file-path").value.trim();
    if (!path) { setStatus("give a file path to save to", true); return; }
    const result = await api("/api/graph/save", { graph: currentDocument(), path: path });
    showProblems(result.problems);
    setStatus(result.saved ? "saved " + result.saved : result.problems.length + " problem(s), not saved", !result.saved);
  }
  async function run() {
    const result = await api("/api/graph", { graph: currentDocument(), path: $("file-path").value.trim() || null, request_id: String(Date.now()) });
    showProblems(result.problems);
    setStatus(result.queued ? "graph sent to the session; start a trial from the dashboard" : result.problems.length + " problem(s)", !result.queued);
  }

  // -- live overlay -------------------------------------------------------------
  function connectEvents() {
    if (/[?&]static=1/.test(window.location.search)) return;   // tests: no event stream
    try {
      const source = new EventSource("/events");
      source.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "status") { if (message.primary) state.primary = message.primary; $("live").textContent = (message.graph ? "graph " + message.graph + " — " : "") + (message.message || message.status); }
        if (message.type === "tick") { state.live = message; renderBlocks(); renderEdges(); }
        if (message.type === "trial_end") { state.live = null; renderBlocks(); }
      };
    } catch (e) { /* no event stream: editing only */ }
  }

  // -- start --------------------------------------------------------------------
  async function start() {
    const blocks = await api("/api/blocks");
    state.types = blocks.types;
    const palette = $("palette-list");
    for (const name in state.types) {
      const type = state.types[name];
      const button = html("button", { class: type.neural ? "neural" : "signal", title: type.doc || "" }, name, palette);
      button.addEventListener("click", () => addBlock(name));
    }
    const examples = await api("/api/graph/examples");
    for (const name of examples.examples) html("option", { value: name }, name, $("example-select"));
    const current = await api("/api/graph");
    if (current.graph) { loadDocument(current.graph, current.path); setStatus("session graph loaded"); }
    else { renderSimulationForms(); render(); setStatus("ready"); }
    $("btn-load-example").addEventListener("click", async () => {
      const name = $("example-select").value;
      if (!name) return;
      const result = await api("/api/graph/examples/" + name);
      loadDocument(result.graph, result.path);
      setStatus("loaded example " + name);
    });
    $("btn-load-file").addEventListener("click", async () => {
      const path = $("file-path").value.trim();
      if (!path) return;
      try { const result = await api("/api/graph/file?path=" + encodeURIComponent(path)); loadDocument(result.graph, result.path); setStatus("loaded " + path); }
      catch (e) { setStatus(e.message, true); }
    });
    $("btn-save").addEventListener("click", () => save().catch((e) => setStatus(e.message, true)));
    $("btn-validate").addEventListener("click", () => validate().catch((e) => setStatus(e.message, true)));
    $("btn-run").addEventListener("click", () => run().catch((e) => setStatus(e.message, true)));
    $("btn-clear").addEventListener("click", () => { loadDocument(emptyDocument()); setStatus("cleared"); });
    $("graph-name").addEventListener("change", () => { state.doc.name = $("graph-name").value; });
    $("canvas").addEventListener("mousedown", () => select(null));
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("keydown", onKeyDown);
    connectEvents();
    document.body.setAttribute("data-editor-ready", "1");
  }
  /* ?selftest=<example>: clear the canvas, add every block of the example through the palette
   * path (fresh id, defaults, rename, parameter edits, positions), draw every edge through
   * connect(), then ask the server for the canonical text and compare it with the example
   * file's.  The result lands in <body data-selftest="ok|..."> for the headless-Chrome test. */
  async function selfTest(name) {
    const example = await api("/api/graph/examples/" + name);
    const reference = example.graph;
    loadDocument(emptyDocument());
    state.doc.name = reference.name;
    $("graph-name").value = reference.name;
    state.doc.simulation = JSON.parse(JSON.stringify(reference.simulation));
    state.doc.robot = JSON.parse(JSON.stringify(reference.robot));
    for (const block of reference.blocks) {
      addBlock(block.type);
      const added = state.doc.blocks[state.doc.blocks.length - 1];
      renameBlock(added, block.id);
      const mine = blockById(block.id);
      for (const key in block.params) mine.params[key] = block.params[key];
      mine.ui = block.ui ? JSON.parse(JSON.stringify(block.ui)) : mine.ui;
    }
    for (const edge of reference.edges) {
      const result = connect(edge.from, edge.to, edge.params);
      if (result.error) { document.body.setAttribute("data-selftest", "connect failed: " + result.error); return; }
    }
    render();
    // The drawn graph carries positions; give the reference the same ones ("same content").
    const expectedDoc = JSON.parse(JSON.stringify(reference));
    for (const block of expectedDoc.blocks) block.ui = JSON.parse(JSON.stringify(blockById(block.id).ui));
    const mine = await api("/api/graph/validate", { graph: currentDocument() });
    const expected = await api("/api/graph/validate", { graph: expectedDoc });
    let verdict = "ok";
    if (!mine.valid) verdict = "invalid: " + mine.problems.join("; ");
    else if (mine.text !== expected.text) {
      const a = mine.text.split("\n"), b = expected.text.split("\n");
      let i = 0;
      while (i < a.length && i < b.length && a[i] === b[i]) i += 1;
      verdict = "text differs at line " + (i + 1) + ": drawn " + JSON.stringify(a[i]) + " vs reference " + JSON.stringify(b[i]);
    }
    document.body.setAttribute("data-selftest", verdict);
    document.body.setAttribute("data-selftest-blocks", String(state.doc.blocks.length));
    document.body.setAttribute("data-selftest-edges", String(state.doc.edges.length));
  }

  window.ringEditor = { state, loadDocument, currentDocument, addBlock, connect, validate, selfTest };
  start().then(() => {
    const match = /[?&]selftest=([a-z_]+)/.exec(window.location.search);
    if (match) return selfTest(match[1]);
  }).catch((e) => setStatus("failed to start: " + e.message, true));
})();
