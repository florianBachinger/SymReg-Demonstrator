"use strict";

/* ==========================================================================
   Canvas & Coordinate Setup
   ========================================================================== */
const canvas       = document.getElementById("c");
const ctx          = canvas.getContext("2d");
const overlay      = document.getElementById("overlay");
const paretoPanel  = document.getElementById("pareto-panel");

const XMIN = -10, XMAX = 10, YMIN = -10, YMAX = 10;

function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  redraw();
}

window.addEventListener("resize", resize);

function mathToCanvas(mx, my) {
  const cx = (mx - XMIN) / (XMAX - XMIN) * canvas.width;
  const cy = (1 - (my - YMIN) / (YMAX - YMIN)) * canvas.height;
  return [cx, cy];
}

function canvasToMath(cx, cy) {
  const mx = XMIN + cx / canvas.width  * (XMAX - XMIN);
  const my = YMIN + (1 - cy / canvas.height) * (YMAX - YMIN);
  return [mx, my];
}

/* ==========================================================================
   Application State
   ========================================================================== */
let drawnPoints      = [];
let drawnCanvasPath  = [];
let fittedCurve      = null;
let expression       = null;
let computeStartTime = null;
let isDrawing        = false;
let paretoData       = [];
let selectedParetoIdx = -1;

const runtimeDisplay = document.getElementById("runtime-display");

/** Computation state machine: "idle" | "running" | "paused" */
let computeState = "idle";

/* ==========================================================================
   Settings Panel
   ========================================================================== */
const burgerBtn        = document.getElementById("burger-btn");
const settingsPanel    = document.getElementById("settings-panel");
const settingsBackdrop = document.getElementById("settings-backdrop");
const settingsClose    = settingsPanel.querySelector(".settings-close");

function openSettings() {
  settingsPanel.classList.add("open");
  settingsBackdrop.classList.add("open");
}

function closeSettings() {
  settingsPanel.classList.remove("open");
  settingsBackdrop.classList.remove("open");
}

burgerBtn.addEventListener("click", () => {
  settingsPanel.classList.contains("open") ? closeSettings() : openSettings();
});
settingsBackdrop.addEventListener("click", closeSettings);
settingsClose.addEventListener("click", closeSettings);

/* ==========================================================================
   Symbol Chips
   ========================================================================== */
const SYMBOL_GROUPS = [
  { label: "Arithmetic",        symbols: ["add", "sub", "mul", "div", "aq"] },
  { label: "Powers / Roots",    symbols: ["pow", "powabs", "square", "sqrt", "sqrtabs", "cbrt", "abs"] },
  { label: "Trigonometric",     symbols: ["sin", "cos", "tan", "asin", "acos", "atan"] },
  { label: "Hyperbolic",        symbols: ["sinh", "cosh", "tanh"] },
  { label: "Exponential / Log", symbols: ["exp", "log", "logabs", "log1p"] },
  { label: "Other",             symbols: ["fmin", "fmax", "ceil", "floor"] },
];

const LOCKED_SYMBOLS = new Set(["constant", "variable"]);
const DEFAULT_ACTIVE = new Set(["add", "sub", "mul", "div", "constant", "variable"]);
let activeSymbols    = new Set(DEFAULT_ACTIVE);

function buildSymbolChips() {
  const container = document.getElementById("symbol-chips-container");
  container.innerHTML = "";

  // Locked (required) chips
  const lockedGroup = document.createElement("div");
  lockedGroup.className = "symbol-group";
  lockedGroup.innerHTML = '<div class="symbol-group-label">Required</div>';

  const lockedChips = document.createElement("div");
  lockedChips.className = "symbol-chips";
  for (const s of LOCKED_SYMBOLS) {
    const chip = document.createElement("span");
    chip.className = "symbol-chip active locked";
    chip.textContent = s;
    lockedChips.appendChild(chip);
  }
  lockedGroup.appendChild(lockedChips);
  container.appendChild(lockedGroup);

  // Togglable groups
  for (const g of SYMBOL_GROUPS) {
    const group = document.createElement("div");
    group.className = "symbol-group";
    group.innerHTML = '<div class="symbol-group-label">' + g.label + '</div>';

    const chips = document.createElement("div");
    chips.className = "symbol-chips";

    for (const s of g.symbols) {
      const chip = document.createElement("span");
      chip.className = "symbol-chip" + (activeSymbols.has(s) ? " active" : "");
      chip.textContent = s;
      chip.addEventListener("click", () => {
        if (activeSymbols.has(s)) {
          activeSymbols.delete(s);
          chip.classList.remove("active");
        } else {
          activeSymbols.add(s);
          chip.classList.add("active");
        }
      });
      chips.appendChild(chip);
    }

    group.appendChild(chips);
    container.appendChild(group);
  }
}

buildSymbolChips();

/* ==========================================================================
   Gather Parameters from Settings Panel
   ========================================================================== */
function gatherParams() {
  const allSymbols = new Set([...LOCKED_SYMBOLS, ...activeSymbols]);

  const p = {
    allowed_symbols:          [...allSymbols].join(","),
    generations:              parseInt(document.getElementById("p-generations").value) || 30,
    population_size:          parseInt(document.getElementById("p-population_size").value) || 200,
    max_length:               parseInt(document.getElementById("p-max_length").value) || 20,
    mutation_probability:     parseFloat(document.getElementById("p-mutation_probability").value) || 0.15,
    tournament_size:          parseInt(document.getElementById("p-tournament_size").value) || 3,
    optimizer_iterations:     parseInt(document.getElementById("p-optimizer_iterations").value) || 10,
    max_evaluations:          parseInt(document.getElementById("p-max_evaluations").value) || 100000,
    model_selection_criterion: document.getElementById("p-model_selection_criterion").value,
    optimizer:                document.getElementById("p-optimizer").value.split(" ")[0],
    pool_size:                parseInt(document.getElementById("p-pool_size").value) || 200,
    epsilon:                  parseFloat(document.getElementById("p-epsilon").value) || 1e-5,
    uncertainty:              parseFloat(document.getElementById("p-uncertainty").value) || 0.05,
    n_threads:                parseInt(document.getElementById("p-n_threads").value) || 0,
  };

  const rs = document.getElementById("p-random_state").value;
  p.random_state = rs === "" ? null : parseInt(rs);

  return p;
}

/* ==========================================================================
   Control Buttons (Pause / Resume / Stop)
   ========================================================================== */
const controlsDiv = document.getElementById("controls");
const btnPause    = document.getElementById("btn-pause");
const btnResume   = document.getElementById("btn-resume");
const btnStop     = document.getElementById("btn-stop");

function updateControlButtons() {
  if (computeState === "idle") {
    controlsDiv.classList.add("hidden");
  } else if (computeState === "running") {
    controlsDiv.classList.remove("hidden");
    btnPause.style.display  = "flex";
    btnResume.style.display = "none";
    btnStop.style.display   = "flex";
  } else if (computeState === "paused") {
    controlsDiv.classList.remove("hidden");
    btnPause.style.display  = "none";
    btnResume.style.display = "flex";
    btnStop.style.display   = "flex";
  }
}

btnPause.addEventListener("click", () => {
  if (computeState !== "running" || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "pause" }));
});

btnResume.addEventListener("click", () => {
  if (computeState !== "paused" || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "resume", params: gatherParams() }));
});

btnStop.addEventListener("click", () => {
  if (computeState === "idle" || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "stop" }));

  drawnPoints       = [];
  drawnCanvasPath   = [];
  fittedCurve       = null;
  expression        = null;
  paretoData        = [];
  selectedParetoIdx = -1;
  paretoPanel.style.display = "none";
  computeState      = "idle";

  updateControlButtons();
  overlay.innerHTML = '<span class="hint">Draw a curve to fit</span>';
  redraw();
});

/* ==========================================================================
   Grid Drawing
   ========================================================================== */
function drawGrid() {
  const w = canvas.width, h = canvas.height;

  ctx.fillStyle = "#1e1e2e";
  ctx.fillRect(0, 0, w, h);

  // Minor gridlines
  ctx.strokeStyle = "#313244";
  ctx.lineWidth   = 0.5;
  for (let x = Math.ceil(XMIN); x <= XMAX; x++) {
    const [cx] = mathToCanvas(x, 0);
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
  }
  for (let y = Math.ceil(YMIN); y <= YMAX; y++) {
    const [, cy] = mathToCanvas(0, y);
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
  }

  // Major gridlines (every 5 units)
  ctx.strokeStyle = "#45475a";
  ctx.lineWidth   = 1;
  for (let x = Math.ceil(XMIN / 5) * 5; x <= XMAX; x += 5) {
    const [cx] = mathToCanvas(x, 0);
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
  }
  for (let y = Math.ceil(YMIN / 5) * 5; y <= YMAX; y += 5) {
    const [, cy] = mathToCanvas(0, y);
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "#585b70";
  ctx.lineWidth   = 1.5;
  const [ax]  = mathToCanvas(0, 0);
  const [, ay] = mathToCanvas(0, 0);
  ctx.beginPath(); ctx.moveTo(ax, 0); ctx.lineTo(ax, h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, ay); ctx.lineTo(w, ay); ctx.stroke();

  // Axis labels
  ctx.fillStyle    = "#7f849c";
  ctx.font         = "11px system-ui, sans-serif";
  ctx.textAlign    = "center";
  ctx.textBaseline = "top";
  for (let x = Math.ceil(XMIN); x <= XMAX; x++) {
    if (x === 0) continue;
    const [cx] = mathToCanvas(x, 0);
    ctx.fillText(x, cx, ay + 4);
  }
  ctx.textAlign    = "right";
  ctx.textBaseline = "middle";
  for (let y = Math.ceil(YMIN); y <= YMAX; y++) {
    if (y === 0) continue;
    const [, cy] = mathToCanvas(0, y);
    ctx.fillText(y, ax - 6, cy);
  }
}

/* ==========================================================================
   Curve Drawing (user stroke & fitted result)
   ========================================================================== */
function drawUserCurve() {
  if (drawnCanvasPath.length < 2) return;

  ctx.strokeStyle = "#cdd6f4";
  ctx.lineWidth   = 2.5;
  ctx.lineJoin    = "round";
  ctx.lineCap     = "round";

  ctx.beginPath();
  ctx.moveTo(drawnCanvasPath[0][0], drawnCanvasPath[0][1]);
  for (let i = 1; i < drawnCanvasPath.length; i++) {
    ctx.lineTo(drawnCanvasPath[i][0], drawnCanvasPath[i][1]);
  }
  ctx.stroke();
}

function drawFittedCurve() {
  if (!fittedCurve || fittedCurve.length < 2) return;

  ctx.strokeStyle = "#f38ba8";
  ctx.lineWidth   = 3;
  ctx.lineJoin    = "round";
  ctx.setLineDash([]);

  ctx.beginPath();
  let started = false;
  for (const [mx, my] of fittedCurve) {
    if (my < YMIN - 5 || my > YMAX + 5) { started = false; continue; }
    const [cx, cy] = mathToCanvas(mx, my);
    if (!started) { ctx.moveTo(cx, cy); started = true; }
    else          { ctx.lineTo(cx, cy); }
  }
  ctx.stroke();
}

function redraw() {
  drawGrid();
  drawUserCurve();
  drawFittedCurve();
}

/* ==========================================================================
   Pointer Events (drawing)
   ========================================================================== */
canvas.addEventListener("pointerdown", (e) => {
  isDrawing = true;
  canvas.setPointerCapture(e.pointerId);

  // Cancel any in-progress computation
  if (computeState !== "idle" && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "cancel" }));
  }

  // Clear previous drawing & result
  drawnPoints     = [];
  drawnCanvasPath = [];
  fittedCurve     = null;
  expression      = null;
  computeState    = "idle";
  updateControlButtons();
  paretoPanel.style.display    = "none";
  runtimeDisplay.style.display = "none";

  const [mx, my] = canvasToMath(e.offsetX, e.offsetY);
  drawnPoints.push([mx, my]);
  drawnCanvasPath.push([e.offsetX, e.offsetY]);

  redraw();
  overlay.innerHTML = '<span class="hint">Drawing...</span>';
});

canvas.addEventListener("pointermove", (e) => {
  if (!isDrawing) return;

  const [mx, my] = canvasToMath(e.offsetX, e.offsetY);
  drawnPoints.push([mx, my]);
  drawnCanvasPath.push([e.offsetX, e.offsetY]);

  // Draw the latest segment incrementally
  const len = drawnCanvasPath.length;
  if (len >= 2) {
    ctx.strokeStyle = "#cdd6f4";
    ctx.lineWidth   = 2.5;
    ctx.lineJoin    = "round";
    ctx.lineCap     = "round";
    ctx.beginPath();
    ctx.moveTo(drawnCanvasPath[len - 2][0], drawnCanvasPath[len - 2][1]);
    ctx.lineTo(drawnCanvasPath[len - 1][0], drawnCanvasPath[len - 1][1]);
    ctx.stroke();
  }
});

canvas.addEventListener("pointerup", (e) => {
  if (!isDrawing) return;
  isDrawing = false;

  if (drawnPoints.length < 5) {
    overlay.innerHTML = '<span class="hint">Draw a longer curve</span>';
    return;
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "fit", points: drawnPoints, params: gatherParams() }));
    computeState     = "running";
    computeStartTime = performance.now();
    runtimeDisplay.style.display = "none";
    updateControlButtons();
    overlay.innerHTML = '<span class="status">Computing symbolic regression\u2026</span>';
  } else {
    overlay.innerHTML = '<span class="status" style="color:#f38ba8">Disconnected \u2013 reload page</span>';
  }
});

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

/* ==========================================================================
   LaTeX Rendering & Pareto Table
   ========================================================================== */
function renderExpressionOverlay(latex) {
  overlay.innerHTML = '<span class="expr" id="katex-target"></span>';
  const el = document.getElementById("katex-target");

  function tryRender() {
    if (typeof katex !== "undefined") {
      try {
        katex.render(latex, el, { throwOnError: false, displayMode: false });
      } catch (e) {
        el.textContent = latex;
      }
    } else {
      setTimeout(tryRender, 100);
    }
  }
  tryRender();
}

function buildParetoTable() {
  let html = '<table><tr><th>R\u00b2</th><th>Len</th><th>MDL</th><th>Expression</th></tr>';
  for (let i = 0; i < paretoData.length; i++) {
    const p    = paretoData[i];
    const sel  = i === selectedParetoIdx ? " selected" : "";
    const best = i === selectedParetoIdx ? " best-row" : "";
    html += '<tr class="pareto-row' + sel + best + '" data-idx="' + i + '">';
    html += '<td>' + p.r2 + '</td><td>' + p.length + '</td><td>' + p.mdl + '</td>';
    html += '<td id="pareto-expr-' + i + '"></td></tr>';
  }
  html += '</table>';
  paretoPanel.innerHTML = html;

  // Render KaTeX in each row
  for (let i = 0; i < paretoData.length; i++) {
    const el = document.getElementById("pareto-expr-" + i);
    if (el && typeof katex !== "undefined") {
      try {
        katex.render(paretoData[i].latex || paretoData[i].expr, el, { throwOnError: false, displayMode: false });
      } catch (e) {
        el.textContent = paretoData[i].expr;
      }
    } else if (el) {
      el.textContent = paretoData[i].expr;
    }
  }

  // Row click handlers
  paretoPanel.querySelectorAll(".pareto-row").forEach((row) => {
    row.addEventListener("click", () => {
      selectParetoSolution(parseInt(row.dataset.idx));
    });
  });
}

function selectParetoSolution(idx) {
  if (idx < 0 || idx >= paretoData.length) return;

  selectedParetoIdx = idx;
  const p = paretoData[idx];

  if (p.curve && p.curve.length > 0) fittedCurve = p.curve;
  renderExpressionOverlay(p.latex || p.expr);
  redraw();

  paretoPanel.querySelectorAll(".pareto-row").forEach((row, i) => {
    row.classList.toggle("selected", i === idx);
  });
}

/* ==========================================================================
   WebSocket Connection
   ========================================================================== */
let ws;

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws");

  ws.onopen = () => {
    overlay.innerHTML = '<span class="hint">Draw a curve to fit</span>';
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);

    switch (msg.type) {
      case "result":
        computeState = "idle";
        updateControlButtons();

        if (computeStartTime !== null) {
          const elapsed = ((performance.now() - computeStartTime) / 1000).toFixed(3);
          runtimeDisplay.textContent   = elapsed + " s";
          runtimeDisplay.style.display = "block";
          computeStartTime = null;
        }

        expression        = msg.expression;
        fittedCurve       = msg.curve;
        paretoData        = msg.pareto || [];
        selectedParetoIdx = msg.bestIdx != null ? msg.bestIdx : -1;

        renderExpressionOverlay(msg.latex || msg.expression);
        redraw();

        if (paretoData.length > 0) {
          buildParetoTable();
          paretoPanel.style.display = "block";
        }
        break;

      case "error":
        computeState = "idle";
        updateControlButtons();
        overlay.innerHTML = '<span class="status" style="color:#f38ba8">' + escapeHtml(msg.message) + '</span>';
        break;

      case "paused":
        computeState = "paused";
        updateControlButtons();
        overlay.innerHTML = '<span class="status" style="color:#f9e2af">Paused \u2013 change settings or resume</span>';
        break;

      case "resumed":
        computeState = "running";
        updateControlButtons();
        overlay.innerHTML = '<span class="status">Computing symbolic regression\u2026</span>';
        break;

      case "stopped":
        computeState      = "idle";
        updateControlButtons();
        drawnPoints       = [];
        drawnCanvasPath   = [];
        fittedCurve       = null;
        expression        = null;
        paretoData        = [];
        selectedParetoIdx = -1;
        paretoPanel.style.display = "none";
        overlay.innerHTML = '<span class="hint">Draw a curve to fit</span>';
        redraw();
        break;
    }
  };

  ws.onclose = () => {
    computeState = "idle";
    updateControlButtons();
    overlay.innerHTML = '<span class="status" style="color:#f38ba8">Disconnected \u2013 reconnecting\u2026</span>';
    setTimeout(connectWS, 2000);
  };
}

/* ==========================================================================
   Utilities
   ========================================================================== */
function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* ==========================================================================
   Initialise
   ========================================================================== */
connectWS();
resize();
