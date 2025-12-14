/**
 * PARALLEL_TEMPERING - Algorithm-specific code
 * Shared utilities (randn, autocorr, calculateESS, drawTracePlot, drawACFXY) are in shared-utils.js
 */

/***************************************************
 * CANVAS & UI SETUP
 ***************************************************/
const replicaGrid = document.getElementById("replicaGrid");
const swapAcceptanceCanvas = document.getElementById("swapAcceptance");
const tracePlotCanvas = document.getElementById("tracePlot");
const bothCoordinatesCanvas = document.getElementById("bothCoordinates");
const coldChainCanvas = document.getElementById("coldChainPosterior");

const swapAcceptanceCtx = swapAcceptanceCanvas.getContext("2d");
const tracePlotCtx = tracePlotCanvas.getContext("2d");
const bothCoordinatesCtx = bothCoordinatesCanvas.getContext("2d");
const coldChainCtx = coldChainCanvas.getContext("2d");

// Scale canvases for retina
function scaleCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
}

scaleCanvas(swapAcceptanceCanvas);
scaleCanvas(tracePlotCanvas);
scaleCanvas(bothCoordinatesCanvas);
scaleCanvas(coldChainCanvas);

/***************************************************
 * PARAMETERS & STATE
 ***************************************************/
let xmin = -4, xmax = 4, ymin = -4, ymax = 4;
let numTemperatures = 4;
let temperatures = []; // β values
let swapFrequency = 10;
let iteration = 0;
let swapIteration = 0;
let running = false;
let animDelay = 50; // milliseconds, matches default slider value

// Replica state: array of {x, y, beta, samples: [{x, y}], accepts: 0, proposals: 0}
let replicas = [];

// Swap statistics: swapAccepts[k] = accepts between k and k+1, swapAttempts[k] = attempts
let swapAccepts = [];
let swapAttempts = [];

// Cold chain samples (only from Chain 1 which has beta=1.0)
let coldSamples = [];

// Replica canvases
let replicaCanvases = [];
let replicaContexts = [];

/***************************************************
 * UI CONTROLS
 ***************************************************/
document.getElementById("numTemps").addEventListener("input", (e) => {
  numTemperatures = parseInt(e.target.value);
  document.getElementById("numTempsValue").textContent = numTemperatures;
  if (!running) {
    setupTemperatures();
    reset();
  }
});

document.getElementById("tempSpacing").addEventListener("input", (e) => {
  const val = parseInt(e.target.value);
  const spacings = ["Linear", "Geometric", "Exponential", "Adaptive"];
  document.getElementById("tempSpacingValue").textContent = spacings[val - 1];
  if (!running) {
    setupTemperatures();
    reset();
  }
});

document.getElementById("swapFreq").addEventListener("input", (e) => {
  swapFrequency = parseInt(e.target.value);
  document.getElementById("swapFreqValue").textContent = swapFrequency;
});

document.getElementById("speed").addEventListener("input", (e) => {
  animDelay = parseInt(e.target.value);
  document.getElementById("speedVal").textContent = animDelay;
});

document.getElementById("dist").addEventListener("change", () => {
  setDomain();
  reset();
});

/***************************************************
 * TEMPERATURE LADDER SETUP
 ***************************************************/
function setupTemperatures() {
  const spacingType = parseInt(document.getElementById("tempSpacing").value);
  temperatures = [];
  
  // Always have cold chain at beta=1.0
  if (spacingType === 1) {
    // Linear spacing
    for (let k = 0; k < numTemperatures; k++) {
      temperatures.push(1.0 - k * 0.8 / (numTemperatures - 1));
    }
  } else if (spacingType === 2) {
    // Geometric spacing (good default)
    const ratio = Math.pow(0.2, 1.0 / (numTemperatures - 1));
    for (let k = 0; k < numTemperatures; k++) {
      temperatures.push(Math.pow(ratio, k));
    }
  } else if (spacingType === 3) {
    // Exponential spacing (more aggressive)
    for (let k = 0; k < numTemperatures; k++) {
      temperatures.push(Math.exp(-2.0 * k / (numTemperatures - 1)));
    }
  } else {
    // Adaptive (aim for ~30% acceptance between adjacent)
    // For demo, use geometric as approximation
    const ratio = Math.pow(0.15, 1.0 / (numTemperatures - 1));
    for (let k = 0; k < numTemperatures; k++) {
      temperatures.push(Math.pow(ratio, k));
    }
  }
  
  // Ensure cold chain is exactly 1.0
  temperatures[0] = 1.0;
}

/***************************************************
 * TARGET DISTRIBUTIONS
 ***************************************************/
function setDomain() {
  const type = document.getElementById("dist").value;
  if (type === "funnel") {
    xmin = -10; xmax = 10;
    ymin = -50; ymax = 50;
  } else if (type === "banana") {
    xmin = -3; xmax = 3;
    ymin = -1; ymax = 8;
  } else if (type === "bimodal") {
    xmin = -5; xmax = 5;
    ymin = -5; ymax = 5;
  } else {
    xmin = -4; xmax = 4;
    ymin = -4; ymax = 4;
  }
}

function target(x, y) {
  const type = document.getElementById("dist").value;
  if (type === "funnel") return targetFunnel(x, y);
  if (type === "banana") return targetBanana(x, y);
  if (type === "bimodal") return targetBimodal(x, y);
  return targetGaussian(x, y);
}

function targetGaussian(x, y) {
  const rho = 0.8;
  const exponent = (x*x - 2*rho*x*y + y*y) / (2 * (1 - rho*rho));
  return Math.exp(-exponent);
}

function targetBanana(x, y) {
  const exponent = (x*x + 100 * (y - x*x)**2) / 200;
  return Math.exp(-exponent);
}

function targetFunnel(x, y) {
  const ex = Math.exp(x);
  const exponent = x*x / 18 + y*y / (2 * ex * ex);
  return Math.exp(-exponent);
}

function targetBimodal(x, y) {
  const sigma = 0.8;
  const dx1 = x + 2, dy1 = y + 2;
  const mode1 = 0.4 * Math.exp(-(dx1*dx1 + dy1*dy1) / (2 * sigma*sigma));
  const dx2 = x - 2, dy2 = y - 2;
  const mode2 = 0.6 * Math.exp(-(dx2*dx2 + dy2*dy2) / (2 * sigma*sigma));
  return mode1 + mode2;
}

/***************************************************
 * INITIALIZATION
 ***************************************************/
function reset() {
  running = false;
  iteration = 0;
  swapIteration = 0;
  coldSamples = [];
  
  setupTemperatures();
  
  // Initialize chains
  replicas = [];
  swapAccepts = Array(numTemperatures - 1).fill(0);
  swapAttempts = Array(numTemperatures - 1).fill(0);
  
  for (let k = 0; k < numTemperatures; k++) {
    replicas.push({
      x: (Math.random() - 0.5) * (xmax - xmin),
      y: (Math.random() - 0.5) * (ymax - ymin),
      beta: temperatures[k],
      samples: [],
      accepts: 0,
      proposals: 0,
      index: k // Track original temperature index
    });
  }
  
  // Create replica canvases
  replicaGrid.innerHTML = "";
  replicaCanvases = [];
  replicaContexts = [];
  
  for (let k = 0; k < numTemperatures; k++) {
    const container = document.createElement("div");
    container.className = "replica-canvas-container";
    
    const label = document.createElement("div");
    label.className = "replica-label";
    label.innerHTML = `
      <span>Chain ${k + 1}</span>
      <span class="temperature-badge">β = ${temperatures[k].toFixed(3)}</span>
    `;
    
    const canvas = document.createElement("canvas");
    canvas.width = 450;
    canvas.height = 450;
    canvas.style.width = "100%";   // Remove the maxWidth line
    canvas.style.height = "auto";
    
    container.appendChild(label);
    container.appendChild(canvas);
    replicaGrid.appendChild(container);
    
    replicaCanvases.push(canvas);
    replicaContexts.push(canvas.getContext("2d"));
  }
  
  updateStats();
  drawAll();
  
  document.getElementById("stepInfo").textContent = "Ready. Click 'Start Sampling' to begin.";
}

/***************************************************
 * MCMC STEP FOR EACH CHAIN
 ***************************************************/
function mcmcStep() {
  for (let k = 0; k < numTemperatures; k++) {
    const replica = replicas[k];
    const beta = replica.beta;
    
    // Proposal (simple random walk)
    const sigma = 0.3;
    const xprop = replica.x + sigma * (Math.random() - 0.5) * 2;
    const yprop = replica.y + sigma * (Math.random() - 0.5) * 2;
    
    // Compute acceptance with temperature
    const curr_pi = target(replica.x, replica.y);
    const prop_pi = target(xprop, yprop);
    
    const ratio = Math.pow(prop_pi / curr_pi, beta);
    const accept = Math.random() < Math.min(1, ratio);
    
    replica.proposals++;
    if (accept) {
      replica.x = xprop;
      replica.y = yprop;
      replica.accepts++;
    }
    
    // Store sample in this chain's trajectory
    replica.samples.push({x: replica.x, y: replica.y});
  }
  
  // CRITICAL: Always collect from replicas[0] which has beta=1.0
  // After swaps, different states visit this position, but it always has beta=1.0
  coldSamples.push({x: replicas[0].x, y: replicas[0].y});
}

/***************************************************
 * CHAIN SWAPS (State Exchange)
 ***************************************************/
function replicaExchange() {
  // Try swaps between adjacent temperatures
  for (let k = 0; k < numTemperatures - 1; k++) {
    const r1 = replicas[k];
    const r2 = replicas[k + 1];
    
    swapAttempts[k]++;
    
    const pi1 = target(r1.x, r1.y);
    const pi2 = target(r2.x, r2.y);
    
    // Swap acceptance probability
    const deltaL = Math.log(Math.max(pi1, 1e-300)) - Math.log(Math.max(pi2, 1e-300));
    const deltaBeta = r1.beta - r2.beta;
    const logAccept = deltaBeta * deltaL;
    
    if (Math.random() < Math.exp(logAccept)) {
      // Accept swap - exchange STATE (positions) but NOT temperatures
      // The temperatures stay with their array positions
      const tmpX = r1.x, tmpY = r1.y;
      r1.x = r2.x;
      r1.y = r2.y;
      r2.x = tmpX;
      r2.y = tmpY;
      
      swapAccepts[k]++;
    }
  }
  
  swapIteration++;
}

/***************************************************
 * SINGLE STEP
 ***************************************************/
function singleStep() {
  mcmcStep();
  iteration++;
  
  // Check if it's time for replica exchange
  if (iteration % swapFrequency === 0) {
    replicaExchange();
  }
  
  updateStats();
  drawAll();
}

/***************************************************
 * CONTINUOUS SAMPLING
 ***************************************************/
async function start() {
  if (running) {
    running = false;
    return;
  }
  
  running = true;
  const button = event.target;
  button.textContent = "⏸ Pause";
  
  while (running) {
    singleStep();
    await new Promise(resolve => setTimeout(resolve, animDelay));
  }
  
  button.textContent = "▶ Start Sampling";
}

/***************************************************
 * DRAWING
 ***************************************************/
function drawAll() {
  drawReplicas();
  drawSwapAcceptance();
  drawTracePlot();
  drawBothCoordinates();
  drawColdChain();
}

function drawReplicas() {
  for (let k = 0; k < numTemperatures; k++) {
    const ctx = replicaContexts[k];
    const canvas = replicaCanvases[k];
    const replica = replicas[k];
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw contours (lighter for hot chains)
    drawContour(ctx, canvas, replica.beta);
    
    // Draw trajectory
    const samples = replica.samples;
    if (samples.length > 1) {
      ctx.strokeStyle = `rgba(59, 130, 246, ${0.3 + 0.4 * replica.beta})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < samples.length; i++) {
        const [cx, cy] = toCanvasReplica(samples[i].x, samples[i].y, canvas);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }
    
    // Draw current position
    const [cx, cy] = toCanvasReplica(replica.x, replica.y, canvas);
    ctx.fillStyle = replica.beta === 1.0 ? "#dc2626" : "#3b82f6";
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw acceptance rate
    const accRate = replica.proposals > 0 ? (replica.accepts / replica.proposals * 100).toFixed(1) : 0;
    ctx.fillStyle = "#2d3748";
    ctx.font = "11px monospace";
    ctx.fillText(`Acc: ${accRate}%`, 5, canvas.height - 5);
  }
}

function drawContour(ctx, canvas, beta) {
  const w = canvas.width;
  const h = canvas.height;
  const imageData = ctx.createImageData(w, h);
  const data = imageData.data;
  
  let maxZ = 0;
  const zValues = [];
  
  // Compute all z values
  for (let j = 0; j < h; j++) {
    zValues[j] = [];
    for (let i = 0; i < w; i++) {
      const x = xmin + (i / w) * (xmax - xmin);
      const y = ymax - (j / h) * (ymax - ymin);
      const z = Math.pow(target(x, y), beta);
      zValues[j][i] = z;
      if (z > maxZ) maxZ = z;
    }
  }
  
  // Fill pixels
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const idx = (j * w + i) * 4;
      const normalized = maxZ > 0 ? zValues[j][i] / maxZ : 0;
      const intensity = Math.floor(255 * (1 - normalized * 0.7));
      data[idx] = intensity;
      data[idx + 1] = intensity;
      data[idx + 2] = intensity;
      data[idx + 3] = 255;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
}

function toCanvasReplica(x, y, canvas) {
  const cx = (x - xmin) / (xmax - xmin) * canvas.width;
  const cy = (ymax - y) / (ymax - ymin) * canvas.height;
  return [cx, cy];
}

function drawSwapAcceptance() {
  const ctx = swapAcceptanceCtx;
  const canvas = swapAcceptanceCanvas;
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;
  
  ctx.clearRect(0, 0, w, h);
  
  if (numTemperatures < 2) return;
  
  const margin = {top: 20, right: 20, bottom: 40, left: 50};
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  
  // Draw grid
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + i / 5 * plotH;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
  }
  
  // Draw target range (20-40%)
  ctx.fillStyle = "rgba(72, 187, 120, 0.1)";
  const y20 = margin.top + (1 - 0.2) * plotH;
  const y40 = margin.top + (1 - 0.4) * plotH;
  ctx.fillRect(margin.left, y40, plotW, y20 - y40);
  
  // Draw bars
  const barWidth = plotW / (numTemperatures - 1) * 0.6;
  for (let k = 0; k < numTemperatures - 1; k++) {
    const rate = swapAttempts[k] > 0 ? swapAccepts[k] / swapAttempts[k] : 0;
    const x = margin.left + (k / (numTemperatures - 1)) * plotW;
    const barH = rate * plotH;
    const y = margin.top + plotH - barH;
    
    // Color based on rate
    if (rate < 0.1 || rate > 0.5) {
      ctx.fillStyle = "#f56565";
    } else if (rate >= 0.2 && rate <= 0.4) {
      ctx.fillStyle = "#48bb78";
    } else {
      ctx.fillStyle = "#ed8936";
    }
    
    ctx.fillRect(x - barWidth / 2, y, barWidth, barH);
  }
  
  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();
  
  // Labels
  ctx.fillStyle = "#2d3748";
  ctx.font = "12px -apple-system, sans-serif";
  ctx.textAlign = "center";
  for (let k = 0; k < numTemperatures - 1; k++) {
    const x = margin.left + (k / (numTemperatures - 1)) * plotW;
    ctx.fillText(`${k + 1}-${k + 2}`, x, h - 15);
  }
  
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + (1 - i / 5) * plotH;
    ctx.fillText(`${(i / 5 * 100).toFixed(0)}%`, margin.left - 5, y + 4);
  }
  
  ctx.font = "bold 12px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Temperature Pair", w / 2, h - 2);
}

function drawTracePlot() {
  const ctx = tracePlotCtx;
  const canvas = tracePlotCanvas;
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (coldSamples.length < 2) return;
  
  const margin = {top: 20, right: 20, bottom: 40, left: 50};
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  
  // Find y range (θ₁ values)
  const yValues = coldSamples.map(s => s.x);
  const yMin = Math.min(...yValues, ymin);
  const yMax = Math.max(...yValues, ymax);
  
  // Draw grid
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + i / 5 * plotH;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
  }
  
  // Draw trace
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  
  for (let i = 0; i < coldSamples.length; i++) {
    const x = margin.left + (i / (coldSamples.length - 1)) * plotW;
    const y = margin.top + (yMax - coldSamples[i].x) / (yMax - yMin) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  
  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();
  
  // Labels
  ctx.fillStyle = "#2d3748";
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Iteration", w / 2, h - 5);
  
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + i / 5 * plotH;
    const val = yMax - i / 5 * (yMax - yMin);
    ctx.fillText(val.toFixed(1), margin.left - 5, y + 3);
  }
  
  ctx.save();
  ctx.translate(12, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.font = "bold 11px -apple-system, sans-serif";
  ctx.fillText("θ₁", 0, 0);
  ctx.restore();
}

function drawBothCoordinates() {
  const ctx = bothCoordinatesCtx;
  const canvas = bothCoordinatesCanvas;
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (coldSamples.length < 2) return;
  
  const margin = {top: 30, right: 80, bottom: 50, left: 60};
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  
  // Find ranges
  const xValues = coldSamples.map(s => s.x);
  const yValues = coldSamples.map(s => s.y);
  const dataYMin = Math.min(...yValues, ymin);
  const dataYMax = Math.max(...yValues, ymax);
  
  // Draw grid
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + i / 5 * plotH;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
  }
  
  // Draw horizontal line at 0 if in range
  if (dataYMin < 0 && dataYMax > 0) {
    const zeroY = margin.top + (dataYMax - 0) / (dataYMax - dataYMin) * plotH;
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(margin.left, zeroY);
    ctx.lineTo(margin.left + plotW, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  
  // Draw θ₁ trace (blue)
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  for (let i = 0; i < coldSamples.length; i++) {
    const x = margin.left + (i / (coldSamples.length - 1)) * plotW;
    const y = margin.top + (dataYMax - coldSamples[i].x) / (dataYMax - dataYMin) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  
  // Draw θ₂ trace (orange)
  ctx.strokeStyle = "#f97316";
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  for (let i = 0; i < coldSamples.length; i++) {
    const x = margin.left + (i / (coldSamples.length - 1)) * plotW;
    const y = margin.top + (dataYMax - coldSamples[i].y) / (dataYMax - dataYMin) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  
  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();
  
  // X-axis labels
  ctx.fillStyle = "#2d3748";
  ctx.font = "12px -apple-system, sans-serif";
  ctx.textAlign = "center";
  
  for (let i = 0; i <= 5; i++) {
    const x = margin.left + i / 5 * plotW;
    const iter = Math.round(i / 5 * (coldSamples.length - 1));
    ctx.fillText(iter, x, h - margin.bottom + 20);
  }
  
  ctx.font = "bold 13px -apple-system, sans-serif";
  ctx.fillText("Iteration", w / 2, h - 10);
  
  // Y-axis labels
  ctx.textAlign = "right";
  ctx.font = "12px -apple-system, sans-serif";
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + i / 5 * plotH;
    const val = dataYMax - i / 5 * (dataYMax - dataYMin);
    ctx.fillText(val.toFixed(1), margin.left - 8, y + 4);
  }
  
  ctx.save();
  ctx.translate(15, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.font = "bold 13px -apple-system, sans-serif";
  ctx.fillText("Position", 0, 0);
  ctx.restore();
  
  // Legend
  ctx.textAlign = "left";
  ctx.font = "12px -apple-system, sans-serif";
  
  ctx.fillStyle = "#3b82f6";
  ctx.fillRect(w - margin.right + 10, margin.top + 10, 15, 3);
  ctx.fillStyle = "#2d3748";
  ctx.fillText("θ₁", w - margin.right + 30, margin.top + 14);
  
  ctx.fillStyle = "#f97316";
  ctx.fillRect(w - margin.right + 10, margin.top + 30, 15, 3);
  ctx.fillStyle = "#2d3748";
  ctx.fillText("θ₂", w - margin.right + 30, margin.top + 34);
}


function drawColdChain() {
  const ctx = coldChainCtx;
  const canvas = coldChainCanvas;
  
  // Get visual dimensions (what the user sees)
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw background contour with proper normalization
  drawColdChainBackground(ctx, w, h);
  
  // Draw samples
  if (coldSamples.length > 0) {
    ctx.fillStyle = "rgba(220, 38, 38, 0.6)";
    for (const sample of coldSamples) {
      const cx = (sample.x - xmin) / (xmax - xmin) * w;
      const cy = (ymax - sample.y) / (ymax - ymin) * h;
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function drawColdChainBackground(ctx, w, h) {
  // First pass: find max Z for normalization
  let maxZ = 0;
  const step = 2;
  for (let j = 0; j < h; j += step) {
    for (let i = 0; i < w; i += step) {
      const x = xmin + (i / w) * (xmax - xmin);
      const y = ymax - (j / h) * (ymax - ymin);
      const z = target(x, y);
      if (z > maxZ) maxZ = z;
    }
  }
  
  // Second pass: draw with normalized colors
  for (let j = 0; j < h; j += step) {
    for (let i = 0; i < w; i += step) {
      const x = xmin + (i / w) * (xmax - xmin);
      const y = ymax - (j / h) * (ymax - ymin);
      const z = target(x, y);
      
      const normalized = maxZ > 0 ? z / maxZ : 0;
      const intensity = Math.floor(255 * (1 - normalized * 0.7));
      ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
      ctx.fillRect(i, j, step, step);
    }
  }
}

/***************************************************
 * STATS UPDATE
 ***************************************************/
function updateStats() {
  document.getElementById("totalIter").textContent = iteration.toLocaleString();
  document.getElementById("coldSamples").textContent = coldSamples.length.toLocaleString();
  
  const totalSwapAttempts = swapAttempts.reduce((a, b) => a + b, 0);
  const totalSwapAccepts = swapAccepts.reduce((a, b) => a + b, 0);
  
  document.getElementById("swapAttempts").textContent = totalSwapAttempts.toLocaleString();
  document.getElementById("swapAccepts").textContent = totalSwapAccepts.toLocaleString();
  
  if (iteration % swapFrequency === 0 && iteration > 0) {
    document.getElementById("stepInfo").textContent = 
      `Iteration ${iteration}: Performed replica exchange. Total swaps: ${totalSwapAccepts}/${totalSwapAttempts}`;
  } else {
    document.getElementById("stepInfo").textContent = 
      `Iteration ${iteration}: MCMC step. Next swap at iteration ${Math.ceil((iteration + 1) / swapFrequency) * swapFrequency}`;
  }
}

/***************************************************
 * INITIALIZATION
 ***************************************************/
setDomain();
reset();

