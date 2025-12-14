/**
 * NESTED_SAMPLING - Algorithm-specific code
 * Shared utilities (randn, autocorr, calculateESS, drawTracePlot, drawACFXY) are in shared-utils.js
 */

/***************************************************
 * CANVAS & UI SETUP
 ***************************************************/
const canvas = document.getElementById("posterior");
const ctx = canvas.getContext("2d");
const hxCanvas = document.getElementById("histX");
const hyCanvas = document.getElementById("histY");
const hx = hxCanvas.getContext("2d");
const hy = hyCanvas.getContext("2d");
const acfCanvas = document.getElementById("acfXY");
const acfXY = acfCanvas.getContext("2d");
const traceXCanvas = document.getElementById("traceX");
const traceYCanvas = document.getElementById("traceY");
const traceX = traceXCanvas.getContext("2d");
const traceY = traceYCanvas.getContext("2d");
const evidencePlotCanvas = document.getElementById("evidencePlot");
const evidencePlot = evidencePlotCanvas.getContext("2d");

// Set canvas widths to actual container width
acfCanvas.width = acfCanvas.offsetWidth;
traceXCanvas.width = traceXCanvas.offsetWidth;
traceYCanvas.width = traceYCanvas.offsetWidth;
evidencePlotCanvas.width = evidencePlotCanvas.offsetWidth;

const stepInfo = document.getElementById("stepInfo");
const nliveSlider = document.getElementById("nlive");
const speedSlider = document.getElementById("speed");
const nliveVal = document.getElementById("nliveVal");
const speedVal = document.getElementById("speedVal");

nliveSlider.oninput = () => {
  nliveVal.textContent = nliveSlider.value;
};

speedSlider.oninput = () => {
  speedVal.textContent = speedSlider.value;
  if (timer) {
    clearInterval(timer);
    timer = setInterval(nestedSamplingStep, parseInt(speedSlider.value));
  }
};

/***************************************************
 * DOMAIN & STATE
 ***************************************************/
// Domain will be set based on distribution
let xmin, xmax, ymin, ymax;

function setDomain() {
  const type = document.getElementById("dist").value;
  if (type === "funnel") {
    // Neal's funnel: x ~ N(0, 3^2), y|x ~ N(0, exp(x)^2)
    // x covers ±3 std devs = ±9, use ±10 to be safe
    // For y: when x is large, exp(x) is huge. At x=5, exp(5)≈150
    // So y needs range of about ±3*150 = ±450, but let's use ±50 for visualization
    xmin = -10; xmax = 10;
    ymin = -50; ymax = 50;
  } else if (type === "banana") {
    // Rosenbrock banana: x² + 100(y - x²)² ≤ C
    // For 95% probability mass, C ≈ 200 * 5.99 ≈ 1200
    // So x² ≤ 1200 → x ∈ [-35, 35], but main mass is closer
    // The banana curves upward: when x=±2, y ≈ x² = 4
    // Most mass is in x ∈ [-2.5, 2.5], y ∈ [-1, 7]
    xmin = -3; xmax = 3;
    ymin = -1; ymax = 8;
  } else if (type === "bimodal") {
    // Bimodal: two modes at (-2, -2) and (+2, +2)
    // Each mode has width σ=0.8, so ±3σ ≈ 2.4 around each mode
    // Cover both modes: [-2-3, 2+3] = [-5, 5]
    xmin = -5; xmax = 5;
    ymin = -5; ymax = 5;
  } else {
    // Gaussian: bivariate with ρ=0.8, unit variance
    // ±4 standard deviations covers 99.99%
    xmin = -4; xmax = 4;
    ymin = -4; ymax = 4;
  }
}

// Initialize domain
setDomain();

const samplesX = [];
const samplesY = [];
const samplesL = []; // Likelihoods
const samplesX_prior = []; // Prior volumes

// Nested sampling state
let livePoints = []; // Array of {x, y, L}
let deadPoints = []; // Array of {x, y, L, X}
let logZ = -1e100; // Log evidence
let H = 0; // Information (Bayesian model complexity)
let iteration = 0;
let timer = null;

/***************************************************
 * TARGET DISTRIBUTIONS
 ***************************************************/
function target(x, y) {
  const type = document.getElementById("dist").value;
  if (type === "funnel") return targetFunnel(x, y);
  if (type === "banana") return targetBanana(x, y);
  if (type === "bimodal") return targetBimodal(x, y);
  return targetGaussian(x, y);
}

function targetGaussian(x, y) {
  // Bivariate Gaussian with correlation rho = 0.8
  const rho = 0.8;
  const exponent = (x*x - 2*rho*x*y + y*y) / (2 * (1 - rho*rho));
  return Math.exp(-exponent);
}

function targetBanana(x, y) {
  // Rosenbrock's banana distribution
  // π(x,y) ∝ exp(-1/200 * (x² + 100(y - x²)²))
  const exponent = (x*x + 100 * (y - x*x)**2) / 200;
  return Math.exp(-exponent);
}

function targetFunnel(x, y) {
  // Neal's funnel distribution
  // x ~ N(0, 3²), y|x ~ N(0, exp(x)²)
  // π(x,y) ∝ exp(-x²/18 - y²/(2*exp(2x)))
  const ex = Math.exp(x);
  const exponent = x*x / 18 + y*y / (2 * ex * ex);
  return Math.exp(-exponent);
}

function targetBimodal(x, y) {
  // Bimodal Gaussian mixture - two well-separated modes
  // Mode 1: centered at (-2, -2)
  // Mode 2: centered at (+2, +2)
  // This tests nested sampling's ability to find multiple modes
  
  const sigma = 0.8; // Width of each mode
  
  // Mode 1: weight = 0.4, center = (-2, -2)
  const dx1 = x + 2;
  const dy1 = y + 2;
  const mode1 = 0.4 * Math.exp(-(dx1*dx1 + dy1*dy1) / (2 * sigma*sigma));
  
  // Mode 2: weight = 0.6, center = (+2, +2)  
  const dx2 = x - 2;
  const dy2 = y - 2;
  const mode2 = 0.6 * Math.exp(-(dx2*dx2 + dy2*dy2) / (2 * sigma*sigma));
  
  return mode1 + mode2;
}


/***************************************************
 * COORDINATE TRANSFORMATION
 ***************************************************/
function toCanvas(x, y) {
  const px = (x - xmin) / (xmax - xmin) * canvas.width;
  const py = canvas.height - (y - ymin) / (ymax - ymin) * canvas.height;
  return [px, py];
}

/***************************************************
 * DRAW DENSITY WITH PROPER AXES
 ***************************************************/
function drawDensity() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Compute density grid
  const n = 200;
  const density = [];
  let maxP = 0;
  
  for (let i = 0; i < n; i++) {
    density[i] = [];
    for (let j = 0; j < n; j++) {
      const xVal = xmin + (xmax - xmin) * i / n;
      const yVal = ymin + (ymax - ymin) * j / n;
      const pVal = target(xVal, yVal);
      density[i][j] = pVal;
      if (pVal > maxP) maxP = pVal;
    }
  }

  // Draw density heatmap
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const alpha = Math.min(0.9, Math.max(0.05, 8 * density[i][j] / maxP));
      ctx.fillStyle = `rgba(17, 24, 39, ${alpha})`;
      ctx.fillRect(
        i * canvas.width / n,
        canvas.height - j * canvas.height / n,
        canvas.width / n + 1,
        canvas.height / n + 1
      );
    }
  }

  // Grid lines
  ctx.strokeStyle = "#e0e0e0";
  ctx.lineWidth = 0.5;
  const gridN = 8;
  for (let i = 0; i <= gridN; i++) {
    const pos = i / gridN * canvas.width;
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, canvas.height);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(canvas.width, pos);
    ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height);
  ctx.lineTo(canvas.width, canvas.height);
  ctx.moveTo(0, 0);
  ctx.lineTo(0, canvas.height);
  ctx.stroke();

  // Ticks and labels
  ctx.fillStyle = "#2d3748";
  ctx.font = "13px -apple-system, sans-serif";
  ctx.textAlign = "center";
  
  const nTicks = 8;
  for (let i = 0; i <= nTicks; i++) {
    const xPos = i / nTicks * canvas.width;
    const yPos = canvas.height - i / nTicks * canvas.height;
    const xLabel = (xmin + (xmax - xmin) * i / nTicks).toFixed(1);
    const yLabel = (ymin + (ymax - ymin) * i / nTicks).toFixed(1);
    
    // X-axis ticks
    ctx.beginPath();
    ctx.moveTo(xPos, canvas.height);
    ctx.lineTo(xPos, canvas.height - 8);
    ctx.stroke();
    ctx.fillText(xLabel, xPos, canvas.height + 20);
    
    // Y-axis ticks
    ctx.textAlign = "right";
    ctx.beginPath();
    ctx.moveTo(0, yPos);
    ctx.lineTo(8, yPos);
    ctx.stroke();
    ctx.fillText(yLabel, -12, yPos + 4);
    ctx.textAlign = "center";
  }

  // Axis labels
  ctx.font = "bold 16px -apple-system, sans-serif";
  ctx.fillStyle = "#1a1a1a";
  ctx.fillText("x₁", canvas.width / 2, canvas.height + 45);
  
  ctx.save();
  ctx.translate(15, canvas.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("x₂", 0, 0);
  ctx.restore();
}

/***************************************************
 * NESTED SAMPLING STEP
 ***************************************************/
function nestedSamplingStep() {
  if (livePoints.length === 0) {
    console.log("No live points");
    return;
  }
  
  // Find point with lowest likelihood
  let minIdx = 0;
  let minL = livePoints[0].L;
  for (let i = 1; i < livePoints.length; i++) {
    if (livePoints[i].L < minL) {
      minL = livePoints[i].L;
      minIdx = i;
    }
  }
  
  const worstPoint = livePoints[minIdx];
  
  // Estimate prior volume shrinkage
  const N = livePoints.length;
  const t = Math.exp(-1.0 / N);
  const X_prev = iteration === 0 ? 1.0 : deadPoints[deadPoints.length - 1].X;
  const X = X_prev * t;
  
  // Save as dead point
  deadPoints.push({
    x: worstPoint.x,
    y: worstPoint.y,
    L: worstPoint.L,
    X: X
  });
  
  // Update evidence using log-sum-exp for numerical stability
  const deltaX = iteration === 0 ? (1.0 - X) : (deadPoints[deadPoints.length - 2].X - X);
  const logLikelihood = Math.log(Math.max(worstPoint.L, 1e-300));
  const logDeltaX = Math.log(Math.max(deltaX, 1e-300));
  const logWeight = logLikelihood + logDeltaX;
  
  if (iteration === 0) {
    logZ = logWeight;
  } else {
    // Log-sum-exp: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    const maxLog = Math.max(logZ, logWeight);
    logZ = maxLog + Math.log(Math.exp(logZ - maxLog) + Math.exp(logWeight - maxLog));
  }
  
  // Information (Bayesian complexity)
  const weight = worstPoint.L * deltaX;
  const Z = Math.exp(logZ);
  if (Z > 0) {
    H += weight * logLikelihood / Z - weight * Math.log(weight) / Z;
  }
  
  iteration++;
  
  // Generate new live point with L > minL using rejection sampling
  let newPoint = null;
  let attempts = 0;
  const maxAttempts = 10000;
  
  while (attempts < maxAttempts && !newPoint) {
    // Sample from prior
    const nx = xmin + Math.random() * (xmax - xmin);
    const ny = ymin + Math.random() * (ymax - ymin);
    const nL = target(nx, ny);
    
    if (nL > minL) {
      newPoint = {x: nx, y: ny, L: nL};
    }
    attempts++;
  }
  
  if (!newPoint) {
    // Couldn't find new point - algorithm has converged
    console.log("Nested sampling converged - couldn't find point above threshold");
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    updateStepInfo(worstPoint.x, worstPoint.y, minL, X, logZ, true);
    updateStats();
    return;
  }
  
  // Replace worst point with new point
  livePoints[minIdx] = newPoint;
  
  // Redraw everything
  drawDensity();
  
  // Draw all dead points
  deadPoints.forEach(pt => {
    const [dpx, dpy] = toCanvas(pt.x, pt.y);
    ctx.fillStyle = "rgba(220, 38, 38, 0.5)";
    ctx.beginPath();
    ctx.arc(dpx, dpy, 2, 0, 2 * Math.PI);
    ctx.fill();
  });
  
  // Draw live points
  livePoints.forEach(pt => {
    const [lpx, lpy] = toCanvas(pt.x, pt.y);
    ctx.fillStyle = "#3b82f6";
    ctx.beginPath();
    ctx.arc(lpx, lpy, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  
  // Store samples for posterior
  samplesX.push(worstPoint.x);
  samplesY.push(worstPoint.y);
  samplesL.push(worstPoint.L);
  samplesX_prior.push(X);
  
  // Update displays
  if (deadPoints.length > 1) {
    drawHistograms();
    drawNestedSamplingDiagnostics();
  }
  updateStepInfo(worstPoint.x, worstPoint.y, minL, X, logZ, false);
  updateStats();
  
  // Check termination criterion
  const maxL = Math.max(...livePoints.map(p => p.L));
  const remainingZ = maxL * X;
  const currentZ = Math.exp(logZ);
  if (remainingZ < 0.01 * currentZ && iteration > 50) {
    console.log("Nested sampling converged: remaining evidence negligible");
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }
}

/***************************************************
 * UPDATE STEP INFO
 ***************************************************/
function updateStepInfo(x, y, L, X, logZ, converged) {
  const statusClass = converged ? 'status-rejected' : 'status-accepted';
  const statusText = converged ? 'CONVERGED' : 'SAMPLING';
  
  stepInfo.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
      <div>
        <strong>Discarded Point (Lowest L)</strong><br>
        θ₁ = ${x.toFixed(4)}<br>
        θ₂ = ${y.toFixed(4)}<br>
        L(θ) = ${L.toExponential(4)}
      </div>
      <div>
        <strong>Nested Sampling Progress</strong><br>
        Iteration: ${iteration}<br>
        Prior Volume: X = ${X.toExponential(4)}<br>
        Live Points: ${livePoints.length}
      </div>
    </div>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
      <strong>Evidence Computation</strong><br>
      log(Z) = ${logZ.toFixed(4)}<br>
      Z = ${Math.exp(logZ).toExponential(4)}<br>
      Information H = ${H.toFixed(4)} nats<br>
      <div style="margin-top: 0.5rem;">
        <span style="font-size: 1.1rem; margin-top: 0.5rem; display: inline-block;" class="${statusClass}">
          ${statusText}
        </span>
      </div>
    </div>
  `;
}

/***************************************************
 * UPDATE STATISTICS
 ***************************************************/
function updateStats() {
  document.getElementById("totalIter").textContent = iteration.toLocaleString();
  document.getElementById("acceptCount").textContent = deadPoints.length.toLocaleString();
  
  document.getElementById("acceptRate").textContent = 
    "log(Z) = " + logZ.toFixed(2);
  
  // Estimate ESS from importance weights
  if (samplesL.length > 10 && deadPoints.length > 0) {
    try {
      // Compute posterior weights
      const Z = Math.exp(logZ);
      if (Z > 0 && isFinite(Z)) {
        const weights = [];
        for (let i = 0; i < samplesL.length; i++) {
          const X_prev = i === 0 ? 1.0 : samplesX_prior[i-1];
          const X_curr = samplesX_prior[i];
          const X_next = i < samplesX_prior.length - 1 ? samplesX_prior[i+1] : X_curr * Math.exp(-1/livePoints.length);
          const deltaX = X_prev - X_next;
          const w = samplesL[i] * deltaX / Z;
          if (isFinite(w) && w > 0) {
            weights.push(w);
          }
        }
        
        if (weights.length > 0) {
          // ESS from weights: 1 / sum(w_i^2)
          const sumW2 = weights.reduce((sum, w) => sum + w*w, 0);
          const ess = sumW2 > 0 ? 1.0 / sumW2 : 0;
          document.getElementById("essValue").textContent = Math.round(ess).toLocaleString();
        }
      }
    } catch (e) {
      console.error("Error computing ESS:", e);
      document.getElementById("essValue").textContent = "—";
    }
  }
}

/***************************************************
 * HISTOGRAMS
 ***************************************************/
function drawHistograms() {
  drawHistogram(hx, samplesX, "x₁", hxCanvas);
  drawHistogram(hy, samplesY, "x₂", hyCanvas);
  drawPosteriorWeights(acfXY, deadPoints);
  // Note: Nested sampling diagnostics are drawn separately via drawNestedSamplingDiagnostics()
}

function drawHistogram(ctx, data, label, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (data.length === 0 || deadPoints.length === 0 || !livePoints || livePoints.length === 0) return;

  const bins = 40;
  const hist = Array(bins).fill(0);
  const binWidth = (xmax - xmin) / bins;

  // Compute posterior weights for importance weighting
  const Z = Math.exp(logZ);
  if (!isFinite(Z) || Z <= 0) return; // Can't compute weights without valid evidence
  
  const weights = [];
  
  for (let i = 0; i < deadPoints.length; i++) {
    const X_prev = i === 0 ? 1.0 : deadPoints[i-1].X;
    const X_curr = deadPoints[i].X;
    const X_next = i < deadPoints.length - 1 ? deadPoints[i+1].X : X_curr * Math.exp(-1/livePoints.length);
    const deltaX = X_prev - X_next;
    const weight = (deadPoints[i].L * deltaX) / Z;
    if (isFinite(weight) && weight >= 0) {
      weights.push(weight);
    } else {
      weights.push(0);
    }
  }

  // Fill histogram with WEIGHTED samples
  data.forEach((v, idx) => {
    if (idx < weights.length) {
      const i = Math.floor((v - xmin) / (xmax - xmin) * bins);
      if (i >= 0 && i < bins) {
        hist[i] += weights[idx]; // Add weight instead of count
      }
    }
  });

  const hmax = Math.max(...hist, 1e-10);
  
  // Normalize histogram to approximate density
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  const normalized = hist.map(h => h / (totalWeight * binWidth));
  const ymax = Math.max(...normalized, 0.01);

  const margin = {top: 20, right: 20, bottom: 50, left: 55};
  const plotWidth = canvas.width - margin.left - margin.right;
  const plotHeight = canvas.height - margin.top - margin.bottom;

  // Draw bars
  hist.forEach((h, i) => {
    const x = margin.left + i * plotWidth / bins;
    const barHeight = (h / hmax) * plotHeight;
    ctx.fillStyle = "#374151";
    ctx.fillRect(x, margin.top + plotHeight - barHeight, plotWidth / bins - 1, barHeight);
  });

  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.stroke();

  // X-axis ticks
  ctx.fillStyle = "#2d3748";
  ctx.font = "12px -apple-system, sans-serif";
  ctx.textAlign = "center";
  const xTicks = 8;
  for (let i = 0; i <= xTicks; i++) {
    const xPos = margin.left + i / xTicks * plotWidth;
    const xLabel = (xmin + (xmax - xmin) * i / xTicks).toFixed(1);
    ctx.beginPath();
    ctx.moveTo(xPos, margin.top + plotHeight);
    ctx.lineTo(xPos, margin.top + plotHeight + 5);
    ctx.stroke();
    ctx.fillText(xLabel, xPos, margin.top + plotHeight + 20);
  }

  // Y-axis ticks
  ctx.textAlign = "right";
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const yPos = margin.top + plotHeight - i / yTicks * plotHeight;
    const yLabel = (ymax * i / yTicks).toFixed(2);
    ctx.beginPath();
    ctx.moveTo(margin.left - 5, yPos);
    ctx.lineTo(margin.left, yPos);
    ctx.stroke();
    ctx.fillText(yLabel, margin.left - 10, yPos + 4);
  }

  // Labels
  ctx.font = "bold 14px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, margin.left + plotWidth / 2, canvas.height - 10);
  
  ctx.save();
  ctx.translate(15, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Density", 0, 0);
  ctx.restore();
  
  // Add note about weighting
  ctx.font = "11px -apple-system, sans-serif";
  ctx.fillStyle = "#6b7280";
  ctx.textAlign = "right";
  ctx.fillText("(importance weighted)", canvas.width - 5, 15);
}

/***************************************************
 * TRACE PLOTS
 ***************************************************/
/***************************************************
 * NESTED SAMPLING DIAGNOSTIC PLOTS
 ***************************************************/
function drawNestedSamplingDiagnostics() {
  if (deadPoints.length < 2) return;
  
  // Extract data
  const iterations = deadPoints.map((_, i) => i);
  const logX = deadPoints.map(pt => Math.log(pt.X));
  const logL = deadPoints.map(pt => Math.log(Math.max(pt.L, 1e-300)));
  
  // History of logZ
  const logZHistory = [];
  let runningLogZ = -1e100;
  for (let i = 0; i < deadPoints.length; i++) {
    const deltaX = i === 0 ? (1.0 - deadPoints[i].X) : (deadPoints[i-1].X - deadPoints[i].X);
    const logWeight = Math.log(deadPoints[i].L) + Math.log(deltaX);
    
    if (i === 0) {
      runningLogZ = logWeight;
    } else {
      const maxLog = Math.max(runningLogZ, logWeight);
      runningLogZ = maxLog + Math.log(Math.exp(runningLogZ - maxLog) + Math.exp(logWeight - maxLog));
    }
    logZHistory.push(runningLogZ);
  }
  
  // Draw log(X) vs iteration
  drawDiagnosticPlot(traceX, iterations, logX, "Iteration", "log(X)", "#1e40af", traceXCanvas);
  
  // Draw log(L) vs iteration  
  drawDiagnosticPlot(traceY, iterations, logL, "Iteration", "log(L)", "#dc2626", traceYCanvas);
  
  // Draw log(Z) vs iteration
  drawDiagnosticPlot(evidencePlot, iterations, logZHistory, "Iteration", "log(Z)", "#059669", evidencePlotCanvas);
}

function drawDiagnosticPlot(ctx, xData, yData, xLabel, yLabel, color, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (xData.length < 2) return;

  const margin = {top: 20, right: 50, bottom: 50, left: 70};
  const plotWidth = canvas.width - margin.left - margin.right;
  const plotHeight = canvas.height - margin.top - margin.bottom;

  // Find data range
  const xMin = Math.min(...xData);
  const xMax = Math.max(...xData);
  const yMin = Math.min(...yData);
  const yMax = Math.max(...yData);
  const yRange = yMax - yMin;
  const yPadding = yRange * 0.1;

  // Grid
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  const xTicks = 10;
  const yTicks = 6;
  
  for (let i = 0; i <= xTicks; i++) {
    const x = margin.left + i / xTicks * plotWidth;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotHeight);
    ctx.stroke();
  }
  
  for (let i = 0; i <= yTicks; i++) {
    const y = margin.top + i / yTicks * plotHeight;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
  }

  // Draw line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  for (let i = 0; i < xData.length; i++) {
    const xPos = margin.left + (xData[i] - xMin) / (xMax - xMin || 1) * plotWidth;
    const yPos = margin.top + plotHeight - (yData[i] - (yMin - yPadding)) / (yMax - yMin + 2 * yPadding || 1) * plotHeight;
    
    if (i === 0) ctx.moveTo(xPos, yPos);
    else ctx.lineTo(xPos, yPos);
  }
  ctx.stroke();

  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.stroke();

  // X-axis ticks and labels
  ctx.fillStyle = "#2d3748";
  ctx.font = "12px -apple-system, sans-serif";
  ctx.textAlign = "center";
  
  for (let i = 0; i <= xTicks; i++) {
    const x = margin.left + i / xTicks * plotWidth;
    const val = xMin + i / xTicks * (xMax - xMin);
    ctx.beginPath();
    ctx.moveTo(x, margin.top + plotHeight);
    ctx.lineTo(x, margin.top + plotHeight + 6);
    ctx.stroke();
    ctx.fillText(Math.round(val), x, margin.top + plotHeight + 20);
  }
  
  ctx.font = "bold 13px -apple-system, sans-serif";
  ctx.fillText(xLabel, margin.left + plotWidth / 2, canvas.height - 10);

  // Y-axis ticks and labels
  ctx.textAlign = "right";
  ctx.font = "12px -apple-system, sans-serif";
  
  for (let i = 0; i <= yTicks; i++) {
    const y = margin.top + plotHeight - i / yTicks * plotHeight;
    const val = (yMin - yPadding) + i / yTicks * (yMax - yMin + 2 * yPadding);
    ctx.beginPath();
    ctx.moveTo(margin.left - 6, y);
    ctx.lineTo(margin.left, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), margin.left - 10, y + 4);
  }
  
  ctx.font = "bold 13px -apple-system, sans-serif";
  ctx.save();
  ctx.translate(20, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

/***************************************************
 * AUTOCORRELATION
 ***************************************************/
function autocorr(data, lagMax = 100) {
  const n = data.length;
  if (n === 0) return [];
  
  const mean = data.reduce((a, b) => a + b, 0) / n;
  const totalVar = data.reduce((a, b) => a + (b - mean)**2, 0);
  
  if (totalVar === 0) return Array(lagMax + 1).fill(0);
  
  const acf = [];
  for (let lag = 0; lag <= lagMax; lag++) {
    let c = 0;
    for (let i = 0; i < n - lag; i++) {
      c += (data[i] - mean) * (data[i + lag] - mean);
    }
    acf.push(c / totalVar);
  }
  return acf;
}

function drawPosteriorWeights(ctx, deadPts) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
  if (!deadPts || deadPts.length < 2 || !livePoints || livePoints.length === 0) return;

  // Calculate posterior weights
  const Z = Math.exp(logZ);
  if (!isFinite(Z) || Z <= 0) return;
  
  const weights = [];
  const iterations = [];
  
  for (let i = 0; i < deadPts.length; i++) {
    const X_prev = i === 0 ? 1.0 : deadPts[i-1].X;
    const X_curr = deadPts[i].X;
    const X_next = i < deadPts.length - 1 ? deadPts[i+1].X : X_curr * Math.exp(-1/livePoints.length);
    const deltaX = X_prev - X_next;
    const weight = (deadPts[i].L * deltaX) / Z;
    
    if (isFinite(weight) && weight >= 0) {
      weights.push(weight);
      iterations.push(i);
    }
  }
  
  if (weights.length === 0) return;

  const margin = {top: 30, right: 50, bottom: 60, left: 70};
  const plotWidth = ctx.canvas.width - margin.left - margin.right;
  const plotHeight = ctx.canvas.height - margin.top - margin.bottom;

  const yMax = Math.max(...weights) * 1.1;
  const yMin = 0;

  // Grid
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  const xTicks = 10;
  const yTicks = 6;
  
  for (let i = 0; i <= xTicks; i++) {
    const x = margin.left + i / xTicks * plotWidth;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotHeight);
    ctx.stroke();
  }
  
  for (let i = 0; i <= yTicks; i++) {
    const y = margin.top + i / yTicks * plotHeight;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
  }

  // Draw weights as bars
  const barWidth = plotWidth / weights.length * 0.8;
  ctx.fillStyle = "#8b5cf6"; // Purple color for weights
  
  for (let i = 0; i < weights.length; i++) {
    const xPos = margin.left + (i / (weights.length - 1)) * plotWidth;
    const barHeight = (weights[i] / yMax) * plotHeight;
    const yPos = margin.top + plotHeight - barHeight;
    
    ctx.fillRect(xPos - barWidth/2, yPos, barWidth, barHeight);
  }

  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.stroke();

  // X-axis ticks and labels
  ctx.fillStyle = "#2d3748";
  ctx.font = "13px -apple-system, sans-serif";
  ctx.textAlign = "center";
  
  for (let i = 0; i <= xTicks; i++) {
    const x = margin.left + i / xTicks * plotWidth;
    const iter = Math.round(i / xTicks * (deadPts.length - 1));
    ctx.beginPath();
    ctx.moveTo(x, margin.top + plotHeight);
    ctx.lineTo(x, margin.top + plotHeight + 6);
    ctx.stroke();
    ctx.fillText(iter, x, margin.top + plotHeight + 22);
  }
  
  ctx.font = "bold 14px -apple-system, sans-serif";
  ctx.fillText("Iteration", margin.left + plotWidth / 2, ctx.canvas.height - 15);

  // Y-axis ticks and labels
  ctx.textAlign = "right";
  ctx.font = "13px -apple-system, sans-serif";
  
  for (let i = 0; i <= yTicks; i++) {
    const y = margin.top + plotHeight - i / yTicks * plotHeight;
    const val = i / yTicks * yMax;
    ctx.beginPath();
    ctx.moveTo(margin.left - 6, y);
    ctx.lineTo(margin.left, y);
    ctx.stroke();
    ctx.fillText(val.toExponential(2), margin.left - 12, y + 4);
  }
  
  ctx.font = "bold 14px -apple-system, sans-serif";
  ctx.save();
  ctx.translate(20, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Posterior Weight wᵢ", 0, 0);
  ctx.restore();

  // Title
  ctx.textAlign = "center";
  ctx.font = "bold 14px -apple-system, sans-serif";
  ctx.fillText("Peak shows which iterations contribute most to posterior", ctx.canvas.width / 2, 15);
}

  ctx.font = "13px -apple-system, sans-serif";
  
  ctx.fillStyle = "#1e40af";
  ctx.fillRect(ctx.canvas.width - 85, 20, 15, 3);
  ctx.fillStyle = "#2d3748";
  ctx.fillText("x₁ chain", ctx.canvas.width - 65, 25);
  
  ctx.fillStyle = "#dc2626";
  ctx.fillRect(ctx.canvas.width - 85, 40, 15, 3);
  ctx.fillStyle = "#2d3748";
  ctx.fillText("x₂ chain", ctx.canvas.width - 65, 45);


/***************************************************
 * CONTROLS
 ***************************************************/
function start() {
  if (timer) {
    clearInterval(timer);
    timer = null;
    return;
  }
  timer = setInterval(nestedSamplingStep, parseInt(speedSlider.value));
}

function singleStep() {
  nestedSamplingStep();
}

function reset() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  
  // Update domain based on selected distribution
  setDomain();
  
  samplesX.length = 0;
  samplesY.length = 0;
  samplesL.length = 0;
  samplesX_prior.length = 0;
  livePoints.length = 0;
  deadPoints.length = 0;
  
  logZ = -1e100;
  H = 0;
  iteration = 0;
  
  // Initialize live points from prior
  const N = parseInt(nliveSlider.value);
  for (let i = 0; i < N; i++) {
    const x = xmin + Math.random() * (xmax - xmin);
    const y = ymin + Math.random() * (ymax - ymin);
    const L = target(x, y);
    livePoints.push({x, y, L});
  }
  
  drawDensity();
  
  // Draw initial live points
  livePoints.forEach(pt => {
    const [px, py] = toCanvas(pt.x, pt.y);
    ctx.fillStyle = "#3b82f6";
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  
  hx.clearRect(0, 0, hxCanvas.width, hxCanvas.height);
  hy.clearRect(0, 0, hyCanvas.width, hyCanvas.height);
  acfXY.clearRect(0, 0, acfCanvas.width, acfCanvas.height);
  traceX.clearRect(0, 0, traceXCanvas.width, traceXCanvas.height);
  traceY.clearRect(0, 0, traceYCanvas.width, traceYCanvas.height);
  evidencePlot.clearRect(0, 0, evidencePlotCanvas.width, evidencePlotCanvas.height);
  
  stepInfo.innerHTML = "Click 'Start Sampling' or 'Single Iteration' to begin nested sampling.";
  
  document.getElementById("totalIter").textContent = "0";
  document.getElementById("acceptCount").textContent = "0";
  document.getElementById("acceptRate").textContent = "—";
  document.getElementById("essValue").textContent = "—";
}

/***************************************************
 * GAUSSIAN RNG (Box-Muller)
 ***************************************************/
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/***************************************************
 * INITIALIZATION
 ***************************************************/
// Wait for DOM to be ready before initializing
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    reset(); // Initialize live points
  });
} else {
  reset(); // Initialize live points
}

// Add event listener to distribution selector to auto-reset on change
document.getElementById("dist").addEventListener("change", function() {
  reset();
});
