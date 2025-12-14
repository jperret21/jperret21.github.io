/**
 * HMC - Algorithm-specific code
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

// Set canvas widths to actual container width
acfCanvas.width = acfCanvas.offsetWidth;
traceXCanvas.width = traceXCanvas.offsetWidth;
traceYCanvas.width = traceYCanvas.offsetWidth;

const stepInfo = document.getElementById("stepInfo");
const epsilonSlider = document.getElementById("epsilon");
const numStepsSlider = document.getElementById("numSteps");
const speedSlider = document.getElementById("speed");
const epsilonVal = document.getElementById("epsilonVal");
const numStepsVal = document.getElementById("numStepsVal");
const speedVal = document.getElementById("speedVal");

epsilonSlider.oninput = () => {
  epsilonVal.textContent = parseFloat(epsilonSlider.value).toFixed(2);
};

numStepsSlider.oninput = () => {
  numStepsVal.textContent = numStepsSlider.value;
};

speedSlider.oninput = () => {
  speedVal.textContent = speedSlider.value;
  if (timer) {
    clearInterval(timer);
    timer = setInterval(hmcStep, parseInt(speedSlider.value));
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

let x = 0, y = 0;
let p = 0;
let accepted = 0, total = 0;
let timer = null;

/***************************************************
 * TARGET DISTRIBUTIONS
 ***************************************************/
function target(x, y) {
  const type = document.getElementById("dist").value;
  if (type === "funnel") return targetFunnel(x, y);
  if (type === "banana") return targetBanana(x, y);
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

/***************************************************
 * GRADIENTS OF NEGATIVE LOG POSTERIOR (for HMC)
 ***************************************************/
// Gradient of U(q) = -log π(q)
function gradU(x, y) {
  const type = document.getElementById("dist").value;
  if (type === "funnel") return gradUFunnel(x, y);
  if (type === "banana") return gradUBanana(x, y);
  return gradUGaussian(x, y);
}

function gradUGaussian(x, y) {
  // U = (x² - 2ρxy + y²)/(2(1-ρ²))
  // ∂U/∂x = (2x - 2ρy)/(2(1-ρ²)) = (x - ρy)/(1-ρ²)
  // ∂U/∂y = (2y - 2ρx)/(2(1-ρ²)) = (y - ρx)/(1-ρ²)
  const rho = 0.8;
  const denom = 1 - rho*rho;
  return [(x - rho*y) / denom, (y - rho*x) / denom];
}

function gradUBanana(x, y) {
  // U = (x² + 100(y - x²)²)/200
  // ∂U/∂x = (2x + 100·2(y - x²)·(-2x))/200 = (x - 200x(y - x²))/100
  // ∂U/∂y = 100·2(y - x²)/200 = (y - x²)
  const diff = y - x*x;
  return [(x - 200*x*diff) / 100, diff];
}

function gradUFunnel(x, y) {
  // U = x²/18 + y²/(2exp(2x))
  // ∂U/∂x = 2x/18 - y²·exp(-2x)
  // ∂U/∂y = y·exp(-2x)
  const ex = Math.exp(x);
  return [x / 9 - y*y / (ex*ex), y / (ex*ex)];
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
 * MH STEP
 ***************************************************/
/***************************************************
 * HMC STEP
 ***************************************************/
function hmcStep() {
  const epsilon = parseFloat(epsilonSlider.value);
  const L = parseInt(numStepsSlider.value);

  // Current position and energy
  const q0 = [x, y];
  const U0 = -Math.log(target(x, y));

  // Sample momentum
  const p0 = [randn(), randn()];
  const K0 = 0.5 * (p0[0]**2 + p0[1]**2);
  const H0 = U0 + K0;

  // Leapfrog integration
  let q = [q0[0], q0[1]];
  let p = [p0[0], p0[1]];

  // Half step for momentum
  let grad = gradU(q[0], q[1]);
  p[0] -= epsilon * 0.5 * grad[0];
  p[1] -= epsilon * 0.5 * grad[1];

  // Full steps
  for (let i = 0; i < L; i++) {
    // Full step for position
    q[0] += epsilon * p[0];
    q[1] += epsilon * p[1];

    // Full step for momentum (except last step)
    grad = gradU(q[0], q[1]);
    if (i < L - 1) {
      p[0] -= epsilon * grad[0];
      p[1] -= epsilon * grad[1];
    } else {
      // Half step for last
      p[0] -= epsilon * 0.5 * grad[0];
      p[1] -= epsilon * 0.5 * grad[1];
    }
  }

  // Proposed state and energy
  const targetVal = target(q[0], q[1]);
  const U1 = targetVal > 0 ? -Math.log(targetVal) : 1e10;
  const K1 = 0.5 * (p[0]**2 + p[1]**2);
  const H1 = U1 + K1;

  // Metropolis acceptance
  const deltaH = H1 - H0;
  const acceptProb = Math.min(1, Math.exp(-deltaH));
  const u = Math.random();

  let acc = false;
  if (u < acceptProb) {
    x = q[0];
    y = q[1];
    acc = true;
    accepted++;
  }

  total++;
  samplesX.push(x);
  samplesY.push(y);

  // Draw point on canvas
  const [px, py] = toCanvas(x, y);
  ctx.fillStyle = acc ? "#38a169" : "#ed8936";
  ctx.beginPath();
  ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.strokeStyle = "white";
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // Draw trajectory (show the leapfrog path)
  if (L > 5) {
    ctx.strokeStyle = acc ? "rgba(56, 161, 105, 0.3)" : "rgba(237, 137, 54, 0.3)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    
    // Recompute trajectory for visualization
    let qt = [q0[0], q0[1]];
    let pt = [p0[0], p0[1]];
    const [px0, py0] = toCanvas(qt[0], qt[1]);
    ctx.moveTo(px0, py0);
    
    let gradt = gradU(qt[0], qt[1]);
    pt[0] -= epsilon * 0.5 * gradt[0];
    pt[1] -= epsilon * 0.5 * gradt[1];
    
    for (let i = 0; i < L; i++) {
      qt[0] += epsilon * pt[0];
      qt[1] += epsilon * pt[1];
      const [pxt, pyt] = toCanvas(qt[0], qt[1]);
      ctx.lineTo(pxt, pyt);
      
      gradt = gradU(qt[0], qt[1]);
      if (i < L - 1) {
        pt[0] -= epsilon * gradt[0];
        pt[1] -= epsilon * gradt[1];
      } else {
        pt[0] -= epsilon * 0.5 * gradt[0];
        pt[1] -= epsilon * 0.5 * gradt[1];
      }
    }
    ctx.stroke();
  }

  // Update displays
  drawHistograms();
  updateStepInfo(q[0], q[1], H0, H1, deltaH, acceptProb, u, acc);
  updateStats();
}

/***************************************************
 * UPDATE STEP INFO
 ***************************************************/
function updateStepInfo(qx, qy, H0, H1, deltaH, acceptProb, u, acc) {
  const statusClass = acc ? 'status-accepted' : 'status-rejected';
  const statusText = acc ? 'ACCEPTED ✓' : 'REJECTED ✗';
  
  stepInfo.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
      <div>
        <strong>Current State (t)</strong><br>
        q₁ = ${x.toFixed(4)}<br>
        q₂ = ${y.toFixed(4)}<br>
        H(q<sub>t</sub>, p) = ${H0.toFixed(4)}
      </div>
      <div>
        <strong>Proposed State (after trajectory)</strong><br>
        q₁′ = ${qx.toFixed(4)}<br>
        q₂′ = ${qy.toFixed(4)}<br>
        H(q′, p′) = ${H1.toFixed(4)}
      </div>
    </div>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
      <strong>Hamiltonian Dynamics</strong><br>
      Initial Hamiltonian: H₀ = ${H0.toFixed(4)}<br>
      Final Hamiltonian: H₁ = ${H1.toFixed(4)}<br>
      Change in energy: ΔH = ${deltaH.toFixed(4)}<br>
      <div style="margin-top: 0.5rem;">
        <strong>Metropolis Acceptance</strong><br>
        Accept prob: α = exp(-ΔH) = ${acceptProb.toFixed(4)}<br>
        Random draw: u = ${u.toFixed(4)}<br>
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
  document.getElementById("totalIter").textContent = total.toLocaleString();
  document.getElementById("acceptCount").textContent = accepted.toLocaleString();
  
  const rate = accepted / total;
  document.getElementById("acceptRate").textContent = 
    (rate * 100).toFixed(1) + "% (" + accepted + "/" + total + ")";
  
  // Estimate ESS from ACF
  if (samplesX.length > 50) {
    const acfX = autocorr(samplesX, 100);
    let sumACF = 0;
    for (let i = 1; i < acfX.length && acfX[i] > 0; i++) {
      sumACF += acfX[i];
    }
    const ess = samplesX.length / (1 + 2 * sumACF);
    document.getElementById("essValue").textContent = Math.round(ess).toLocaleString();
  }
}

/***************************************************
 * HISTOGRAMS
 ***************************************************/
function drawHistograms() {
  drawHistogram(hx, samplesX, "x₁", hxCanvas);
  drawHistogram(hy, samplesY, "x₂", hyCanvas);
  drawACFXY(acfXY, samplesX, samplesY);
  drawTracePlot(traceX, samplesX, "x₁", traceXCanvas);
  drawTracePlot(traceY, samplesY, "x₂", traceYCanvas);
}

function drawHistogram(ctx, data, label, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (data.length === 0) return;

  const bins = 40;
  const hist = Array(bins).fill(0);
  const binWidth = (xmax - xmin) / bins;

  data.forEach(v => {
    const i = Math.floor((v - xmin) / (xmax - xmin) * bins);
    if (i >= 0 && i < bins) hist[i]++;
  });

  const hmax = Math.max(...hist, 1);
  
  // Normalize histogram to approximate density
  const normalized = hist.map(h => h / (data.length * binWidth));
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
}

/***************************************************
 * TRACE PLOTS
 ***************************************************/
function drawTracePlot(ctx, data, label, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (data.length < 2) return;

  const margin = {top: 20, right: 50, bottom: 50, left: 70};
  const plotWidth = canvas.width - margin.left - margin.right;
  const plotHeight = canvas.height - margin.top - margin.bottom;

  // Find data range
  const yMin = Math.min(...data, -4);
  const yMax = Math.max(...data, 4);
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

  // Draw trace line - use different color based on label
  ctx.strokeStyle = label === "x₁" ? "#1e40af" : "#dc2626";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  
  for (let i = 0; i < data.length; i++) {
    const xPos = margin.left + i / (data.length - 1) * plotWidth;
    const yPos = margin.top + plotHeight - (data[i] - (yMin - yPadding)) / (yMax - yMin + 2 * yPadding) * plotHeight;
    
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
    const iterLabel = Math.round(i / xTicks * (data.length - 1));
    ctx.beginPath();
    ctx.moveTo(x, margin.top + plotHeight);
    ctx.lineTo(x, margin.top + plotHeight + 6);
    ctx.stroke();
    ctx.fillText(iterLabel, x, margin.top + plotHeight + 20);
  }
  
  ctx.font = "bold 13px -apple-system, sans-serif";
  ctx.fillText("Iteration", margin.left + plotWidth / 2, canvas.height - 10);

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
  ctx.fillText(label + " value", 0, 0);
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

function drawACFXY(ctx, dataX, dataY) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
  if (dataX.length < 10) return;

  // Start with a reasonable max lag
  const initialLagMax = Math.min(300, Math.floor(dataX.length / 2));
  const acfX = autocorr(dataX, initialLagMax);
  const acfY = autocorr(dataY, initialLagMax);
  
  // Find where ACF crosses zero for both chains
  function findZeroCrossing(acf) {
    for (let i = 1; i < acf.length; i++) {
      if (acf[i] <= 0) return i;
    }
    return acf.length - 1;
  }
  
  const zeroCrossX = findZeroCrossing(acfX);
  const zeroCrossY = findZeroCrossing(acfY);
  const maxZeroCross = Math.max(zeroCrossX, zeroCrossY);
  
  // Show zero crossing + 1/3 more, but at least 50 lags and at most initialLagMax
  const adaptiveLagMax = Math.min(initialLagMax, Math.max(50, Math.floor(maxZeroCross * 1.33)));
  
  // Trim ACF arrays to adaptive length
  const acfXTrimmed = acfX.slice(0, adaptiveLagMax + 1);
  const acfYTrimmed = acfY.slice(0, adaptiveLagMax + 1);
  const n = acfXTrimmed.length;

  const margin = {top: 30, right: 100, bottom: 60, left: 70};
  const plotWidth = ctx.canvas.width - margin.left - margin.right;
  const plotHeight = ctx.canvas.height - margin.top - margin.bottom;

  const yMin = -0.2;
  const yMax = 1.0;

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

  // Zero line
  const zeroY = margin.top + (yMax - 0) / (yMax - yMin) * plotHeight;
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(margin.left, zeroY);
  ctx.lineTo(margin.left + plotWidth, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw ACF for x₁
  ctx.strokeStyle = "#1e40af";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const xPos = margin.left + i / (n - 1) * plotWidth;
    const yPos = margin.top + (yMax - acfXTrimmed[i]) / (yMax - yMin) * plotHeight;
    if (i === 0) ctx.moveTo(xPos, yPos);
    else ctx.lineTo(xPos, yPos);
  }
  ctx.stroke();

  // Draw ACF for x₂
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const xPos = margin.left + i / (n - 1) * plotWidth;
    const yPos = margin.top + (yMax - acfYTrimmed[i]) / (yMax - yMin) * plotHeight;
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
  ctx.font = "13px -apple-system, sans-serif";
  ctx.textAlign = "center";
  
  for (let i = 0; i <= xTicks; i++) {
    const x = margin.left + i / xTicks * plotWidth;
    const lag = Math.round(i / xTicks * adaptiveLagMax);
    ctx.beginPath();
    ctx.moveTo(x, margin.top + plotHeight);
    ctx.lineTo(x, margin.top + plotHeight + 6);
    ctx.stroke();
    ctx.fillText(lag, x, margin.top + plotHeight + 22);
  }
  
  ctx.font = "bold 14px -apple-system, sans-serif";
  ctx.fillText("Lag τ", margin.left + plotWidth / 2, ctx.canvas.height - 15);

  // Y-axis ticks and labels
  ctx.textAlign = "right";
  ctx.font = "13px -apple-system, sans-serif";
  const yLabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0];
  
  yLabels.forEach(val => {
    const y = margin.top + (yMax - val) / (yMax - yMin) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(margin.left - 6, y);
    ctx.lineTo(margin.left, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), margin.left - 12, y + 4);
  });
  
  ctx.font = "bold 14px -apple-system, sans-serif";
  ctx.save();
  ctx.translate(20, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("ACF", 0, 0);
  ctx.restore();

  // Legend
  ctx.textAlign = "left";
  ctx.font = "13px -apple-system, sans-serif";
  
  ctx.fillStyle = "#1e40af";
  ctx.fillRect(ctx.canvas.width - 85, 20, 15, 3);
  ctx.fillStyle = "#2d3748";
  ctx.fillText("x₁ chain", ctx.canvas.width - 65, 25);
  
  ctx.fillStyle = "#dc2626";
  ctx.fillRect(ctx.canvas.width - 85, 40, 15, 3);
  ctx.fillStyle = "#2d3748";
  ctx.fillText("x₂ chain", ctx.canvas.width - 65, 45);
}

/***************************************************
 * CONTROLS
 ***************************************************/
function start() {
  if (timer) {
    clearInterval(timer);
    timer = null;
    return;
  }
  timer = setInterval(hmcStep, parseInt(speedSlider.value));
}

function singleStep() {
  hmcStep();
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
  
  // Set initial position based on distribution
  const type = document.getElementById("dist").value;
  if (type === "funnel") {
    // Start near the neck of the funnel
    x = 0;
    y = 0;
  } else if (type === "banana") {
    // Start near the banana center
    x = 0;
    y = 0;
  } else {
    // Gaussian - start at origin
    x = 0;
    y = 0;
  }
  
  p = target(x, y);
  accepted = 0;
  total = 0;
  
  drawDensity();
  hx.clearRect(0, 0, hxCanvas.width, hxCanvas.height);
  hy.clearRect(0, 0, hyCanvas.width, hyCanvas.height);
  acfXY.clearRect(0, 0, acfCanvas.width, acfCanvas.height);
  traceX.clearRect(0, 0, traceXCanvas.width, traceXCanvas.height);
  traceY.clearRect(0, 0, traceYCanvas.width, traceYCanvas.height);
  
  stepInfo.innerHTML = "Click 'Start Sampling' or 'Single Step' to begin the MCMC algorithm.";
  
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
p = target(x, y);
drawDensity();

// Add event listener to distribution selector to auto-reset on change
document.getElementById("dist").addEventListener("change", function() {
  reset();
});
