/**
 * SHARED UTILITY FUNCTIONS
 * Functions used across multiple sampling method implementations
 */

/**
 * Gaussian Random Number Generator (Box-Muller transform)
 * @returns {number} A random number from standard normal distribution
 */
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Compute autocorrelation function
 * @param {number[]} data - Time series data
 * @param {number} lagMax - Maximum lag to compute
 * @returns {number[]} Autocorrelation values for lags 0 to lagMax
 */
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

/**
 * Calculate Effective Sample Size (ESS)
 * @param {number[]} samples - Array of sample values
 * @returns {number} Effective sample size
 */
function calculateESS(samples) {
  if (samples.length < 10) return samples.length;
  
  const lagMax = Math.min(100, Math.floor(samples.length / 2));
  const acf = autocorr(samples, lagMax);
  
  let sumRho = 1;
  for (let t = 1; t < acf.length; t++) {
    if (acf[t] < 0.05) break;
    sumRho += 2 * acf[t];
  }
  
  return samples.length / sumRho;
}

/**
 * Draw a trace plot (parameter value vs iteration)
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number[]} data - Sample values
 * @param {string} label - Label for the y-axis
 */
function drawTracePlot(ctx, data, label) {
  const canvas = ctx.canvas;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (data.length < 2) return;

  const margin = {top: 30, right: 50, bottom: 60, left: 70};
  const plotWidth = canvas.width - margin.left - margin.right;
  const plotHeight = canvas.height - margin.top - margin.bottom;

  const yMin = Math.min(...data);
  const yMax = Math.max(...data);
  const yPadding = (yMax - yMin) * 0.1 || 0.1;

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

  // Draw the trace
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  for (let i = 0; i < data.length; i++) {
    const xPos = margin.left + i / (data.length - 1) * plotWidth;
    const yPos = margin.top + plotHeight - (data[i] - (yMin - yPadding)) / (yMax - yMin + 2 * yPadding || 1) * plotHeight;
    
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

/**
 * Draw autocorrelation function for two variables
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number[]} dataX - First variable samples
 * @param {number[]} dataY - Second variable samples
 */
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

/**
 * Draw a 1D histogram
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number[]} data - Sample values
 * @param {number} xmin - Minimum x value for histogram range
 * @param {number} xmax - Maximum x value for histogram range
 * @param {number} bins - Number of histogram bins
 */
function drawHistogram(ctx, data, xmin, xmax, bins = 30) {
  const canvas = ctx.canvas;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (data.length < 5) return;

  const margin = {top: 20, right: 20, bottom: 50, left: 60};
  const plotWidth = canvas.width - margin.left - margin.right;
  const plotHeight = canvas.height - margin.top - margin.bottom;

  // Create histogram
  const binWidth = (xmax - xmin) / bins;
  const counts = new Array(bins).fill(0);
  
  data.forEach(val => {
    const binIndex = Math.floor((val - xmin) / binWidth);
    if (binIndex >= 0 && binIndex < bins) {
      counts[binIndex]++;
    }
  });

  const maxCount = Math.max(...counts);

  // Grid
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + i / 5 * plotHeight;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
  }

  // Draw bars
  ctx.fillStyle = "#3b82f6";
  for (let i = 0; i < bins; i++) {
    const barHeight = (counts[i] / maxCount) * plotHeight;
    const x = margin.left + (i / bins) * plotWidth;
    const y = margin.top + plotHeight - barHeight;
    const barWidth = plotWidth / bins * 0.9;
    
    ctx.fillRect(x, y, barWidth, barHeight);
  }

  // Axes
  ctx.strokeStyle = "#2d3748";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#2d3748";
  ctx.font = "12px -apple-system, sans-serif";
  ctx.textAlign = "center";
  
  for (let i = 0; i <= 5; i++) {
    const x = margin.left + i / 5 * plotWidth;
    const val = xmin + i / 5 * (xmax - xmin);
    ctx.fillText(val.toFixed(1), x, canvas.height - margin.bottom + 20);
  }

  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const y = margin.top + plotHeight - i / 5 * plotHeight;
    const val = Math.round(i / 5 * maxCount);
    ctx.fillText(val, margin.left - 10, y + 4);
  }
}
