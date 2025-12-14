---
layout: default
title: Introduction to Bayesian Inference
---

# Introduction: Bayesian Inference

At the heart of Bayesian statistics lies **Bayes' theorem**, which describes how we update our beliefs about parameters θ given observed data D:

$$
P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}
$$

Where:
- $P(\theta \mid D)$ — *Posterior distribution*  
  The probability distribution of the parameters $\theta$ given the observed data $D$.  
  This is the quantity we aim to infer.

- $P(D \mid \theta)$ — *Likelihood*  
  The probability of observing the data $D$ assuming the parameters take the value $\theta$.

- $P(\theta)$ — *Prior distribution*  
  Encodes prior knowledge or assumptions about the parameters before observing the data.

- $P(D)$ — *Marginal likelihood* (or *evidence*)  
  A normalization constant ensuring that the posterior integrates to one
  

### The Challenge

The denominator P(D) requires integrating over all possible parameter values:

$$
P(D) = \int P(D \mid \theta) \cdot P(\theta) \, d\theta
$$

For most real-world problems, this integral is **intractable** — impossible to compute analytically. This is where **sampling algorithms** come in.

### Why We Need Samplers

Instead of computing the posterior distribution directly, samplers generate representative samples from P(θ \| D). With enough samples, we can:
- Estimate posterior means, medians, and credible intervals
- Visualize the posterior distribution
- Make predictions on new data
- Perform model comparison

---

## 📌 Available Samplers

This page gathers different sampling algorithms I experiment with, mostly in the context of Bayesian inference and high-dimensional problems. Each sampler has its own dedicated page with interactive visualizations, implementation details, and diagnostics.

---

### 🔹 Markov Chain Monte Carlo (MCMC)

A foundational Metropolis–Hastings sampler that forms the basis for understanding modern sampling methods. This implementation serves as a reference for exploring fundamental concepts like:

- **Convergence diagnostics** — How to know when your chain has reached the posterior
- **Autocorrelation** — Understanding sample dependencies and effective sample size
- **Acceptance rates** — Tuning proposals for efficient exploration
- **Scaling behavior** — Why vanilla MCMC struggles in high dimensions

While not the most efficient for complex posteriors, MCMC remains invaluable for building intuition and as a diagnostic baseline.

➡️ **[Explore the MCMC sampler](/html_src/interactive_mcmc.html)**  
*Tags:* Bayesian inference, Metropolis-Hastings, diagnostics, autocorrelation

---

### 🔹 Hamiltonian Monte Carlo (HMC)

 A gradient-based sampler that treats sampling as a physics simulation problem.

HMC leverages Hamiltonian dynamics to propose distant states with high acceptance probability, making it particularly effective for:

- **High-dimensional posteriors** — Efficient exploration where MCMC fails
- **Complex geometries** — Following curved posterior landscapes
- **Reduced autocorrelation** — Longer jumps mean fewer correlated samples

By simulating the motion of a particle with momentum through the posterior landscape, HMC can traverse the distribution much more efficiently than random-walk methods.

➡️ **[Explore the HMC sampler](/html_src/interactive_hmc.html)** 
*Tags:* HMC, gradients, high-dimensional inference, Hamiltonian dynamics

---

### 🔹 Nested sampling 

➡️ **[Explore Nested sampling](/html_src/interactive_nested_sampling.html)** 
*Tags:* HMC, gradients, high-dimensional inference, Hamiltonian dynamics

---

### 🔹 MCMC Parralel tempering

➡️ **[Explore MCMC parralel tempering](/html_src/interactive_parralel_tempering.html)** 
*Tags:* HMC, gradients, high-dimensional inference, Hamiltonian dynamics

---






## 🔧 Implementation Notes

All samplers are implemented with:
- **Interactive visualizations** — See the algorithms in action
- **Step-by-step explanations** — Understand what's happening at each iteration
- **Diagnostic tools** — Assess convergence and sample quality
- **Comparative benchmarks** — Performance across different problem types

---

## 📚 Further Reading

- **Bayesian Data Analysis** (Gelman et al.) — Comprehensive treatment of Bayesian methods
- **MCMC Handbook** (Brooks et al.) — Deep dive into sampling algorithms
- **[Stan Documentation](https://mc-stan.org/docs/)** — Practical implementation patterns