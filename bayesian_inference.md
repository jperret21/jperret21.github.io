---

layout: default
title: Introduction to Bayesian Inference

---

# Introduction to Bayesian Inference

Bayesian inference is fundamentally about updating what we know as we gather more information. Unlike frequentist approaches that treat parameters as fixed but unknown values, Bayesian methods represent our uncertainty about parameters as probability distributions, and we update these distributions as data comes in.

## The Core Principle: Bayes' Theorem

At the heart of Bayesian statistics lies Bayes' theorem, which formalizes how we update our beliefs about parameters θ given observed data D:

$$
P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}
$$

- **Posterior** $P(\theta \mid D)$: What we want to know—our updated beliefs about the parameters after seeing the data.
- **Likelihood** $P(D \mid \theta)$: How well different parameter values explain what we actually observed.
- **Prior** $P(\theta)$: What we knew (or assumed) before seeing the data. This can encode physical constraints, previous measurements, or regularization preferences.
- **Evidence** $P(D)$: A normalization constant that ensures the posterior integrates to 1. Also crucial for model comparison.

The elegance of Bayes' theorem is in how intuitive it is: **the posterior balances what the data tells us (likelihood) with what we already knew (prior)**.


## The Fundamental Challenge

The catch is that computing the posterior requires the evidence:

$$
P(D) = \int P(D \mid \theta) \cdot P(\theta) \, d\theta
$$

For most real-world problems, this integral is intractable. Why? High-dimensional parameter spaces can have hundreds or thousands of dimensions. Posteriors are often multimodal with complex correlations. Many models don't have conjugate priors, so there's no closed-form solution. Some models require running simulations or solving differential equations just to evaluate the likelihood once.

This is where computational methods become essential—we can't compute the posterior analytically, so we need to approximate it.


## Why Sampling?

Instead of computing $P(\theta \mid D)$ directly, modern Bayesian inference relies on Monte Carlo sampling: we generate samples $\{\theta_1, \theta_2, ..., \theta_N\}$ distributed according to the posterior.

With enough samples, we can:
- Estimate any posterior statistic (means, medians, credible intervals)
- Visualize marginal distributions and correlations between parameters
- Make predictions by propagating uncertainty to new data:
$$
P(D_{\text{new}} \mid D) = \int P(D_{\text{new}} \mid \theta) P(\theta \mid D) d\theta
$$
- Compare models via Bayes factors

The beauty of sampling is its generality—the same framework works whether you're fitting a simple linear model or simulating galaxy formation.


## The Sampling Zoo: Different Algorithms for Different Problems

There's no universal best sampler. Each method trades off efficiency, scalability, robustness, and ease of use differently. Some excel in high dimensions but struggle with multimodality. Others naturally estimate the evidence but are slower for pure parameter inference. Understanding these trade-offs is key to choosing the right tool.

Below are several sampling algorithms I've worked with extensively, complete with interactive demonstrations showing when each method shines and when it struggles.

### Markov Chain Monte Carlo (MCMC)

**The foundation of computational Bayesian inference.**

MCMC builds a Markov chain whose stationary distribution is the posterior we want to sample. The classic Metropolis-Hastings algorithm is simple: propose a new parameter value, then accept or reject based on the posterior density ratio.

This works well for low-dimensional problems (up to ~10-20 dimensions) with smooth, unimodal posteriors. It's also incredibly valuable pedagogically—MCMC is where most people build intuition about sampling.

The key concepts:
- **Burn-in**: How long until the chain forgets its starting point?
- **Autocorrelation**: Successive samples are correlated, so the effective sample size is smaller than the number of iterations.
- **Proposal tuning**: Acceptance rates around 20-40% are typically optimal for random-walk proposals.
- **Curse of dimensionality**: Random-walk behavior becomes exponentially inefficient as dimensions increase.

While not competitive for high-dimensional or complex problems, MCMC is essential for understanding why more sophisticated methods exist.

➡️ [Explore the MCMC sampler](/html_src/interactive_mcmc.html)
  
*Includes: trace plots, autocorrelation analysis, effective sample size computation, proposal scaling experiments*

---

### Hamiltonian Monte Carlo (HMC)

**Geometry-aware sampling using gradient information.**

HMC treats sampling as physics: imagine a frictionless particle sliding through the log-posterior landscape. By using gradient information, it proposes distant moves that follow the posterior's geometry rather than wandering randomly.

Why is this so effective?
- **Gradients guide proposals** along level sets of the posterior
- **Momentum** lets the sampler traverse large distances coherently
- **Volume preservation** from Hamiltonian dynamics means high acceptance rates even for distant proposals

HMC handles high-dimensional smooth posteriors with ease—hundreds or thousands of dimensions where vanilla MCMC fails completely. This is why modern tools like Stan and PyMC use HMC as their default engine.

The challenges:
- Requires tuning step size and number of leapfrog steps (though NUTS largely automates this)
- Sensitive to pathological geometries like funnels or strongly correlated parameters
- Computing gradients has a cost
- Can occasionally get stuck in periodic orbits

Despite these issues, HMC has become the workhorse of contemporary Bayesian computation.


➡️ [Explore the HMC sampler](/html_src/interactive_hmc.html)
 
*Includes: trajectory visualization, leapfrog integration, energy diagnostics, comparison with MCMC*

---

### Nested Sampling

**Simultaneous sampling and evidence computation.**

Unlike MCMC methods that target the posterior directly, nested sampling explores nested shells of increasing likelihood. It starts with N "live points" sampled from the prior. At each iteration, it removes the point with the lowest likelihood and replaces it with a new point from the prior that has higher likelihood. This systematically contracts the prior volume toward high-likelihood regions.

The key advantage: this process naturally computes the evidence $P(D)$ as a byproduct, making nested sampling invaluable for model comparison.

When to use it:
- **Model comparison** matters as much as parameter inference
- **Multimodal posteriors** where you need to find and properly weight multiple modes
- **Low-to-moderate dimensions** where you want both posterior samples and evidence

The trade-offs:
- Slower than MCMC/HMC for pure parameter inference
- Efficiency depends critically on the "constrained prior sampling" step
- Number of live points controls accuracy vs. cost

In astrophysics and cosmology, where model selection is often the main question, nested sampling has become a standard tool.


➡️ [Explore Nested Sampling](/html_src/interactive_nested_sampling.html)  

*Includes: live point evolution, evidence computation, posterior reconstruction from nested samples*

---

### Parallel Tempering

**Escaping local modes via temperature ladder.**

Parallel tempering (also called replica exchange MCMC) runs multiple chains at different "temperatures"—high temperatures flatten the posterior to make mode-hopping easy, while the cold chain samples the true posterior accurately.

The algorithm runs M chains targeting $[P(\theta \mid D)]^{1/T_i}$ for temperatures $T_1 = 1 < T_2 < ... < T_M$. The hot chains explore freely across a flattened landscape. Periodically, we propose swapping states between adjacent temperature chains. This lets information from high-temperature exploration gradually inform the cold chain, allowing it to discover and properly weight all modes.

When to use it:
- **Highly multimodal posteriors** where standard MCMC gets stuck
- **Rugged likelihood surfaces** with many local optima
- **Phase transitions** where the posterior splits between discrete regimes
- When you suspect your posterior has structure you haven't found yet

Practical considerations:
- Temperature schedule needs tuning (too few temperatures = failed swaps, too many = wasted computation)
- Cost scales linearly with number of chains (but chains run in parallel)
- Swap acceptance rates diagnose whether temperature spacing is appropriate

When configured properly, parallel tempering can turn impossible sampling problems into tractable ones.


➡️ [Explore Parallel Tempering](/html_src/interactive_parralel_tempering.html) 

*Includes: temperature ladder visualization, swap statistics, mode discovery demos*

---

## Further Reading

If you want to go deeper, here are resources I've found particularly valuable:

**Core texts:**
- **Bayesian Data Analysis** (Gelman et al., 3rd ed.) — The definitive reference. Comprehensive and surprisingly readable.
- **Information Theory, Inference, and Learning Algorithms** (MacKay) — A physicist's perspective that resonates well with astrophysics intuition.
- **Pattern Recognition and Machine Learning** (Bishop) — Beautifully connects Bayesian inference to modern ML.

**Sampling methods:**
- **Handbook of Markov Chain Monte Carlo** (Brooks et al.) — Everything you could want to know about MCMC.
- **A Conceptual Introduction to Hamiltonian Monte Carlo** (Betancourt, 2017) — Builds geometric intuition that makes HMC feel inevitable.
- **Nested Sampling for General Bayesian Computation** (Skilling, 2006) — The original paper, still the clearest introduction.

**Software:**
- [Stan](https://mc-stan.org/) — Production-quality HMC with automatic differentiation
- [PyMC](https://www.pymc.io/) — Pythonic probabilistic programming with excellent docs
- [Turing.jl](https://turing.ml/) — Julia's speed meets probabilistic inference

I recommend trying multiple frameworks—each one will deepen your understanding in different ways.
