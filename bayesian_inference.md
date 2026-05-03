---
layout: default
title: Bayesian Inference
---

# Introduction to Bayesian Inference

Bayesian inference is about updating what we know as we gather more data. Rather than treating parameters as fixed unknown values, we represent uncertainty about them as probability distributions and revise those distributions as evidence accumulates.

## A Brief History

The theorem is named after Thomas Bayes, an English minister and amateur mathematician who developed the core idea in the 1740s. He never published it; the paper was found in his notes and released posthumously in 1763 by his friend Richard Price, where it attracted little attention. Pierre-Simon Laplace independently rediscovered and fully formalized the same framework in 1774, and spent the following decades applying it to real problems in astronomy and demography. He is arguably more responsible than Bayes himself for establishing the method.

The framework then fell out of fashion. By the early 20th century, R. A. Fisher's frequentist paradigm, built around p-values, confidence intervals, and maximum likelihood, had become the standard for scientific inference, and Bayesian methods were largely pushed to the margins. The revival came from computation: the widespread adoption of Markov Chain Monte Carlo methods in the late 1980s and 1990s suddenly made it feasible to compute posteriors for complex, high-dimensional models. This triggered what is now called the Bayesian revolution across statistics, machine learning, and the physical sciences.

## Bayesian vs. Frequentist

The disagreement between Bayesian and frequentist statistics is not about mathematics. Bayes' theorem is a theorem. The disagreement is about what the word *probability* means.

For a frequentist, probability is the long-run frequency of an outcome over many identical, repeatable trials. Parameters are treated as fixed but unknown constants; attaching a probability to them is meaningless. For a Bayesian, probability is a degree of belief, a numerical measure of how confident a rational agent is in a proposition given available evidence. Parameters can have probability distributions because those distributions represent our uncertainty about them.

This difference becomes concrete when interpreting results. A frequentist 95% confidence interval does not mean there is a 95% probability the true value lies inside it. It means that if you repeated the experiment and the procedure infinitely many times, 95% of the intervals constructed would contain the true value. Once a specific interval is computed, no probability attaches to it. A Bayesian 95% credible interval means exactly what most scientists intuitively want: given the data and the model, there is a 95% probability the parameter lies within that range.

In astrophysics, the Bayesian framework is not just convenient, it is often the only one that makes conceptual sense. A gravitational-wave event like GW170817 happens once. A particular supernova explodes once. There is no ensemble of universes to average over, and frequentist probability has no clean answer to questions about unique, non-repeatable events. Bayesian inference handles this naturally: probability is about knowledge and uncertainty, not about imagined repetitions. As Roberto Trotta noted, questions like "what is the probability it was raining in Oxford when William I was crowned?" cannot even be formulated in frequentist statistics. In Bayesian terms, they are perfectly ordinary.

## Bayes' Theorem

Everything follows from one equation:

$$
P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}
$$

The **posterior** $P(\theta \mid D)$ is what we want: our updated belief about the parameters after seeing the data. It is proportional to the **likelihood** $P(D \mid \theta)$, which scores how well each parameter value explains the observations, multiplied by the **prior** $P(\theta)$, which encodes what we knew before. The denominator $P(D)$ normalizes the result to a proper distribution and is also the key quantity for model comparison.

The prior is often misunderstood. It is not a weakness of the framework but an honest acknowledgment that we are never completely ignorant before collecting data. Physical constraints, previous experiments, and theoretical expectations all belong in the prior.

A concrete example helps fix ideas. Suppose you are fitting a straight line to noisy measurements: $\theta$ is the slope and intercept, $D$ is your data points. The likelihood scores how well a given line explains the scatter in the data. The prior might reflect that the slope is expected to be positive and not wildly large. The posterior is then the distribution over all lines, weighted by how well they fit and how plausible they were a priori. This is the pattern that repeats across every Bayesian analysis, from simple regressions to full gravitational-wave source reconstructions.

## Why the Problem is Hard

Computing the posterior requires evaluating the evidence:

$$
P(D) = \int P(D \mid \theta) \cdot P(\theta) \, d\theta
$$

This integral is intractable for almost any real problem. Parameter spaces have tens to thousands of dimensions. Posteriors can be multimodal, highly correlated, and the likelihood itself may require solving a differential equation to evaluate once. There is rarely a closed-form solution.

The standard response is to forget about computing the posterior directly and instead *sample* from it.

## Sampling

The key insight that makes sampling possible is that we never actually need $P(D)$ to draw samples. Every algorithm below only needs to evaluate the **unnormalized** posterior $P(D \mid \theta) \cdot P(\theta)$ — the numerator of Bayes' theorem — which is straightforward to compute for any proposed $\theta$. Because $P(D)$ is just a constant that does not depend on $\theta$, any algorithm that works with ratios or gradients of the posterior density can operate without it.

The goal therefore shifts from computing $P(\theta \mid D)$ to generating a set of samples $\{\theta_1, \ldots, \theta_N\}$ distributed according to it. From those samples you can estimate any posterior quantity you care about: means, credible intervals, marginal distributions, predictions. The framework is general enough to apply whether your model is a two-parameter Gaussian or a full gravitational-wave signal analysis.

The difficulty is that sampling from a distribution you can only evaluate up to a normalization constant — in high dimensions, with complex geometry — is itself a hard problem. Each algorithm below was invented to address specific failure modes of the previous one. Understanding those failures is the fastest way to build genuine intuition.

## Algorithms

Each sampler below comes with an interactive demonstration. The goal is to build intuition about *when* each method works and, more usefully, *when* it fails.

### Markov Chain Monte Carlo

MCMC is the foundation. The idea is to construct a Markov chain whose stationary distribution is the target posterior. The Metropolis-Hastings algorithm is the simplest version: at each step, propose a new point $\theta'$ drawn from some proposal distribution centered on the current point, then accept it with probability

$$
\min\!\left(1,\, \frac{P(D \mid \theta') P(\theta')}{P(D \mid \theta) P(\theta)}\right).
$$

Notice that $P(D)$ cancels exactly in this ratio, which is why sampling sidesteps the intractable integral. Over time the chain explores the posterior, and the sequence of accepted points can be treated as samples.

The reason this works is detailed balance: the accept/reject rule is constructed so that the probability flux between any two states is symmetric under the stationary distribution, which guarantees that distribution is the posterior. The chain is not sampling randomly; it is performing a biased random walk that spends time in proportion to posterior probability.

It has real limitations. The chain explores by random walk, which becomes exponentially inefficient as dimensionality grows. Successive samples are correlated, so the effective sample size is much smaller than the number of iterations. Tuning the proposal width matters a lot: too narrow and the chain mixes slowly, too wide and nearly every proposal is rejected. For low-dimensional, smooth, unimodal posteriors it remains a solid and interpretable choice, and it is where most people should start before moving to more sophisticated methods.

[Explore the MCMC sampler](/html_src/interactive_mcmc.html) — trace plots, autocorrelation analysis, effective sample size, proposal scaling.

### Hamiltonian Monte Carlo

The core failure of MCMC is the random walk: proposals are ignorant of the local geometry, so most of the time the chain diffuses slowly rather than traversing the posterior efficiently. HMC fixes this by using gradient information.

HMC treats sampling as a physics problem. Each parameter $\theta$ is given an auxiliary momentum variable $p$, and the system evolves according to a Hamiltonian

$$
H(\theta, p) = U(\theta) + K(p), \quad U(\theta) = -\log P(\theta \mid D), \quad K(p) = \tfrac{1}{2} p^T p.
$$

Simulating this system for a fixed time, using a reversible volume-preserving integrator called the leapfrog scheme, produces proposals that travel large distances through parameter space while maintaining high acceptance rates, because energy is approximately conserved along the trajectory.

The practical consequence is dramatic: HMC scales to hundreds or thousands of dimensions where random-walk MCMC fails completely. The cost is that gradients of the log-posterior must be computed at every leapfrog step, and the method is sensitive to pathological geometries such as sharp funnels or highly anisotropic distributions that require careful reparametrization. The No-U-Turn Sampler (NUTS) extension automates tuning of the trajectory length, which is why Stan and PyMC use it as their default engine and most practitioners never implement vanilla HMC by hand.

[Explore the HMC sampler](/html_src/interactive_hmc.html) — trajectory visualization, leapfrog integration, energy diagnostics, comparison with MCMC.

### Nested Sampling

Both MCMC and HMC struggle when the posterior has multiple well-separated modes: a chain initialized near one mode may never discover the others. Nested sampling was designed partly for this regime, and it has a second advantage in that it produces the evidence $P(D)$ directly, which neither MCMC nor HMC do.

The algorithm maintains a set of $N$ live points drawn from the prior. At each iteration it identifies the point with the lowest likelihood $L^*$, discards it, and replaces it with a new point drawn from the prior subject to the constraint $L > L^*$. This progressively contracts the live points toward regions of higher likelihood, like a shrinking shell. The discarded points, together with estimates of the prior volume associated with each likelihood contour, give a Riemann-sum approximation of the evidence integral:

$$
P(D) = \int P(D \mid \theta)\, P(\theta)\, d\theta \approx \sum_i L_i \, \Delta X_i
$$

where $\Delta X_i$ is the prior volume shed at step $i$. The posterior samples fall out as a byproduct, weighted by $L_i \Delta X_i$. This makes nested sampling the method of choice when model comparison is the primary goal, or when you need to correctly account for all modes of a multimodal posterior. The main drawback is computational cost relative to HMC for pure parameter inference. In astrophysics, where model selection is often as important as parameter estimation, MultiNest and dynesty are standard tools.

[Explore Nested Sampling](/html_src/interactive_nested_sampling.html) — live point evolution, evidence computation, posterior reconstruction.

### Parallel Tempering

Nested sampling handles multimodality by construction, but it can be expensive. Parallel tempering takes a different approach: run several chains simultaneously, each sampling a *tempered* version of the posterior,

$$
P_T(\theta \mid D) \propto \left[P(D \mid \theta)\right]^{1/T} P(\theta),
$$

at a different temperature $T$. At $T = 1$ this is the true posterior. As $T$ increases, the likelihood is flattened, peaks are smoothed out and the chain hops freely between modes. At very high temperature the distribution approaches the prior. Periodically, adjacent chains propose to swap states; a Metropolis-style acceptance criterion ensures the swaps preserve the correct stationary distributions. The cold chain benefits because the hot chains discover modes it would otherwise miss and shuttle configurations across barriers.

The method works well when the posterior has clearly separated modes, as is common in gravitational-wave parameter estimation where waveform symmetries produce degenerate solutions. Since the chains run in parallel, wall-clock time scales more gracefully than the raw number of likelihood evaluations might suggest. Tuning the temperature schedule is the main practical challenge: swap acceptance rates between adjacent chains are the diagnostic to watch, and poorly spaced temperatures mean information propagates slowly down the ladder.

[Explore Parallel Tempering](/html_src/interactive_parralel_tempering.html) — temperature ladder visualization, swap statistics, mode discovery.

## Further Reading

**Books:**
- *Bayesian Data Analysis* (Gelman et al., 3rd ed.) — the standard reference, comprehensive and more readable than its size suggests.
- *Information Theory, Inference, and Learning Algorithms* (MacKay) — a physicist's approach; the intuition translates well to astrophysics problems.

**Papers:**
- Betancourt (2017), *A Conceptual Introduction to Hamiltonian Monte Carlo* — builds the geometric intuition that makes HMC feel inevitable rather than arbitrary.
- Skilling (2006), *Nested Sampling for General Bayesian Computation* — the original paper, still the clearest introduction to the method.
- Trotta (2008), *Bayes in the Sky* — a thorough review of Bayesian methods in cosmology and the philosophical case for them in astrophysics.

**Software:**
- [Stan](https://mc-stan.org/) and [PyMC](https://www.pymc.io/) for HMC-based inference.
- [dynesty](https://dynesty.readthedocs.io/) and [MultiNest](https://github.com/farhanferoz/MultiNest) for nested sampling.
