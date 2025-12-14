---

layout: default
title: Introduction to Bayesian Inference

---

# Introduction: Bayesian Inference

Bayesian inference provides a principled framework for reasoning under uncertainty. Unlike frequentist approaches that treat parameters as fixed unknowns, Bayesian methods treat them as random variables with probability distributions that we update as we observe data.

## The Core Principle: Bayes' Theorem

At the heart of Bayesian statistics lies Bayes' theorem,, which formalizes how we update our beliefs about parameters θ given observed data D:
$$
P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}
$$

The posterior distribution $P(\theta \mid D)$ represents our updated beliefs about the parameters after observing the data—this is what we seek to infer. The likelihood $P(D \mid \theta)$ quantifies how probable our observed data is under different parameter values, connecting our mathematical model to empirical reality. Our prior distribution $P(\theta)$ encodes initial beliefs or constraints on the parameters before seeing the data, which can incorporate physical constraints, domain expertise, or regularization preferences. Finally, the marginal likelihood  $P(D)$ (also called the evidence) serves as a normalization constant ensuring the posterior is a proper probability distribution. The beauty of this formula lies in its interpretability: the posterior is proportional to how well the parameters explain the data (likelihood) weighted by how plausible those parameters were a priori (prior).


## The Fundamental Challenge

The marginal likelihood requires integrating over the entire parameter space:
$$
P(D) = \int P(D \mid \theta) \cdot P(\theta) \, d\theta
$$
For most real-world problems, this integral is intractable. Consider the challenges: high-dimensional parameter spaces where we must integrate over hundreds or thousands of dimensions; complex, multimodal posterior landscapes with multiple peaks and valleys; non-conjugate prior-likelihood pairs that don't yield closed-form solutions; and implicit models where the likelihood itself requires solving differential equations or running expensive simulations. This is where computational methods become essential. We cannot compute the posterior analytically, so we must approximate it.


## Why Sampling?

Instead of computing $P(\theta \mid D)$ directly, modern Bayesian inference relies on  Monte Carlo sampling
: we generate a collection of samples $\{\theta_1, \theta_2, ..., \theta_N\}$ that are distributed according to the posterior. With enough representative samples, we can estimate posterior statistics like means, medians, standard deviations, and quantiles. We can construct credible intervals—Bayesian confidence regions with direct probabilistic interpretation. These samples allow us to visualize marginal distributions and understand correlations and constraints between parameters. Perhaps most importantly, they enable us to make predictions by propagating posterior uncertainty to new data via the integral $P(D_{\text{new}} \mid D) = \int P(D_{\text{new}} \mid \theta) P(\theta \mid D) d\theta$, and they facilitate model comparison through Bayes factors and model selection.


## The Sampling Zoo: Different Algorithms for Different Problems

No single sampling algorithm dominates all scenarios. Each method makes different trade-offs in efficiency (how many samples are needed for reliable inference?), scalability (how does performance degrade in high dimensions?), robustness (does it work on multimodal, heavy-tailed, or pathological distributions?), automation (how much tuning is required from the user?), and specialization (does it provide additional information like evidence estimates?). Understanding these trade-offs is essential for choosing the right tool for your specific problem.
Below, I present several sampling algorithms I've implemented and explored, each with interactive demonstrations showing their strengths, limitations, and appropriate use cases.

### Markov Chain Monte Carlo (MCMC)

**The foundation of computational Bayesian inference.**

MCMC constructs a Markov chain whose stationary distribution is the target posterior. The Metropolis-Hastings algorithm is the canonical example: propose a new state, accept or reject based on the posterior density ratio.
The method shines in low-dimensional problems (typically fewer than 10-20 dimensions) with unimodal, well-behaved posteriors. It remains invaluable for teaching and building intuition about sampling, and serves as a diagnostic baseline for comparing more sophisticated methods.


Understanding MCMC requires grasping several key concepts. First is the question of burn-in and convergence: how long must the chain run until it "forgets" its initialization? Autocorrelation between successive samples means the effective sample size is less than the raw sample count—samples aren't truly independent. Proposal tuning matters enormously; acceptance rates around 20-40% are often optimal (Roberts & Rosenthal, 2001). Most critically, the curse of dimensionality strikes hard: random-walk proposals become exponentially inefficient as dimension increases.
While not competitive for complex modern problems, understanding MCMC is essential for grasping why gradient-based and adaptive methods were developed.
➡️ [Explore the MCMC sampler](/html_src/interactive_mcmc.html)
  
*Includes: trace plots, autocorrelation analysis, effective sample size computation, proposal scaling experiments*

---

### Hamiltonian Monte Carlo (HMC)

**Geometry-aware sampling via Hamiltonian dynamics.**

HMC exploits gradient information to propose distant states that follow the posterior's geometry, dramatically improving on random-walk MCMC. The key idea: treat sampling as simulating a frictionless particle moving through the log-posterior landscape.
Why does this work so well? Gradient guidance means proposals move along level sets of the posterior rather than wandering randomly. Momentum allows the particle to traverse large distances without random drift. The volume-preserving nature of Hamiltonian dynamics enables high acceptance rates even for distant proposals—a particle that travels far through phase space under Hamiltonian evolution naturally lands in regions of similar probability density.


The method excels for high-dimensional smooth posteriors, handling hundreds or even thousands of dimensions where MCMC fails catastrophically. It's particularly effective for problems where gradients are available (or can be approximated) and for exploring complex geometries with curved correlations. Modern probabilistic programming systems like Stan and PyMC have made HMC the workhorse of contemporary Bayesian computation.
However, challenges remain. The method requires tuning both step size and the number of leapfrog steps. It can be sensitive to stiff directions and pathological geometries like funnels. Gradient computation carries its own cost, and periodic orbits can emerge in pathological cases. HMC variants like NUTS (No-U-Turn Sampler) address many tuning issues automatically, which explains the method's widespread adoption.


➡️ [Explore the HMC sampler](/html_src/interactive_hmc.html)
 
*Includes: trajectory visualization, leapfrog integration, energy diagnostics, comparison with MCMC*

---

### Nested Sampling

**Simultaneous sampling and evidence computation.**

Unlike MCMC methods that target the posterior, nested sampling explores the likelihood-constrained prior: iteratively sample from regions of increasing likelihood. The algorithm starts by sampling N "live points" from the prior. At each iteration, it identifies the point with lowest likelihood, replaces it with a new point sampled from the prior with likelihood exceeding the removed point's, and repeats this process, shrinking the prior volume at each step. As a byproduct, this procedure naturally computes the marginal likelihood $P(D)$, making it invaluable for model comparison.


This approach proves ideal for several scenarios. When model comparison and Bayes factor computation are as important as parameter inference, nested sampling excels. It naturally handles multimodal posteriors—think phase transitions or degenerate models—by its systematic exploration of the likelihood landscape. In low-to-moderate dimensions, when evidence estimation matters as much as posterior sampling, nested sampling provides both simultaneously.


The trade-offs are worth understanding. For purely sampling tasks, nested sampling can be slower than MCMC or HMC. Its efficiency hinges critically on the "constrained prior sampling" step—drawing new samples from the prior that satisfy the likelihood constraint. The number of live points controls the accuracy-cost trade-off: more points mean better accuracy but higher computational cost. Despite these considerations, nested sampling has become increasingly popular in astronomy and cosmology where model selection is often the primary scientific question.


➡️ 
[Explore Nested Sampling](/html_src/interactive_nested_sampling.html)  

*Includes: live point evolution, evidence computation, posterior reconstruction from nested samples*

---

### MCMC Parallel Tempering

**Overcoming energy barriers via temperature ladder.**

Parallel tempering (also called replica exchange MCMC) runs multiple chains at different "temperatures"—effectively flattening the posterior at high temperatures to escape local modes, then exchanging states between chains to transfer information. The mechanism is elegant. The algorithm runs M chains targeting $[P(\theta \mid D)]^{1/T_i}$ for temperatures $T_1 = 1 < T_2 < ... < T_M$. High-temperature chains explore freely across a flattened landscape, easily hopping between modes that would trap a standard MCMC chain. Low-temperature chains sample accurately from the true posterior. The magic happens when we periodically propose swapping states between adjacent temperature chains—information from high-temperature exploration gradually informs low-temperature sampling, allowing the cold chain to discover and properly weight all modes.


The method excels at highly multimodal posteriors where standard MCMC would remain trapped in a single mode. Rugged likelihood surfaces with many local optima present no special difficulty. Phase transition problems, where the posterior splits between discrete regimes, are natural applications. Whenever you suspect your posterior has structure you haven't discovered yet, parallel tempering provides a systematic way to find it.


Several practical considerations arise. The temperature schedule requires tuning—too few temperatures and swaps fail, too many wastes computation. The computational cost scales linearly with the number of chains, though these chains can run in parallel. Swap acceptance rates provide diagnostic information about whether the temperature spacing is appropriate. When properly configured, parallel tempering transforms impossible sampling problems into tractable ones.


➡️ [Explore Parallel Tempering](/html_src/interactive_parralel_tempering.html) 

*Includes: temperature ladder visualization, swap statistics, mode discovery demos*

---

## 📚 Further Reading

For those seeking a deeper understanding, several foundational texts stand out.

- **Bayesian Data Analysis**  
  *Gelman et al.*, 3rd edition — a definitive reference that is both comprehensive and accessible.

- **Information Theory, Inference, and Learning Algorithms**  
  *David J. C. MacKay* — offers deep insights from a physics perspective, particularly resonant for astrophysicists.

- **Pattern Recognition and Machine Learning**  
  *Christopher M. Bishop* — elegantly connects Bayesian inference with modern machine learning.

For sampling-specific depth:

- **Handbook of Markov Chain Monte Carlo**  
  *Brooks et al.* — a comprehensive treatment of MCMC methods.

- **A Conceptual Introduction to Hamiltonian Monte Carlo** (2017)  
  *Michael Betancourt* — builds geometric intuition, rendering HMC both natural and inevitable.

- **Nested Sampling for General Bayesian Computation** (2006)  
  *John Skilling* — introduces nested sampling with characteristic clarity.

Modern implementations have made these methods broadly accessible:

- **[Stan](https://mc-stan.org/)**  
  Production-quality Hamiltonian Monte Carlo with automatic differentiation.

- **[PyMC](https://www.pymc.io/)**  
  Pythonic probabilistic programming with excellent documentation.

- **[Turing.jl](https://turing.ml/)**  
  Brings Julia’s performance to probabilistic inference.

Each framework has its own strengths; exploring multiple implementations can significantly deepen conceptual understanding.
