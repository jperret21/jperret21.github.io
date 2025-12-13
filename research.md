---
layout: default
title: Research
---

# Research summary

My research sits at the interface of Bayesian inference, deep learning, and high-performance computing, with applications to gravitational wave astrophysics. I focus on developing scalable and statistically rigorous methods for parameter estimation in high-dimensional, computationally expensive models — particularly in the context of compact binary coalescences such as binary neutron star mergers.

As part of my PhD, I have developed DeepHMC, a custom inference framework that combines Hamiltonian Monte Carlo (HMC) with deep neural networks trained to approximate gradients of the log-posterior. This hybrid approach enables fast and accurate exploration of complex posterior distributions, even in cases where traditional sampling methods struggle due to costly or non-differentiable models. DeepHMC integrates adaptive trajectory strategies, on-the-fly diagnostics, and efficient GPU acceleration, making it suitable for both offline and low-latency inference scenarios in gravitational wave data analysis.

While DeepHMC is the core focus of my doctoral research, my broader interests extend well beyond it. I work on incorporating deep learning techniques such as surrogate modeling and representation learning to support probabilistic inference under uncertainty. I also apply parallel and GPU-accelerated computing to scale these methods to large simulations and high-throughput inference pipelines.

Overall, my work aims to connect modern machine learning tools with physically motivated Bayesian modeling, to enable faster and more interpretable data analysis in astrophysics and other scientific domains.
