---
layout: default
title: Welcome
---

# Jules Perret

I am a PhD researcher and research engineer in gravitational-wave astrophysics at the [Astroparticle and Cosmology Laboratory (APC, CNRS)](https://apc.u-paris.fr/APC_CS/). My work revolves around one central problem: how do we extract reliable astrophysical information from gravitational-wave signals, efficiently enough to keep pace with current and future detectors?

My current work is focused on **LISA**, the ESA-NASA space interferometer scheduled for launch in the early 2030s. LISA will observe massive binary black holes across cosmic history, but its data analysis pipeline is a genuine computational challenge. I work on parameter estimation for these systems, both on the sampling side and, increasingly, on building **fast, differentiable likelihoods** that run efficiently on CPU and GPU using JAX.

## Research

My work sits at the intersection of Bayesian statistics, scientific computing, and gravitational-wave physics. The central bottleneck in gravitational-wave parameter estimation is the likelihood — evaluating it is expensive, and you need it millions of times. My current focus is on developing fast likelihood implementations for massive binary black holes in the LISA band, written in **JAX** to exploit autodifferentiation and run seamlessly on CPU and GPU.

On the inference side, I design and implement sampling algorithms — HMC variants and parallel tempering schemes — that can handle the multimodal, high-dimensional posteriors LISA will produce. During my PhD, I also worked on neural network-based approaches to accelerate sampling for ground-based detector sources.

## Projects

### DeepHMC 
A Hamiltonian Monte Carlo sampler that uses a trained neural network to learn the geometry of the posterior before sampling. The idea is to precompute a good metric so that the sampler explores efficiently from the start, reducing the wall-clock time for gravitational-wave parameter estimation significantly.

### JaxMBHB 
A JAX library for computing the LISA instrument response and the time-domain likelihood for massive binary black holes. The time-domain formulation avoids the approximations of frequency-domain approaches, and the JAX backend makes the likelihood fully differentiable and hardware-agnostic, which is a prerequisite for gradient-based samplers at scale.

### Parallel Tempering HMC
A parallel tempering extension of HMC, implemented in JAX, designed for multimodal posteriors — the kind LISA will routinely produce for massive binary black holes. The JAX backend gives us autodiff for the Hamiltonian dynamics and makes GPU scaling straightforward.

### GW Event Visualizer
An interactive dashboard pulling live data from the Gravitational Wave Open Science Center (GWOSC). It displays the full catalog of detected events with scatter plots of component masses, SNR-weighted markers, KDE mass distributions, and per-event detail views. The catalog updates automatically via a daily GitHub Actions pipeline. [Explore the dashboard](https://perretjules.com/GW_event_viz/)

### Bayesian Inference — An Interactive Introduction
A set of educational materials I put together to make Bayesian methods more accessible:
- **[Introduction to Bayesian Inference](bayesian_inference)** — From Bayes' theorem to posterior sampling
  - [MCMC Sampler](/html_src/interactive_mcmc.html) — Metropolis-Hastings, visually
  - [HMC Sampler](/html_src/interactive_hmc.html) — Hamiltonian dynamics, interactively

## Beyond the Lab

I am a member of the **Société Astronomique de Bourgogne**, where I contribute to public outreach and astronomy education. Explaining orbital mechanics to a curious twelve-year-old is a surprisingly good test of whether you actually understand it.

Outside physics, I build things — drones, UAVs, whatever requires soldering and patience. It is a useful counterweight to work that lives entirely in abstract probability spaces.

## Contact

- [GitHub](https://github.com/jperret21) — open-source code and projects
- [APC Laboratory](https://apc.u-paris.fr/APC_CS/) — research group
