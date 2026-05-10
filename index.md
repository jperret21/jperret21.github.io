---
layout: default
title: Welcome
---

# Jules Perret

I am a PhD researcher and research engineer in gravitational-wave astrophysics at the [Astroparticle and Cosmology Laboratory (APC, CNRS)](https://apc.u-paris.fr/APC_CS/). My work revolves around one central problem: how do we extract reliable astrophysical information from gravitational-wave signals, efficiently enough to keep pace with current and future detectors?

My current work is focused on **LISA**, the ESA-NASA space interferometer scheduled for launch in the early 2035s. LISA will observe massive binary black holes across cosmic history, but its data analysis pipeline is a genuine computational challenge. I work on parameter estimation for these systems, both on the sampling side and, increasingly, on building **fast, differentiable likelihoods** that run efficiently on CPU and GPU using JAX library. I also work on the devellopment of the french globalfit pipeline develloped in France. 

## Research

My work sits at the intersection of Bayesian statistics, scientific computing, and gravitational-wave physics. The central bottleneck in gravitational-wave parameter estimation is the likelihood: evaluating it is expensive, and you need it millions of times. My current focus is on developing fast likelihood implementations for massive binary black holes in the LISA band, written in **JAX** to exploit autodifferentiation and run seamlessly on CPU and GPU.

On the inference side, I design and implement sampling algorithms (HMC variants and parallel tempering schemes) that can handle the multimodal, high-dimensional posteriors LISA will produce. During my PhD, I also worked on neural network-based approaches to accelerate sampling for ground-based detector sources.

## Projects

| Project | Description | Stack |
|---|---|---|
| **JaxMBHB** *(current)* | Fast time-domain LISA likelihood for massive binary black holes | JAX, XLA, TDI |
| **PT-HMC** | Parallel tempering HMC for multimodal posteriors on LISA sources | JAX, GPU |
| **DeepHMC** *(PhD)* | Neural metric learning to accelerate HMC for GW parameter estimation | PyTorch, CUDA |
| **[GW Event Visualizer](https://perretjules.com/GW_event_viz/)** | Interactive GWOSC catalog dashboard, auto-updated via CI | Python, GitHub Actions |
| **[Bayesian Inference Guide](bayesian_inference)** | Interactive intro to Bayesian methods and sampling algorithms | MCMC, HMC, Nested Sampling |

## Beyond the Lab

I am a member of the **Société Astronomique de Bourgogne**, where I contribute to public outreach and astronomy education. Explaining orbital mechanics to a curious twelve-year-old is a surprisingly good test of whether you actually understand it.

Outside physics, I build things: drones, UAVs, whatever flying and requires soldering and patience.

## Contact

- [GitHub](https://github.com/jperret21)
- [APC Laboratory](https://apc.u-paris.fr/APC_CS/)
- perretjules@gmail.com
