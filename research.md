---
layout: default
title: Research
---

# Research

## Gravitational-wave detectors: ground and space

Current gravitational-wave detectors — LIGO, Virgo, and KAGRA — are ground-based interferometers sensitive to frequencies between roughly 10 Hz and a few kHz. This frequency band is well suited to stellar-mass compact binaries: neutron star mergers, black hole mergers, and mixed systems. The detectors have already collected hundreds of events and established gravitational-wave astronomy as a mature observational field.

LISA operates in a fundamentally different regime. As a space-based interferometer with arm lengths of 2.5 million kilometers, it is sensitive to millihertz frequencies, a band inaccessible from the ground due to seismic and Newtonian noise. At these frequencies, LISA will observe massive binary black holes with masses between $10^4$ and $10^7$ solar masses, merging across cosmic history out to high redshift. It will also detect thousands of compact galactic binaries simultaneously, producing a rich and complex signal. LISA is scheduled for launch in the early 2030s under ESA leadership, with NASA as a partner.

The data analysis challenges for LISA are qualitatively different from those of ground-based detectors. Signals last much longer, the parameter spaces are larger, the posteriors are more often multimodal, and the sheer volume of overlapping sources demands methods that are both accurate and computationally tractable.

## PhD work: DeepHMC

My doctoral research addressed a core bottleneck in gravitational-wave parameter estimation for ground-based detectors: Bayesian inference is expensive. A single posterior for a binary neutron star merger can require millions of likelihood evaluations, each involving a waveform computation and a noise-weighted inner product. Standard Markov Chain Monte Carlo scales poorly in this setting.

DeepHMC combines Hamiltonian Monte Carlo with a neural network trained to approximate the geometry of the posterior before sampling begins. Rather than learning the posterior itself, the network learns a metric that captures local curvature, allowing the HMC sampler to take long, informed steps from the start rather than spending the early phase of sampling adapting blindly. The result is a significant reduction in the number of likelihood evaluations needed to reach convergence, with posterior quality that matches standard methods.

The framework also integrates GPU acceleration and adaptive trajectory strategies, making it viable for both offline analysis and low-latency scenarios. DeepHMC was applied to binary neutron star parameter estimation within the LVK analysis context.


## Current work: parameter estimation for LISA

My current research focuses on massive binary black holes in the LISA band. The scientific questions are compelling — these systems carry information about galaxy formation, black hole seeding mechanisms, and the large-scale structure of the universe — but the inference problem is genuinely hard.

The primary bottleneck is the likelihood. Evaluating how well a waveform template matches the data requires computing the LISA instrument response to the gravitational-wave signal, which is itself a non-trivial calculation involving the motion of the three spacecraft and the time-delay interferometry (TDI) combination that suppresses laser noise. I am developing **JaxMBHB**, a JAX library that implements this response and the time-domain likelihood for massive binary black holes in a fully differentiable, hardware-agnostic way. Working in the time domain avoids the stationarity approximations of frequency-domain approaches and keeps the physics closer to the raw data stream.

The differentiable likelihood opens the door to gradient-based samplers. I also work on parallel tempering HMC schemes designed for the multimodal posteriors that LISA sources routinely produce, where waveform symmetries generate degenerate solutions that a single-chain sampler would miss.
