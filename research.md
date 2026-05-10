---
layout: default
title: Research
---

# Research

## Gravitational-wave detectors: ground and space

Current gravitational-wave detectors — LIGO, Virgo, and KAGRA — are ground-based interferometers sensitive to frequencies between roughly 10 Hz and a few kHz. This frequency band is well suited to stellar-mass compact binaries: neutron star mergers, black hole mergers, and mixed systems. The detectors have already collected hundreds of events and established gravitational-wave astronomy as a mature observational field.

LISA operates in a fundamentally different regime. As a space-based interferometer with arm lengths of 2.5 million kilometers, it is sensitive to millihertz frequencies, a band inaccessible from the ground due to seismic and Newtonian noise. At these frequencies, LISA will observe massive binary black holes with masses between $10^4$ and $10^7$ solar masses, merging across cosmic history out to high redshift. It will also detect thousands of compact galactic binaries simultaneously, producing a rich and complex signal. LISA is scheduled for launch in the early 2030s under ESA leadership, with NASA as a partner.

<figure style="margin: 2rem 0; text-align: center;">
  <img src="/assets/media/research/gw_spectrum.png" alt="Gravitational-wave frequency spectrum showing sensitivity curves for current and future detectors alongside their expected sources" style="width: 100%; max-width: 760px; border-radius: 4px;">
  <figcaption style="margin-top: 0.75rem; font-size: 0.85rem; color: #718096; line-height: 1.5;">
    The gravitational-wave spectrum: characteristic strain as a function of frequency for current and future detectors, overlaid with the expected signal families each observatory targets. Ground-based interferometers (LIGO, Virgo, Einstein Telescope) cover the high-frequency band, while LISA operates at millihertz frequencies where massive black hole binaries and galactic compact binaries dominate. Credit: <a href="https://gwplotter.com/" style="color: #718096;">gwplotter.com</a>.
  </figcaption>
</figure>

The data analysis challenges for LISA are qualitatively different from those of ground-based detectors. Signals last much longer, the parameter spaces are larger, the posteriors are more often multimodal, and the sheer volume of overlapping sources demands methods that are both accurate and computationally tractable.

## Current work: parameter estimation for LISA

My current research focuses on massive binary black holes in the LISA band. The scientific questions are compelling: these systems carry information about galaxy formation, black hole seeding mechanisms, and the large-scale structure of the universe. The goal is Bayesian parameter estimation, recovering the posterior distribution over source parameters given the LISA data stream. This means repeatedly evaluating how well a waveform template, characterized by masses, spins, sky position, and redshift, matches the observed signal, across a parameter space of 15 or more dimensions.

The primary bottleneck is the likelihood. Evaluating how well a waveform template matches the data requires computing the LISA instrument response to the gravitational-wave signal, which is itself a non-trivial calculation involving the motion of the three spacecraft and the time-delay interferometry (TDI) combination that suppresses laser noise. I am developing **JaxMBHB**, a JAX library that implements this response and the frequency-domain likelihood for massive binary black holes in a fully differentiable, hardware-agnostic way. Working in the time domain avoids the stationarity approximations of frequency-domain approaches and keeps the physics closer to the raw data stream.

Because the likelihood is fully differentiable, its gradient with respect to all source parameters is available at no extra implementation cost via JAX autodiff. This makes Hamiltonian Monte Carlo a natural fit: the sampler can follow the geometry of the posterior rather than exploring it blindly. I work on parallel tempering HMC schemes that exploit this gradient while handling the multimodal posteriors that LISA sources routinely produce, where waveform symmetries generate degenerate solutions that a single-chain sampler would miss.

## PhD work: DeepHMC

My doctoral research addressed a core bottleneck in gravitational-wave parameter estimation for ground-based detectors: Bayesian inference is expensive. A single posterior for a binary neutron star merger can require millions of likelihood evaluations, each involving a waveform computation and a noise-weighted inner product. Standard Markov Chain Monte Carlo scales poorly in this setting.

DeepHMC combines Hamiltonian Monte Carlo with a neural network trained to approximate the geometry of the posterior before sampling begins. Rather than learning the posterior itself, the network learns a metric that captures local curvature, allowing the HMC sampler to take long, informed steps from the start rather than spending the early phase of sampling adapting blindly. The result is a significant reduction in the number of likelihood evaluations needed to reach convergence, with posterior quality that matches standard methods.

The framework also integrates GPU acceleration and adaptive trajectory strategies, making it viable for both offline analysis and low-latency scenarios. DeepHMC was applied to binary neutron star parameter estimation within the LVK analysis context.


