---
layout: post
title: "HMC from scratch in JAX"
date: 2026-05-01
---

*Coming soon.*

This post will walk through a minimal implementation of Hamiltonian Monte Carlo in JAX, from the leapfrog integrator to a working sampler on a toy posterior. The goal is to show how JAX's autodiff and JIT compilation make HMC both clean to implement and fast to run on CPU and GPU.
