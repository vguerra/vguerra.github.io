---
title: "Latent Variables & Generative Models"
description: "latent variables (data lower-D than observed), generative models (simple latent prior → complex data via deep net), skeleton behind VAE/GAN/flows/diffusion"
category: "Misc ML Concepts"
order: 83
updatedDate: "2026-09-10T21:28:43.769Z"
---
Data is often **lower-dimensional** than the raw number of observed variables suggests — each example can
be described by a smaller set of underlying **latent variables**.

**Generative models** use a deep network to describe the relationship between a **low-dimensional latent
variable** and the **high-dimensional observed data**. By design, the latent variable has a **simple
probability distribution** (e.g. a standard Gaussian) that's easy to sample; the network maps samples
from that simple latent space to the complex data distribution.

This is the shared skeleton behind **VAEs** (encoder to latent + decoder back), **GANs** (generator maps
latent noise → data), **normalizing flows**, and **diffusion models** — they differ in *how* they learn
the latent↔data mapping and how they're trained, but all exploit a simple latent prior mapped to complex
data.

Connects to [[pca-svd]] (linear latent structure via principal components) and [[embeddings]] (learned
low-dimensional representations).

---

Related: [[pca-svd]], [[embeddings]], [[probability-distributions]], [[loss-functions]]
