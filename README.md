# Morphological Fingerprinting of Evaporating Colloidal Droplets

**Quantitative inference of particle concentration from post-evaporation morphology**
## Overview

The evaporation of a sessile colloidal droplet involves coupled **capillary, evaporative, and solutal transport processes**. These processes determine how particles redistribute during drying and ultimately shape the morphology of the dried deposit.

This project investigates a simple question:

> **Can the morphology of a dried colloidal deposit retain enough information about the droplet's transport history to quantitatively infer its particle concentration?**

We study evaporating polystyrene colloidal droplets containing varying glycerol concentrations and use spatially resolved microscopy to extract quantitative morphological descriptors from the resulting deposits.

The broader goal is to investigate **dried colloidal deposits as quantitative physical fingerprints of evaporation-driven transport** and develop a framework for physics-informed inverse modelling.

---

## Research Question

Most dried-droplet studies characterize deposition patterns qualitatively, for example as coffee-ring, uniform, mixed, or centrally deposited structures.

Here, we instead treat the final morphology as a **high-dimensional measurement** of the underlying drying process.

The central hypothesis is:

$$
\text{Transport history}
\longrightarrow
\text{particle redistribution}
\longrightarrow
\text{final morphology}
\longrightarrow
\text{inferred concentration}
$$

If the mapping is sufficiently reproducible, the dried deposit can act as a **morphological fingerprint** of the original droplet.

---

## Experimental System

The experiments use:

* **Polystyrene colloidal particles**
* **Glycerol-water mixtures**
* **Glass substrates**
* Sessile droplets undergoing controlled evaporation
* Spatially resolved optical microscopy

The resulting deposits are divided into spatial regions to capture both radial and directional information.

---

## Morphological Representation

Each dried deposit is represented using a **40-dimensional feature vector**.

The representation combines measurements from:

* Center-region descriptors
* North
* South
* East
* West

The extracted features characterize several aspects of deposit morphology, including:

* Radial deposition
* Texture
* Cracking
* Clustering
* Anisotropy
* Structural heterogeneity

Examples of physically interpretable descriptors include:

* Ring width
* Profile variance
* Crack density
* Grey-level co-occurrence statistics

The objective is not simply to maximize predictive accuracy, but to identify morphological observables that may retain information about the underlying transport dynamics.

---

## Machine Learning

A **cosine-similarity K-nearest-neighbour (KNN)** model is used for inverse prediction.

The model maps:

$$
\mathbf{M}
=
(M_1,M_2,\ldots,M_{40})
$$

to an estimated particle concentration:

$$
\hat{C}=f(\mathbf{M})
$$

where \(\mathbf{M}\) is the morphological fingerprint of the dried deposit.

### Why KNN?

The initial objective is to investigate whether samples with similar morphologies correspond to similar concentrations, rather than immediately imposing a complex parametric model.

This makes similarity-based inference useful as a baseline for studying the structure of the morphological feature space.

---

## Validation Strategy

A major focus of the project is preventing information leakage between experimentally related samples.

Instead of randomly splitting individual images, we use:

### Leave-One-Experimental-Group-Out (LOEGO)

The dataset contains **32 independent experimental batches**.

For each validation iteration:

1. One experimental batch is completely held out.
2. The model is trained using the remaining batches.
3. The held-out batch is used for testing.
4. The process is repeated across all experimental groups.

This evaluates whether morphological fingerprints generalize to **previously unseen experimental batches**, rather than merely predicting images that resemble images already seen during training.

---

## Current Results

Using the current 40-dimensional morphological representation and cosine-similarity KNN:

| Metric                    | Result |
| ------------------------- | -----: |
| LOEGO \(R^2\)             |  ~0.70 |
| LOEGO log₁₀-space \(R^2\) |  ~0.81 |
| Experimental groups       |     32 |
| Morphological features    |     40 |

The prediction quality is not uniform across concentration regimes.

In particular, **low-concentration samples exhibit larger relative errors**, suggesting limitations associated with measurement sensitivity and morphological degeneracy.

This regime is therefore treated as an important part of the physical interpretation rather than being excluded from the analysis.

---

## From Morphology to Physics

The machine-learning model provides an inverse mapping:

$$
\text{morphology}
\rightarrow
\text{concentration}
$$

The next stage is to investigate the forward physical process:

$$
\text{evaporation}
\rightarrow
\text{transport}
\rightarrow
\text{particle redistribution}
\rightarrow
\text{morphology}
$$

A time-dependent morphological representation,

$$
\mathbf{M}(t),
$$

is being explored to connect the final deposit to the transient processes occurring during evaporation.

The goal is to determine whether changes in the morphological feature vector can be interpreted in terms of evaporative and particle-transport dynamics.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── features/
│   ├── extraction/
│   └── analysis/
│
├── models/
│   ├── knn/
│   └── validation/
│
├── notebooks/
│   ├── exploratory_analysis/
│   ├── feature_analysis/
│   └── model_evaluation/
│
├── figures/
│
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── modelling/
│   └── evaluation/
│
├── requirements.txt
└── README.md
```

---

## Reproducibility

The analysis is designed around experimental-group-aware validation.

When reproducing the machine-learning results, avoid random image-level train/test splitting because images originating from the same experimental batch can contain correlated information.

The recommended evaluation protocol is therefore:

```text
Experimental batch
        │
        ▼
Feature extraction
        │
        ▼
40D morphological fingerprint
        │
        ▼
LOEGO split
        │
        ├── Training groups
        │
        └── Held-out group
                │
                ▼
          Cosine KNN
                │
                ▼
      Predicted concentration
```

---

## Scientific Context

The project builds upon established work on:

* Coffee-ring formation
* Evaporation of sessile droplets
* Colloidal transport
* Capillary-driven flows
* Marangoni effects
* Particle deposition
* Morphological characterization of dried droplets

Key references include work by Deegan *et al.*, Hu & Larson, Yunker *et al.*, Sefiane, and others.

See the project manuscript/abstract for the complete reference list.

---

## Current Limitations

The current study has several limitations:

* Morphology-to-concentration mapping does not uniquely establish the underlying physical mechanism.
* Prediction performance decreases in low-concentration regimes.
* The current feature representation is descriptive rather than a complete mechanistic model.
* The relationship between transient transport and final morphology requires further quantitative investigation.
* The present ML model is primarily an inverse inference tool rather than a complete predictive model of droplet evaporation.

These limitations motivate the ongoing physics-informed analysis.

---

## Future Work

Planned directions include:

* Quantitative analysis of the evolving morphological state \(\mathbf{M}(t)\)
* Connecting individual morphological descriptors to transport mechanisms
* Physics-informed inverse modelling
* Investigation of morphological degeneracy
* Improved treatment of low-concentration samples
* Comparison with alternative regression and similarity-based models
* Synthetic evaporation trajectories informed by existing droplet-transport literature

---

## Authors

**Jaya Aditya[1],**
**Vishal Singh,**
**Dr. Manigandan Sabapathy**






Indian Institute of Science Education and Research,Trivandrum[1]



Indian Institute of Technology Ropar, Punjab, India

---

## Citation

If you use this repository or the associated analysis, please cite the corresponding research work.
