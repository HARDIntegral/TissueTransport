# TissueTransport

A computational framework for simulating molecular diffusion and transport in biological tissues using finite difference methods. The project models species transport through heterogeneous tissue environments while accounting for porosity, tortuosity, effective diffusivity, and vascular source regions.

## Overview

Transport phenomena govern oxygen delivery, nutrient exchange, and drug penetration in biological tissues. This project aims to simulate these processes numerically using discretized diffusion equations and customizable tissue properties.

Current implementation includes:

- 2D diffusion simulation
- Spatially varying effective diffusivity
- Tissue porosity (`ε`) effects
- Tortuosity (`τ`) effects
- Temperature-dependent diffusivity
- Fixed vascular concentration sources
- Explicit finite difference time stepping
- Flux and flux divergence calculations
- Species-specific transport properties

Future goals:

- Blood flow and advection
- Vessel wall permeability
- Oxygen consumption kinetics
- Michaelis-Menten metabolism
- 3D tissue domains
- Multi-species transport
- Reaction-diffusion systems
- GPU acceleration
- Visualization/interactive simulations

---

## Mathematical Model

Diffusion is modeled using Fick's Law:

\[
\mathbf{J} = -D_{\text{eff}} \nabla C
\]

where:

- \( \mathbf{J} \): flux vector
- \( D_{\text{eff}} \): effective diffusivity
- \( C \): concentration

Concentration evolution:

\[
\frac{\partial C}{\partial t}
=
-\nabla \cdot \mathbf{J}
\]

Effective diffusivity depends on local tissue properties:

\[
D_{\text{eff}}
=
D
\frac{\epsilon}{\tau}
\]

where:

- \( D \): intrinsic diffusivity
- \( \epsilon \): porosity
- \( \tau \): tortuosity

---
