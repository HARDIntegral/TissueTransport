# TissueTransport

A computational framework for simulating molecular diffusion and transport in biological tissues using finite difference methods. The project models species transport through heterogeneous tissue environments while accounting for porosity, tortuosity, effective diffusivity, and vascular source regions.

## Overview

Transport phenomena govern oxygen delivery, nutrient exchange, and drug penetration in biological tissues. This project aims to simulate these processes numerically using discretized diffusion equations and customizable tissue properties.

Current implementation includes:

- 2D diffusion simulation
- Spatially varying effective diffusivity
- Tissue porosity ($\epsilon$) effects
- Tortuosity ($\tau$) effects
- Temperature-dependent diffusivity
- Fixed vascular concentration sources
- Explicit finite difference time stepping
- Flux and flux divergence calculations
- Species-specific transport properties
- Michaelis-Menten oxygen consumption
- GIF generation for diffusion dynamics
- Parameter sensitivity analysis

Future goals:

- Blood flow and advection
- Vessel wall permeability
- 3D tissue domains
- Multi-species transport
- Reaction-diffusion systems
- GPU acceleration
- Visualization/interactive simulations
- Vessel radius-dependent oxygen delivery
- Coupled angiogenesis and hypoxia models

---

## Metabolic Oxygen Consumption

Oxygen consumption is modeled using Michaelis-Menten kinetics:

$$R(C)=\frac{V_{max}C}{K_m+C}$$

where:

- $V_{max}$: maximum oxygen consumption rate
- $K_m$: concentration at half-maximal consumption
- $C$: local oxygen concentration

The governing equation becomes:

$$\frac{\partial C}{\partial t}=D\nabla^2 C-\frac{V_{max}C}{K_m+C}$$

which creates biologically realistic steady-state oxygen gradients around vascular networks.

---

## Oxygen Diffusion Dynamics

The animation below shows oxygen diffusing outward from segmented blood vessels into surrounding tissue while metabolism continuously removes oxygen.

<table>
<tr>
<td align="center"><b>Original vessel structure</b></td>
<td align="center"><b>Oxygen diffusion over time</b></td>
</tr>
<tr>
<td>
<img src="blood_vessel_network_images/structure3.png" width="300">
</td>
<td>
<img src="oxygen_diffusion.gif" width="500">
</td>
</tr>
</table>

Observed behavior:

- High oxygen concentration near vessels
- Hypoxic regions in poorly vascularized tissue
- Emergence of steady-state concentration gradients
- Competition between diffusion and metabolic consumption

---

## Parameter Sensitivity Analysis

Steady-state oxygen distributions were compared across different Michaelis-Menten parameters.

Columns vary:

- $V_{max}$: maximum oxygen consumption

Rows vary:

- $K_m$: oxygen affinity of tissue metabolism

Increasing $V_{max}$ strengthens depletion and sharpens gradients, while changing $K_m$ modifies how strongly low-oxygen regions consume oxygen.

The purpose of these sweeps is to determine parameter regimes that produce biologically plausible oxygen penetration depths and hypoxic regions.

![Parameter sweep](parameter_sweep.png)

This is effectively a sensitivity analysis of the transport model and helps calibrate consumption parameters against expected tissue behavior.

## Mathematical Model

Diffusion is modeled using Fick's Law:

$$\mathbf{J}=-D_{\mathrm{eff}}\nabla C$$

where:

- $\mathbf{J}$: flux vector
- $D_{\mathrm{eff}}$: effective diffusivity
- $C$: concentration

Concentration evolution:

$$\frac{\partial C}{\partial t}=-\nabla\cdot\mathbf{J}$$

Effective diffusivity depends on local tissue properties:

$$D_{\mathrm{eff}}=D\frac{\epsilon}{\tau}$$

where:

- $D$: intrinsic diffusivity
- $\epsilon$: porosity
- $\tau$: tortuosity

---
