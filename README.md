# TissueTransport

A computational framework for simulating molecular diffusion and transport in biological tissues using finite difference methods. The project models species transport through heterogeneous tissue environments while accounting for porosity, tortuosity, effective diffusivity, and vascular source regions.

## Overview

Transport phenomena govern oxygen delivery, carbon dioxide removal, nutrient exchange, and drug penetration in biological tissues. This project aims to simulate these processes numerically using discretized reaction-diffusion equations, heterogeneous tissue properties, and realistic vascular architectures extracted from biological images.

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
- Physiological oxygen transport assumptions
- Coupled oxygen (O₂) and carbon dioxide (CO₂) transport
- Carbon dioxide production coupled to oxygen consumption
- Vessels acting as O₂ sources and CO₂ sinks
- Validation against CPU reference implementations
- GPU acceleration
- GIF generation for diffusion dynamics
- Parameter sensitivity analysis
- Machine learning-based vessel segmentation through the standalone VeSeg package

Future goals:

- Automated estimation of porosity and tortuosity from segmented tissue
- Blood flow and advection
- Vessel wall permeability
- Vessel radius-dependent oxygen delivery
- Hypoxia/anoxia detection
- Coupled angiogenesis and hypoxia models
- Dynamic vascular remodeling
- Hemodynamic flow coupling
- 3D tissue domains
- 3D GPU acceleration
- Adaptive mesh refinement
- Visualization and interactive simulations
- Experimental parameter fitting against biological data

---

## Dependencies and Setup

This project uses both Python and Rust. Python handles image processing, simulation setup, visualization, and benchmarking. Rust/WGPU handles the accelerated reaction-diffusion solver and exposes it back to Python through a NumPy-compatible extension module.

Required tools:

- Python 3.13 or compatible Python 3.x version
- Rust and Cargo
- A GPU supported by WGPU/Metal/Vulkan/DirectX/OpenGL backend support
- A Python virtual environment
- `maturin` for building the Rust extension into the Python environment

Python dependencies include:

- `numpy`
- `matplotlib`
- `tqdm`
- `pillow`
- `veseg` (machine learning-based vascular segmentation)

Install Python dependencies inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib tqdm pillow maturin
python -m pip install git+https://github.com/HARDIntegral/VeSeg.git
```

On macOS or Linux, the virtual environment should appear in the terminal prompt before running the project:

```bash
(.venv)
```

VeSeg is installed separately because TissueTransport uses pretrained U-Net vessel segmentation before transport simulation:

```bash
python -m pip install git+https://github.com/HARDIntegral/VeSeg.git
```

This provides:

```text
vascular image
↓
VeSeg segmentation
↓
binary vessel mask
↓
TissueTransport diffusion domain
```

---

## VeSeg Integration

TissueTransport now integrates directly with **[VeSeg](https://github.com/HARDIntegral/VeSeg)**, a standalone installable Python package for machine learning-based vascular segmentation.

Pipeline:

```text
Raw vascular image
↓
VeSeg (U-Net inference)
↓
Binary vessel mask
↓
Upscaling to simulation resolution
↓
GPU reaction-diffusion solver
↓
Coupled O₂ / CO₂ transport
```

This removes dependence on handcrafted thresholding pipelines and allows biologically realistic vessel structures extracted from microscopy or histology images to act as transport boundary conditions.

The current workflow:

```text
Microscopy / vessel image
→ VeSeg segmentation
→ Binary vascular mask
→ TissueTransport simulation domain
→ GPU accelerated O₂ / CO₂ transport
```

VeSeg masks become fixed vascular source regions inside the transport solver, allowing learned vascular geometry to directly influence emergent oxygen gradients, carbon dioxide accumulation, and hypoxic regions.

---

## Building the Rust/WGPU Python Extension

The Rust GPU solver lives inside:

```bash
simulation/gpu_solver
```

Build and install the Rust extension into the active Python virtual environment:

```bash
cd simulation/gpu_solver
maturin develop --release --features python
```

Then return to the project root:

```bash
cd ../..
```

Verify that Python can import the compiled solver:

```bash
python -c "import gpu_solver; print(gpu_solver)"
```

If the import succeeds, the Python side can call the Rust/WGPU solver through:

```python
gpu_solver.run_gas_exchange_steps_auto_numpy(...)
```

---

## Running the Simulation

From the project root, run the main simulation script:

```bash
python simulation/main.py
```

This will:

- load a vascular image and generate vessel masks using the VeSeg segmentation package
- create the tissue domain
- simulate coupled O₂ diffusion, CO₂ production, and metabolism through the Rust/WGPU solver
- treat vessels as persistent O₂ sources and CO₂ sinks
- save diffusion visualizations and final concentration maps

The main simulation uses chunked GPU execution. Instead of returning to Python every timestep, many intermediate timesteps stay inside Rust/GPU memory before a sampled frame is returned for plotting or GIF generation.

---

## Running Tests

Rust solver tests are located in:

```bash
simulation/gpu_solver/tests
```

Run all Rust tests from the GPU solver directory:

```bash
cd simulation/gpu_solver
cargo test
```

To check the Python feature build without producing a full optimized extension:

```bash
cargo check --features python
```

To check the optimized Python-enabled build:

```bash
cargo check --release --features python
```

The tests validate the CPU fallback, WGPU solver behavior, vessel reset logic, Michaelis-Menten consumption, and equivalence against reference outputs.

---

## Running Benchmarks

After building the Python extension with `maturin`, run the benchmark script from the project root:

```bash
python simulation/benchmark.py
```

The benchmark compares:

- NumPy-optimized Python reference solver
- Rust/WGPU solver called through the Python NumPy interface

The script performs one warmup run for each solver and excludes it from the reported statistics. This helps remove one-time initialization overhead from GPU pipeline setup, shader compilation, memory allocation, and caching. It then records five measured runs, calculates mean execution time, standard deviation, per-run speedups, and saves the performance plot as:

```bash
benchmark_performance.png
```

---

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

## Long-Term Vision

The long-term objective of TissueTransport is to evolve from a diffusion simulator into a GPU-accelerated computational physiology framework capable of simulating coupled gas exchange, metabolism, vascular adaptation, and biologically realistic tissue transport using vascular networks extracted directly from imaging data. The current workflow already integrates the standalone VeSeg machine learning package to segment vascular structures before transport simulation.

Planned biological extensions include angiogenesis,
dynamic vascular remodeling, and machine learning-based vessel segmentation from microscopy images.

---

## Modeling Assumptions

Current simulations assume:

- Tissue behaves as a homogeneous porous medium
- Diffusion occurs in two spatial dimensions
- Temperature remains constant during simulation
- Vessel oxygen concentration is fixed
- Blood flow and advection are neglected
- Diffusion is isotropic within local tissue regions
- Tissue metabolism follows Michaelis-Menten kinetics
- Carbon dioxide is produced proportionally to oxygen consumption
- Vascular geometry is generated through VeSeg, a pretrained U-Net segmentation package for vessel extraction

---

## Oxygen Diffusion Dynamics

The animation below shows oxygen diffusing outward from blood vessels segmented by VeSeg (U-Net based vascular segmentation) while metabolism continuously removes oxygen. VeSeg-generated masks are upscaled to simulation resolution and used as fixed vascular source regions.

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

## Coupled Oxygen and Carbon Dioxide Transport

The current solver advances oxygen and carbon dioxide simultaneously. Oxygen is consumed through Michaelis-Menten metabolism while carbon dioxide is generated as a metabolic byproduct. Vessel regions act as fixed oxygen sources and carbon dioxide sinks.

The visualization below shows steady-state coupled gas transport after long simulation times.

![Coupled O2 and CO2 transport](gas_exchange_final.png)

Observed behavior:

- Oxygen remains elevated near vessel regions and depleted in poorly perfused tissue
- Carbon dioxide accumulates where oxygen consumption is sustained
- Vascular regions remain low in CO₂ due to sink boundary conditions
- Emergent gradients arise from coupled metabolism and diffusion rather than diffusion alone

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

---

## CPU vs GPU Performance Benchmark

The Rust/WGPU implementation was benchmarked against the original NumPy-optimized Python reference solver using identical reaction-diffusion simulations on a $1000\times1000$ grid. Performance measurements were repeated across multiple runs and summarized with average execution time, standard deviation, and per-run variability.

![Benchmark comparison](benchmark_performance.png)

Observed behavior:

- The Rust/WGPU solver consistently outperformed the NumPy-based implementation
- Variability between runs remained small after initialization overhead was removed
- GPU acceleration becomes increasingly beneficial as spatial resolution increases
- Larger grids benefit from keeping timesteps and buffers resident on the GPU rather than repeatedly transferring arrays

**Benchmark note:** One preliminary warmup run was intentionally excluded from the reported statistics for each solver. This allows initialization costs such as GPU pipeline creation, shader compilation, memory allocation, and caching overhead to stabilize before measuring steady-state performance.

**Hardware note:** Benchmark results shown here were generated on a local development machine using a MacBook Pro with Apple Silicon (M5) and the integrated Apple GPU through Metal/WGPU. Absolute timings and speedups will vary across hardware, operating systems, GPU architectures, driver implementations, and backend support. These measurements are intended to demonstrate relative performance gains and scaling behavior rather than establish universal benchmark values.

---

## GPU Optimization and Validation Workflow

Performance improvements alone are not sufficient for scientific simulation. The GPU implementation was validated against the original NumPy-optimized Python reference solver throughout development to ensure numerical consistency while optimizing execution speed.

Validation process:

1. **Reference implementation**
	- The original explicit finite-difference solver was preserved as a CPU baseline (`solver_reference/`)
	- All new GPU logic was compared against this implementation rather than replacing it directly

2. **Numerical equivalence testing**
	- Identical concentration fields, diffusivity maps, vessel masks, and Michaelis-Menten parameters were supplied to both solvers
	- Output concentrations were compared after multiple timesteps
	- Center-cell concentrations and absolute error metrics were monitored during optimization

3. **Incremental optimization**
	- Initial GPU implementations suffered from repeated buffer allocation and CPU↔GPU transfer overhead
	- Persistent buffers and batched timestep execution were introduced to minimize readbacks
	- Double buffering allowed concentration fields to alternate between GPU buffers without intermediate copies

4. **Fallback verification**
	- A CPU fallback path remains available when compatible GPU hardware is unavailable
	- GPU and CPU paths share equivalent validated physics models

5. **Benchmarking under realistic workloads**
	- Benchmarks were performed on large grids ($1000\times1000$) to evaluate scaling behavior
	- Warmup runs were excluded to remove one-time initialization costs
	- Variability, standard deviation, and average speedups were recorded across repeated runs


---

## Related Projects

- **[VeSeg](https://github.com/HARDIntegral/VeSeg)**
  - Standalone machine learning package for vascular segmentation
  - Uses pretrained U-Net inference to generate binary vessel masks from microscopy or vascular images
  - Integrated into TissueTransport as the vascular preprocessing and source-region generation pipeline

---

## License

This project is distributed under the MIT License.

See the [LICENSE](LICENSE) file for additional details regarding permissions, limitations, and usage.
