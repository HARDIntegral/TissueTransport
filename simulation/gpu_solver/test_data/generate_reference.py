"""Generate small reference datasets from the Python solver.

The resulting JSON is used to verify that the Rust solver produces
numerically equivalent output for the same inputs.

This file includes both:

	1. the original single-species oxygen reference case
	2. a coupled oxygen/carbon dioxide gas-exchange reference case
"""

import json
from pathlib import Path

import numpy as np

from simulation.solver_reference.flux import (
	compute_flux,
	compute_flux_divergence
)


# --- Reference problem setup -------------------------------------------------
width = 100
height = 100

dx = 1.0
dy = 1.0
dt = 0.1

rng = np.random.default_rng(42)

# Random but reproducible oxygen concentration field.
oxygen_concentration = rng.uniform(
	low=0.5,
	high=10.0,
	size=(height, width)
).astype(float)

# Carbon dioxide starts lower than oxygen but still nonzero so the diffusion
# term can be validated independently of metabolic production.
carbon_dioxide_concentration = rng.uniform(
	low=0.0,
	high=2.0,
	size=(height, width)
).astype(float)

# Keep the original single-species name for backward-compatible Rust tests.
concentration = oxygen_concentration

diffusivity = rng.uniform(0.8, 1.2, size=(height, width))
vmax = np.full((height, width), 0.5)
km = np.full((height, width), 1.0)

vessel_mask = np.zeros((height, width), dtype=bool)
vessel_mask[20:25, 20:25] = True
vessel_mask[50:55, 70:75] = True
vessel_mask[80:85, 40:45] = True

vessel_concentration = np.zeros((height, width), dtype=float)
vessel_concentration[vessel_mask] = 10.0

vessel_oxygen_concentration = np.zeros((height, width), dtype=float)
vessel_oxygen_concentration[vessel_mask] = 10.0

vessel_carbon_dioxide_concentration = np.zeros((height, width), dtype=float)
vessel_carbon_dioxide_concentration[vessel_mask] = 0.0

co2_yield = 1.0


# --- Compute one explicit oxygen step (match original Rust logic exactly) -----
Jx, Jy = compute_flux(concentration, diffusivity, dx, dy)
diffusion = compute_flux_divergence(Jx, Jy, dx, dy, concentration.shape)

consumption = vmax * concentration / (km + concentration)

expected = concentration + dt * (diffusion - consumption)
expected[vessel_mask] = vessel_concentration[vessel_mask]


# --- Compute one coupled O2/CO2 gas-exchange step ----------------------------
oxygen_Jx, oxygen_Jy = compute_flux(
	oxygen_concentration,
	diffusivity,
	dx,
	dy
)
oxygen_diffusion = compute_flux_divergence(
	oxygen_Jx,
	oxygen_Jy,
	dx,
	dy,
	oxygen_concentration.shape
)

carbon_dioxide_Jx, carbon_dioxide_Jy = compute_flux(
	carbon_dioxide_concentration,
	diffusivity,
	dx,
	dy
)
carbon_dioxide_diffusion = compute_flux_divergence(
	carbon_dioxide_Jx,
	carbon_dioxide_Jy,
	dx,
	dy,
	carbon_dioxide_concentration.shape
)

oxygen_consumption = (
	vmax * oxygen_concentration / (km + oxygen_concentration)
)

expected_oxygen = (
	oxygen_concentration + dt * (oxygen_diffusion - oxygen_consumption)
)
expected_carbon_dioxide = (
	carbon_dioxide_concentration
	+ dt * (carbon_dioxide_diffusion + co2_yield * oxygen_consumption)
)

expected_oxygen[vessel_mask] = vessel_oxygen_concentration[vessel_mask]
expected_carbon_dioxide[vessel_mask] = (
	vessel_carbon_dioxide_concentration[vessel_mask]
)


# --- Save flattened reference case -------------------------------------------
reference = {
	"width": width,
	"height": height,
	"dx": dx,
	"dy": dy,
	"dt": dt,
	"concentration": concentration.flatten().tolist(),
	"oxygen_concentration": oxygen_concentration.flatten().tolist(),
	"carbon_dioxide_concentration": carbon_dioxide_concentration.flatten().tolist(),
	"diffusivity": diffusivity.flatten().tolist(),
	"vmax": vmax.flatten().tolist(),
	"km": km.flatten().tolist(),
	"vessel_mask": vessel_mask.flatten().tolist(),
	"vessel_concentration": vessel_concentration.flatten().tolist(),
	"vessel_oxygen_concentration": vessel_oxygen_concentration.flatten().tolist(),
	"vessel_carbon_dioxide_concentration": (
		vessel_carbon_dioxide_concentration.flatten().tolist()
	),
	"co2_yield": co2_yield,
	"expected": expected.flatten().tolist(),
	"expected_oxygen": expected_oxygen.flatten().tolist(),
	"expected_carbon_dioxide": expected_carbon_dioxide.flatten().tolist(),
}

output_path = Path(__file__).parent / "reference_case.json"

with open(output_path, "w") as f:
	json.dump(reference, f, indent=2)

print(f"Saved reference case to: {output_path}")