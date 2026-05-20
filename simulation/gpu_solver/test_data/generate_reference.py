"""Generate a small reference dataset from the Python solver.

The resulting JSON is used to verify that the Rust solver produces
numerically equivalent output for the same inputs.
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

# Random but reproducible concentration field.
concentration = rng.uniform(
	low=0.5,
	high=10.0,
	size=(height, width)
).astype(float)

diffusivity = rng.uniform(0.8, 1.2, size=(height, width))
vmax = np.full((height, width), 0.5)
km = np.full((height, width), 1.0)

vessel_mask = np.zeros((height, width), dtype=bool)
vessel_mask[20:25, 20:25] = True
vessel_mask[50:55, 70:75] = True
vessel_mask[80:85, 40:45] = True

vessel_concentration = np.zeros((height, width), dtype=float)
vessel_concentration[vessel_mask] = 10.0


# --- Compute one explicit step (match Rust logic exactly) --------------------
Jx, Jy = compute_flux(concentration, diffusivity, dx, dy)
diffusion = compute_flux_divergence(Jx, Jy, dx, dy, concentration.shape)

consumption = vmax * concentration / (km + concentration)

expected = concentration + dt * (diffusion - consumption)
expected[vessel_mask] = vessel_concentration[vessel_mask]


# --- Save flattened reference case -------------------------------------------
reference = {
	"width": width,
	"height": height,
	"dx": dx,
	"dy": dy,
	"dt": dt,
	"concentration": concentration.flatten().tolist(),
	"diffusivity": diffusivity.flatten().tolist(),
	"vmax": vmax.flatten().tolist(),
	"km": km.flatten().tolist(),
	"vessel_mask": vessel_mask.flatten().tolist(),
	"vessel_concentration": vessel_concentration.flatten().tolist(),
	"expected": expected.flatten().tolist(),
}

output_path = Path(__file__).parent / "reference_case.json"

with open(output_path, "w") as f:
	json.dump(reference, f, indent=2)

print(f"Saved reference case to: {output_path}")