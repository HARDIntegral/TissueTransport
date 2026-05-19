

"""Generate a small reference dataset from the Python solver.

The resulting JSON is used to verify that the Rust solver produces
numerically equivalent output for the same inputs.
"""

import json
from pathlib import Path

import numpy as np

from simulation.solver.flux import (
    compute_flux,
    compute_flux_divergence
)


# --- Reference problem setup -------------------------------------------------
width = 3
height = 3

dx = 1.0
dy = 1.0
dt = 0.1

concentration = np.array([
    [5.0, 5.0, 5.0],
    [5.0, 1.0, 5.0],
    [5.0, 5.0, 5.0],
], dtype=float)

diffusivity = np.ones((height, width), dtype=float)
vmax = np.full((height, width), 0.5)
km = np.full((height, width), 1.0)

vessel_mask = np.zeros((height, width), dtype=bool)
vessel_mask[1, 1] = True

vessel_concentration = np.zeros((height, width), dtype=float)
vessel_concentration[1, 1] = 10.0


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