import time
import numpy as np

from simulation.solver.flux import (
    compute_flux,
    compute_flux_divergence
)


def explicit_step(
    concentration,
    diffusivity,
    vmax,
    km,
    vessel_mask,
    vessel_concentration,
    dt,
    dx,
    dy
):
    Jx, Jy = compute_flux(
        concentration,
        diffusivity,
        dx,
        dy
    )

    diffusion = compute_flux_divergence(
        Jx,
        Jy,
        dx,
        dy,
        concentration.shape
    )

    consumption = (
        vmax *
        concentration /
        (km + concentration)
    )

    concentration_new = (
        concentration +
        dt * (diffusion - consumption)
    )

    concentration_new[vessel_mask] = (
        vessel_concentration[vessel_mask]
    )

    return concentration_new


width = 250
height = 250
steps = 1000

dx = 1.0
dy = 1.0
dt = 0.1

concentration = np.ones(
    (height, width),
    dtype=np.float32
)

diffusivity = np.ones_like(
    concentration
)

vmax = np.full_like(
    concentration,
    0.01
)

km = np.ones_like(
    concentration
)

vessel_mask = np.zeros_like(
    concentration,
    dtype=bool
)

vessel_concentration = np.zeros_like(
    concentration
)


start = time.perf_counter()

for _ in range(steps):
    concentration = explicit_step(
        concentration,
        diffusivity,
        vmax,
        km,
        vessel_mask,
        vessel_concentration,
        dt,
        dx,
        dy
    )

elapsed = time.perf_counter() - start


print(f"Grid: {width}x{height}")
print(f"Steps: {steps}")
print(f"Elapsed: {elapsed:.3f} s")
print(
    f"Time per step: "
    f"{elapsed/steps*1e6:.3f} µs"
)