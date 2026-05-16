from solver import compute_flux, compute_flux_divergence
from domain import TissueDomain
from species import base

def explicit_step(domain: TissueDomain, species: base, dt, reset_vessel=True, vessel_C=1.0):
	D_eff = species.effective_diffusivity_grid(
		temperature = domain.temperature,
		mu_grid = domain.mu,
		epsilon_grid = domain.epsilon,
		tau_grid = domain.tau
	)

	Jx, Jy = compute_flux(
		C = domain.concentration,
		D = D_eff,
		dx = domain.dx,
		dy = domain.dy
	)

	dCdt = compute_flux_divergence(
		Jx = Jx,
		Jy = Jy,
		dx = domain.dx,
		dy = domain.dy,
		shape = domain.shape
	)

	C_new = domain.concentration + dt * dCdt

	if reset_vessel:
		C_new[domain.vessel_mask] = vessel_C

	return C_new