from solver_reference import compute_flux, compute_flux_divergence
from domain import TissueDomain
from species import base

def explicit_step(domain: TissueDomain, species: base, dt, reset_vessel=True):
	"""
	Advance the concentration field by one explicit finite-difference timestep.

	Parameters:
		domain			-> TissueDomain containing concentration, tissue properties,
							vessel masks, and metabolic consumption parameters
		species			-> transported species object used to compute effective diffusivity
		dt 				-> timestep size in seconds
		reset_vessel	-> if True, vessel pixels are reset to fixed concentrations
							after the update

	Returns:
		C_new -> updated concentration field after one timestep

	Notes:
		The update follows an explicit reaction-diffusion formulation:

			dC/dt = diffusion - consumption

		where diffusion is computed through:

			dC/dt = -∇·J

		and flux follows Fick's law:

			J = -D∇C

		Combining these gives:

			∂C/∂t = D∇²C - R(C)

		where:

			R(C) = oxygen consumption

		The update is performed explicitly:

			C_new = C_old + dt * dCdt

		This means numerical stability depends on:

			dt << dx² / (2D)

		for sufficiently large diffusivities.

		After updating concentration, vessel pixels may optionally be reset:

			C[vessel] = fixed vessel concentration

		This treats vessels as Dirichlet boundary/source conditions and prevents
		vascular oxygen from being depleted over time.
	"""
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

	dCdt = (
		compute_flux_divergence(
			Jx = Jx,
			Jy = Jy,
			dx = domain.dx,
			dy = domain.dy,
			shape = domain.shape
		)
		- domain.oxygen_consumption()
	)

	C_new = domain.concentration + dt * dCdt

	if reset_vessel:
		C_new[domain.vessel_mask] = domain.vessel_concentration[domain.vessel_mask]

	return C_new