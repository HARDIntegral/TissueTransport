from solver_reference import compute_flux, compute_flux_divergence
from domain import TissueDomain
from species import base


def diffusion_tendency(concentration, domain: TissueDomain, species: base):
	"""
	Compute the diffusive contribution to dC/dt for a transported species.

	Parameters:
		concentration	-> concentration field for the transported species
		domain			-> TissueDomain containing tissue transport properties
		species			-> transported species object used to compute effective diffusivity

	Returns:
		dCdt_diffusion -> diffusion-only contribution to the concentration update

	Notes:
		This helper separates the shared diffusion calculation from any
		species-specific reaction, source, or sink terms.
	"""
	D_eff = species.effective_diffusivity_grid(
		temperature = domain.temperature,
		mu_grid = domain.mu,
		epsilon_grid = domain.epsilon,
		tau_grid = domain.tau
	)

	Jx, Jy = compute_flux(
		C = concentration,
		D = D_eff,
		dx = domain.dx,
		dy = domain.dy
	)

	return compute_flux_divergence(
		Jx = Jx,
		Jy = Jy,
		dx = domain.dx,
		dy = domain.dy,
		shape = domain.shape
	)

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
	dCdt = (
		diffusion_tendency(
			concentration = domain.concentration,
			domain = domain,
			species = species
		)
		- domain.oxygen_consumption()
	)

	C_new = domain.concentration + dt * dCdt

	if reset_vessel:
		C_new[domain.vessel_mask] = domain.vessel_concentration[domain.vessel_mask]

	return C_new


def explicit_gas_exchange_step(
	domain: TissueDomain,
	oxygen_species: base,
	carbon_dioxide_species: base,
	oxygen_concentration,
	carbon_dioxide_concentration,
	dt,
	reset_vessel=True,
	co2_yield=1.0,
	vessel_oxygen_concentration=None,
	vessel_carbon_dioxide_concentration=0.0
):
	"""
	Advance coupled oxygen and carbon dioxide fields by one explicit timestep.

	Parameters:
		domain							-> TissueDomain containing tissue properties,
										vessel masks, and oxygen metabolism parameters
		oxygen_species					-> oxygen species object
		carbon_dioxide_species			-> carbon dioxide species object
		oxygen_concentration			-> current oxygen concentration field
		carbon_dioxide_concentration	-> current carbon dioxide concentration field
		dt								-> timestep size in seconds
		reset_vessel					-> if True, vessel pixels are reset after update
		co2_yield						-> CO2 produced per unit O2 consumed
		vessel_oxygen_concentration	-> optional fixed oxygen value for vessel cells
		vessel_carbon_dioxide_concentration -> fixed CO2 value for vessel cells

	Returns:
		O2_new, CO2_new -> updated oxygen and carbon dioxide concentration fields

	Notes:
		The coupled gas exchange model follows:

			∂O2/∂t  = D_O2∇²O2 - R(O2)
			∂CO2/∂t = D_CO2∇²CO2 + αR(O2)

		where R(O2) is the existing Michaelis-Menten oxygen consumption rate and
		α is the carbon dioxide yield coefficient.

		Oxygen consumption drives carbon dioxide production, so CO2 production is
		coupled to the oxygen field rather than the carbon dioxide field.
	"""
	original_concentration = domain.concentration
	domain.concentration = oxygen_concentration
	oxygen_consumption = domain.oxygen_consumption()
	domain.concentration = original_concentration

	oxygen_dCdt = (
		diffusion_tendency(
			concentration = oxygen_concentration,
			domain = domain,
			species = oxygen_species
		)
		- oxygen_consumption
	)

	carbon_dioxide_dCdt = (
		diffusion_tendency(
			concentration = carbon_dioxide_concentration,
			domain = domain,
			species = carbon_dioxide_species
		)
		+ co2_yield * oxygen_consumption
	)

	O2_new = oxygen_concentration + dt * oxygen_dCdt
	CO2_new = carbon_dioxide_concentration + dt * carbon_dioxide_dCdt

	if reset_vessel:
		if vessel_oxygen_concentration is None:
			O2_new[domain.vessel_mask] = domain.vessel_concentration[domain.vessel_mask]
		else:
			O2_new[domain.vessel_mask] = vessel_oxygen_concentration

		CO2_new[domain.vessel_mask] = vessel_carbon_dioxide_concentration

	return O2_new, CO2_new