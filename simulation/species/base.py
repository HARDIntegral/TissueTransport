from abc import ABC, abstractmethod

class Species(ABC):
	"""
	Abstract base class representing a transported molecular species.

	Parameters:
		name				-> name of the species (e.g. Oxygen)
		molecular_weight	-> molecular weight of the species
		molecular_radius	-> effective molecular radius in meters

	Attributes:
		name				-> species identifier
		molecular_weight	-> molecular weight used for physical calculations
		molecular_radius	-> molecular size used in diffusivity models
		diffusivity			-> optional cached diffusivity value

	Notes:
		Each transported species should implement its own free diffusivity model.
		Examples include:

			- Oxygen in plasma
			- Glucose in tissue
			- VEGF signaling molecules
			- Drugs or nanoparticles

		The species class separates molecular physics from tissue physics.
		Tissue properties such as porosity and tortuosity are handled elsewhere.
	"""

	def __init__(self, name, molecular_weight, molecular_radius=None):
		"""
		Initialize a transported species.

		Parameters:
			name -> species name
			molecular_weight -> molecular weight
			molecular_radius -> effective molecular radius
		"""
		self.name = name
		self.molecular_weight = molecular_weight
		self.molecular_radius = molecular_radius

		self.diffusivity = None

	def free_diffusivity(self, temperature, mu):
		"""
		Compute the free diffusivity of the species.

		Parameters:
			temperature	-> local temperature in Kelvin
			mu			-> fluid viscosity

		Returns:
			D_free	-> diffusivity before tissue corrections

		Notes:
			This should be implemented by subclasses.

			Examples:

				Stokes-Einstein:

				D = kT / (6πμr)

			or empirical diffusivity relations.

			The returned diffusivity assumes transport in an unobstructed medium.
		"""
		raise NotImplementedError

	def effective_diffusivity_grid(self, temperature, mu_grid, epsilon_grid, tau_grid):
		"""
		Compute spatially varying effective diffusivity across tissue.

		Parameters:
			temperature		-> temperature field or scalar temperature
			mu_grid			-> viscosity field
			epsilon_grid	-> porosity field
			tau_grid	-> tortuosity field

		Returns:
			D_eff	-> effective diffusivity at every grid cell

		Notes:
			Free diffusivity is first computed:

				D_free = free_diffusivity(T, μ)

			Then tissue effects are applied:

				D_eff = D_free * ε / τ

			Higher porosity increases diffusion because more transport volume exists.
			Higher tortuosity decreases diffusion because molecules travel longer,
			more obstructed paths.

			The resulting D_eff field is later used by the flux calculations:

				J = -D_eff ∇C
		"""
		D_free = self.free_diffusivity(temperature, mu_grid)
		return D_free * epsilon_grid / tau_grid