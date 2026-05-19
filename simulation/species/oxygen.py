from typing import override
from .base import Species

class Oxygen(Species):
	"""
	Oxygen transport model.

	Notes:
		This subclass inherits diffusivity correction logic and shared species
		behavior from the parent `Species` class.

		Only oxygen-specific properties are defined here:

			- molecular weight = 32 g/mol
			- prescribed free diffusivity

		Current implementation assumes:

			D_free = 2.0 × 10^-9 m²/s

		which approximates oxygen diffusion in physiological fluids.
	"""
	def __init__(self):
		"""
		Initialize oxygen-specific molecular properties.

		Notes:
			Shared initialization behavior comes from the parent `Species` class.

			Oxygen defines:

				molecular_weight = 32 g/mol
				molecular_radius = None

			Radius is left undefined because diffusivity is prescribed directly.
		"""
		super().__init__(
			name="oxygen",
			molecular_weight=32.00,
			molecular_radius=None
		)

	@override
	def free_diffusivity(self, temperature, mu):
		"""
		Override the parent `Species.free_diffusivity()` implementation.

		Returns:
			D_free	-> prescribed oxygen diffusivity in m²/s

		Notes:
			Unlike generic Species, oxygen currently uses a constant diffusivity:

				D_free = 2.0 × 10^-9 m²/s

			Temperature and viscosity dependence may later be added through
			Stokes-Einstein or empirical relations.
		"""
		return 2.0e-9