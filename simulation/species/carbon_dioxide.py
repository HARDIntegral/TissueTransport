from typing import override
from .base import Species


class CarbonDioxide(Species):
	"""
	Carbon dioxide transport model.

	Notes:
		This subclass inherits diffusivity correction logic and shared species
		behavior from the parent `Species` class.

		Only carbon dioxide-specific properties are defined here:

			- molecular weight = 44.01 g/mol
			- prescribed free diffusivity

		Current implementation assumes:

			D_free = 1.6 × 10^-9 m²/s

		which approximates carbon dioxide diffusion in physiological fluids.
	"""

	def __init__(self):
		"""
		Initialize carbon dioxide-specific molecular properties.

		Notes:
			Shared initialization behavior comes from the parent `Species` class.

			Carbon dioxide defines:

				molecular_weight = 44.01 g/mol
				molecular_radius = None

			Radius is left undefined because diffusivity is prescribed directly.
		"""
		super().__init__(
			name="carbon_dioxide",
			molecular_weight=44.01,
			molecular_radius=None
		)


	@override
	def free_diffusivity(self, temperature, mu):
		"""
		Override the parent `Species.free_diffusivity()` implementation.

		Returns:
			D_free -> prescribed carbon dioxide diffusivity in m²/s

		Notes:
			Carbon dioxide currently uses a constant diffusivity:

				D_free = 1.6 × 10^-9 m²/s

			Temperature and viscosity dependence may later be added through
			Stokes-Einstein or empirical relations.
		"""
		return 1.6e-9