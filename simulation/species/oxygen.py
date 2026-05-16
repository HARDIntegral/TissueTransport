from typing import override
from .base import Species

class Oxygen(Species):
	def __init__(self):
		super().__init__(
			name="oxygen",
			molecular_weight=32.00,
			molecular_radius=None
		)

	@override
	def free_diffusivity(self, temperature, mu):
		return 1.0	# placeholder