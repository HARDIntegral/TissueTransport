from abc import ABC, abstractmethod

class Species(ABC):
	def __init__(self, name, molecular_weight, molecular_radius=None):
		self.name = name
		self.molecular_weight = molecular_weight
		self.molecular_radius = molecular_radius

		self.diffusivity = None

	def free_diffusivity(self, temperature, mu):
		raise NotImplementedError

	def effective_diffusivity_grid(self, temperature, mu_grid, epsilon_grid, tau_grid):
		D_free = self.free_diffusivity(temperature, mu_grid)
		return D_free * epsilon_grid / tau_grid