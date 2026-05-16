import numpy as np 

class TissueDomain:
	def __init__(self, shape, dx=1.0, dy=1.0):
		self.shape = shape
		self.dx = dx
		self.dy = dy
		
		self.concentration = np.zeros(shape)

		self.epsilon = np.ones(shape)
		self.tau = np.ones(shape)
		self.mu = np.ones(shape)

		self.temperature = np.full(shape, 310.15)	# Kelvin

		self.vessel_mask = np.zeros(shape, dtype=bool)

	def set_uniform_properties(self, epsilon, tau, mu):
		self.epsilon[:, :] = epsilon
		self.tau[:, :] = tau
		self.mu[:, :] = mu
	
	def set_initial_concentration(self, concentration):
		self.concentration[:, :] = concentration

	def set_vessel_mask(self, mask, concentration):
		if mask.shape != self.shape:
			raise ValueError("Vessel mask must have the same shape as the tissue domain.")

		self.vessel_mask = mask.astype(bool)
		self.concentration[self.vessel_mask] = concentration

	def add_circular_vessel(self, center, radius, concentration):
		y, x = np.indices(self.shape)

		cx, cy = center
		distance_squared = (x - cx) ** 2 + (y - cy) ** 2

		mask = distance_squared <= radius ** 2

		self.vessel_mask[mask] = True
		self.concentration[mask] = concentration