import numpy as np 

class TissueDomain:
	"""
	Represents the physical tissue domain used by the diffusion solver.

	Parameters:
		shape	-> shape of the concentration grid as (rows, columns)
		scale	-> physical size of the domain as (height_m, width_m)

	Attributes:
		dx						-> physical width of one grid cell in meters
		dy						-> physical height of one grid cell in meters
		concentration			-> local oxygen concentration field
		epsilon					-> local tissue porosity field
		tau						-> local tissue tortuosity field
		mu						-> local dynamic viscosity field
		temperature				-> local temperature field in Kelvin
		vessel_mas				-> boolean mask identifying vessel pixels
		vessel_concentration	-> concentration values assigned to vessel pixels
		consumption_vmax		-> Michaelis-Menten maximum consumption field
		consumption_km			-> Michaelis-Menten half-saturation concentration field

	Notes:
		The TissueDomain stores all spatially varying fields needed by the
		transport solver. The solver should not need to know how the tissue was
		created, only the local properties at each grid cell.

		The scale argument converts image-space pixels into physical distances:

			dx = width_m / number_of_columns
			dy = height_m / number_of_rows

		This makes the finite difference solver physically meaningful because
		diffusion depends on dx and dy through terms like:

			d2C/dx2 and d2C/dy2

		Smaller dx and dy represent a finer physical grid and require smaller
		time steps for explicit numerical stability.
	"""

	def __init__(self, shape, scale):
		"""
		Initialize the tissue domain and allocate all spatial fields.

		Parameters:
			shape	-> shape of the domain as (rows, columns)
			scale	-> physical size of the tissue region as (height_m, width_m)

		Notes:
			The first dimension of shape corresponds to rows, which is the y-direction.
			The second dimension corresponds to columns, which is the x-direction.

			Therefore:

				dy = height_m / rows
				dx = width_m / columns

			All arrays are initialized with the same shape so that every grid cell can
			store its own concentration, tissue properties, vessel state, and metabolic
			parameters.
		"""
		self.shape = shape
		self.width_m = scale[1]
		self.height_m = scale[0]
		
		self.dx = scale[1] / shape [1]
		self.dy = scale[0] / shape [0]
		
		self.concentration = np.zeros(shape)

		self.epsilon = np.ones(shape)
		self.tau = np.ones(shape)
		self.mu = np.ones(shape)

		self.temperature = np.full(shape, 310.15)  # Kelvin

		self.vessel_mask = np.zeros(shape, dtype=bool)
		self.vessel_concentration = np.zeros(shape)

		self.consumption_vmax = np.zeros(shape)
		self.consumption_km = np.ones(shape)

	def set_uniform_properties(self, epsilon, tau, mu):
		"""
		Assign uniform tissue transport properties across the full domain.

		Parameters:
			epsilon	-> tissue porosity, representing available transport volume
			tau		-> tissue tortuosity, representing path-length obstruction
			mu		-> dynamic viscosity of the local fluid environment

		Notes:
			These values are stored as fields even when they are uniform so that the
			model can later support heterogeneous tissue properties.

			Porosity and tortuosity are usually combined into an effective diffusivity:

				D_eff = D_free * epsilon / tau

			Higher porosity increases effective diffusion because more space is
			available for transport. Higher tortuosity decreases effective diffusion
			because molecules must travel through more obstructed paths.
		"""
		self.epsilon[:, :] = epsilon
		self.tau[:, :] = tau
		self.mu[:, :] = mu
	
	def set_initial_concentration(self, concentration):
		"""
		Set the initial concentration everywhere in the tissue domain.

		Parameters:
			concentration	-> starting concentration assigned to every grid cell

		Notes:
			This initializes the tissue before vessel concentrations are applied.
			For normalized simulations, concentration usually ranges from 0 to 1:

				0 -> no oxygen
				1 -> vessel/source oxygen level

			If physical concentration units are used later, this same field can store
			values such as mol/m^3 instead.
		"""
		self.concentration[:, :] = concentration

	def set_vessel_mask(self, mask, concentration):
		"""
		Assign vessel locations and set their oxygen concentration.

		Parameters:
			mask	-> boolean array where True values represent vessel pixels
			concentration	-> oxygen concentration assigned inside vessel pixels

		Raises:
			ValueError	-> if the vessel mask shape does not match the tissue shape

		Notes:
			The vessel mask defines where vascular oxygen sources exist in the image.
			Those pixels are assigned the supplied concentration immediately.

			Consumption is disabled inside vessel pixels by setting:

				consumption_vmax = 0

			This prevents the model from treating blood vessels like metabolically
			active tissue. Surrounding tissue consumes oxygen, but vessel pixels act
			as sources or reservoirs depending on how the solver handles them.
		"""
		if mask.shape != self.shape:
			raise ValueError("Vessel mask must have the same shape as the tissue domain.")

		self.vessel_mask = mask.astype(bool)
		self.vessel_concentration[self.vessel_mask] = concentration
		self.concentration[self.vessel_mask] = concentration
		self.consumption_vmax[self.vessel_mask] = 0.0

	def set_uniform_consumption(self, vmax, km=0.05):
		"""
		Assign uniform Michaelis-Menten oxygen consumption parameters.

		Parameters:
			vmax	> maximum oxygen consumption rate
			km		-> concentration where consumption reaches half of vmax

		Notes:
			Oxygen consumption is modeled as:

				R(C) = vmax * C / (km + C)

			When concentration is normalized from 0 to 1:

				vmax has units of 1/s
				km is dimensionless

			When concentration is physical, such as mol/m^3:

				vmax has units of concentration/s
				km has the same units as concentration

			At low oxygen concentration, where C << km:

				R(C) approx (vmax / km) * C

			so consumption behaves almost linearly.

			At high oxygen concentration, where C >> km:

				R(C) approx vmax

			so consumption saturates near its maximum value.

			Vessel pixels are forced to have zero vmax so that oxygen is not consumed
			inside the vascular mask.
		"""
		self.consumption_vmax[:, :] = vmax
		self.consumption_km[:, :] = km
		self.consumption_vmax[self.vessel_mask] = 0.0

	def oxygen_consumption(self):
		"""
		Compute the local oxygen consumption rate field.

		Returns:
			consumption	-> array containing oxygen consumed per unit time at each cell

		Notes:
			The local metabolic consumption rate is calculated using Michaelis-Menten
			kinetics:

				R(C) = vmax * C / (km + C)

			Concentration is clipped below at zero before evaluating the expression:

				C = max(C, 0)

			This prevents negative concentrations from producing nonphysical negative
			consumption rates.

			After the consumption field is computed, vessel pixels are set to zero:

				R(C) = 0 inside vessels

			The resulting field is subtracted from the diffusion update in the solver:

				dC/dt = diffusion - consumption

			This creates a reaction-diffusion model where oxygen is supplied by vessels,
			spreads through tissue, and is removed by cellular metabolism.
		"""
		C = np.maximum(self.concentration, 0.0)
		consumption = self.consumption_vmax * C / (self.consumption_km + C)
		consumption[self.vessel_mask] = 0.0
		return consumption
