

"""
Physical constants and default parameters for vascular transport.

Units:
- Length: micrometers (µm)
- Pressure: Pascals (Pa)
- Time: seconds (s)
- Concentration: mol/m^3 unless otherwise noted
"""

# Blood properties
PLASMA_VISCOSITY = 1.2e-3			# Pa·s
BLOOD_DENSITY = 1060.0				# kg/m^3
DEFAULT_HEMATOCRIT = 0.45			# Fraction (0–1)

# Oxygen transport
O2_DIFFUSIVITY_PLASMA = 2.0e-9		# m^2/s
O2_SOLUBILITY = 1.3e-3				# mol/(m^3·Pa)
ARTERIAL_PO2 = 100.0				# mmHg
VENOUS_PO2 = 40.0					# mmHg

# Carbon dioxide transport
CO2_DIFFUSIVITY_PLASMA = 1.6e-9		# m^2/s
ARTERIAL_PCO2 = 40.0				# mmHg
VENOUS_PCO2 = 46.0					# mmHg

# Vessel wall exchange
VESSEL_WALL_PERMEABILITY = 1e-5		# m/s (placeholder)
DEFAULT_WALL_THICKNESS = 1.0		# µm

# Flow assumptions
DEFAULT_INLET_PRESSURE = 4000.0		# Pa (~30 mmHg)
DEFAULT_OUTLET_PRESSURE = 2000.0	# Pa (~15 mmHg)
MIN_VESSEL_RADIUS = 2.0				# µm
MAX_VESSEL_RADIUS = 100.0			# µm

# Scaling
DEFAULT_SCALE_UM_PER_PIXEL = 1.0 	# µm/pixel