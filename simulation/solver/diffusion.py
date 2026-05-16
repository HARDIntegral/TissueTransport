import numpy as np
from domain import TissueDomain
from species import Oxygen

domain = TissueDomain((100, 100))
oxygen = Oxygen()

D_eff = oxygen.effective_diffusivity_grid(
	temperature=domain.temperature,
	mu=domain.mu,
	epsilon=domain.epsilon,
	tau=domain.tau
)