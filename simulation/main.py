import matplotlib.pyplot as plt
from tqdm import tqdm
from domain import TissueDomain
from solver import explicit_step
from species import Oxygen

shape = (100, 100)
test_domain = TissueDomain(shape)
test_domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
test_domain.set_initial_concentration(0.0)
test_domain.add_circular_vessel(center=(50,50), radius=5, concentration=1.0)

oxygen = Oxygen()

for step in tqdm(range(10000), desc="Simulating Diffusion"):
	C_new = explicit_step(test_domain, oxygen, dt=0.1)
	test_domain.concentration = C_new
plt.imshow(C_new)
plt.colorbar()
plt.show()