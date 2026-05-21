import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import numpy as np
import gpu_solver
from domain import TissueDomain
from species import Oxygen
from image_processing import load_image_rgb, threshold_vessels, remove_stray_pixels, smooth_image


# Convert arrays into a format that the Rust/PyO3 interface can accept safely.
def _rust_array(value, dtype):
	"""Convert a value into a real C-contiguous NumPy array for PyO3/numpy."""
	return np.ascontiguousarray(np.asarray(value, dtype=dtype))


# Prepare simulation arrays that do not change during a run.
def prepare_rust_solver_arrays(domain, species):
	"""Prepare static NumPy arrays for the Rust GPU solver once."""
	return {
		"diffusivity": _rust_array(
			species.effective_diffusivity_grid(
				domain.temperature,
				domain.mu,
				domain.epsilon,
				domain.tau
			),
			np.float32
		),
		"vmax": _rust_array(domain.consumption_vmax, np.float32),
		"km": _rust_array(domain.consumption_km, np.float32),
		"vessel_mask": _rust_array(domain.vessel_mask, bool),
		"vessel_concentration": _rust_array(domain.vessel_concentration, np.float32),
		"carbon_dioxide": np.zeros_like(domain.concentration, dtype=np.float32),
		"dx": domain.dx,
		"dy": domain.dy,
	}


# Send one simulation chunk to the Rust GPU solver and return the updated field.
def rust_reaction_diffusion_steps(concentration, arrays, steps, dt):
	"""Run coupled O2/CO2 gas exchange through the Rust GPU solver."""
	return gpu_solver.run_gas_exchange_steps_auto_numpy(
		_rust_array(concentration, np.float32),
		_rust_array(arrays["carbon_dioxide"], np.float32),
		arrays["diffusivity"],
		arrays["diffusivity"] * 0.8,
		arrays["vmax"],
		arrays["km"],
		arrays["vessel_mask"],
		arrays["vessel_concentration"],
		np.zeros_like(arrays["vessel_concentration"], dtype=np.float32),
		arrays["dx"],
		arrays["dy"],
		dt,
		1.0,
		steps,
		True,
	)

# Create the tissue domain and initialize uniform tissue properties.
shape = (1000, 1000)
scale = (1e-3, 1e-3)
test_domain = TissueDomain(shape, scale)
test_domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
test_domain.set_initial_concentration(0.0)

# Load the vessel image and convert it into a clean binary vessel mask.
image = load_image_rgb(
	"blood_vessel_network_images/structure3.png",
	shape,
	5.0
)
image = smooth_image(image, 1)
mask = threshold_vessels(image)
mask = remove_stray_pixels(mask, 2)

# Use the vessel mask as fixed oxygen sources and set tissue consumption.
test_domain.set_vessel_mask(mask, 1.0)
test_domain.set_uniform_consumption(vmax=0.05, km=0.05)

oxygen = Oxygen()

plt.ion()

# Set up the live concentration plot used while the simulation runs.
fig, (ax_o2, ax_co2) = plt.subplots(1, 2, figsize=(12, 5))

concentration_plot = ax_o2.imshow(
	test_domain.concentration,
	vmin=0,
	vmax=1.0
)
co2_plot = ax_co2.imshow(
	np.zeros_like(test_domain.concentration),
	vmin=0,
	vmax=0.3
)

ax_o2.imshow(mask, cmap="Reds", alpha=0.4)
ax_co2.imshow(mask, cmap="Blues", alpha=0.4)

plt.colorbar(concentration_plot, ax=ax_o2, label="O2 concentration")
plt.colorbar(co2_plot, ax=ax_co2, label="CO2 concentration")

ax_o2.set_title("Oxygen | step 0")
ax_co2.set_title("Carbon dioxide | step 0 | dt = {simulation_dt}")

# Store sampled frames so the simulation can be saved as a GIF later.
frames = []
frame_steps = []

frames.append(test_domain.concentration.copy())
frame_steps.append(0)

# Configure the physical simulation time and how often frames are sampled.
simulation_dt = 0.0001
simulation_time = 10.0
total_steps = int(simulation_time / simulation_dt)
target_frames = 100
frame_interval_steps = max(1, total_steps // target_frames)
rust_arrays = prepare_rust_solver_arrays(test_domain, oxygen)

# Run the main simulation in chunks so intermediate timesteps stay inside Rust/GPU.
for step in tqdm(
	range(frame_interval_steps, total_steps + 1, frame_interval_steps),
	desc="Main simulation"
):
	o2, co2 = rust_reaction_diffusion_steps(
		test_domain.concentration,
		rust_arrays,
		steps=frame_interval_steps,
		dt=simulation_dt
	)

	test_domain.concentration = o2
	rust_arrays["carbon_dioxide"] = co2

	concentration_plot.set_data(test_domain.concentration)
	co2_plot.set_data(rust_arrays["carbon_dioxide"])

	ax_o2.set_title(f"Oxygen | step {step}")
	ax_co2.set_title(f"Carbon dioxide | step {step} | dt = {simulation_dt}")
	plt.pause(0.001)

	frames.append(test_domain.concentration.copy())
	frame_steps.append(step)

plt.ioff()

# Save the final O2/CO2 state for documentation or README images.
fig.savefig(
	"gas_exchange_final.png",
	dpi=300,
	bbox_inches="tight"
)

plt.show(block=True)
