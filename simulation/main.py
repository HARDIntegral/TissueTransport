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
		"dx": domain.dx,
		"dy": domain.dy,
	}


# Send one simulation chunk to the Rust GPU solver and return the updated field.
def rust_reaction_diffusion_steps(concentration, arrays, steps, dt):
	"""Run a chunk of reaction-diffusion steps through the Rust GPU solver."""
	return gpu_solver.run_steps_auto_numpy(
		_rust_array(concentration, np.float32),
		arrays["diffusivity"],
		arrays["vmax"],
		arrays["km"],
		arrays["vessel_mask"],
		arrays["vessel_concentration"],
		arrays["dx"],
		arrays["dy"],
		dt,
		steps,
		True
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
fig, ax = plt.subplots()

concentration_plot = ax.imshow(
	test_domain.concentration,
	vmin=0,
	vmax=1.0
)

ax.imshow(mask, cmap="Reds", alpha=0.4)

plt.colorbar(concentration_plot, ax=ax, label="O2 concentration")
ax.set_title("Oxygen diffusion | step 0")

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
	test_domain.concentration = rust_reaction_diffusion_steps(
		test_domain.concentration,
		rust_arrays,
		steps=frame_interval_steps,
		dt=simulation_dt
	)

	concentration_plot.set_data(test_domain.concentration)
	ax.set_title(f"Oxygen diffusion | step {step} | dt = {simulation_dt}s")
	plt.pause(0.001)

	frames.append(test_domain.concentration.copy())
	frame_steps.append(step)

plt.ioff()

# Save the sampled concentration frames as an animated GIF.
fig_gif, ax_gif = plt.subplots()
gif_plot = ax_gif.imshow(
	frames[0],
	vmin=0,
	vmax=1.0
)
ax_gif.imshow(mask, cmap="Reds", alpha=0.4)
plt.colorbar(gif_plot, ax=ax_gif, label="O2 concentration")


# Update function used by Matplotlib's animation writer.
def update_gif(frame_index):
	gif_plot.set_data(frames[frame_index])
	ax_gif.set_title(f"Oxygen diffusion | step {frame_steps[frame_index]} | dt = {simulation_dt}s")
	return [gif_plot]


animation = FuncAnimation(
	fig_gif,
	update_gif,
	frames=len(frames),
	interval=50,
	blit=False
)

animation.save(
	"oxygen_diffusion.gif",
	writer=PillowWriter(fps=20)
)

# Compare final oxygen distributions across Michaelis-Menten parameter choices.
vmax_values = [0.01, 0.05, 0.1]
km_values = [0.01, 0.05, 0.1]

fig_maps, axes = plt.subplots(
	len(km_values),
	len(vmax_values),
	figsize=(12,12)
)

# Run one independent simulation for each vmax/km pair.
for row, km in enumerate(km_values):
	for col, vmax in enumerate(vmax_values):
		domain = TissueDomain(shape, scale)
		domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
		domain.set_initial_concentration(0.0)
		domain.set_vessel_mask(mask, 1.0)
		domain.set_uniform_consumption(vmax=vmax, km=km)
		sweep_arrays = prepare_rust_solver_arrays(domain, oxygen)

		for _ in tqdm(
			range(frame_interval_steps, total_steps + 1, frame_interval_steps),
			desc=f"Sweep vmax={vmax}, km={km}",
			leave=True
		):
			domain.concentration = rust_reaction_diffusion_steps(
				domain.concentration,
				sweep_arrays,
				steps=frame_interval_steps,
				dt=simulation_dt
			)

		ax_map = axes[row, col]
		im = ax_map.imshow(
			domain.concentration,
			vmin=0,
			vmax=1.0
		)
		ax_map.imshow(mask, cmap="Reds", alpha=0.25)
		ax_map.set_title(f"vmax={vmax}\nkm={km}")
		ax_map.axis("off")

fig_maps.suptitle("Michaelis-Menten parameter sweep")
plt.tight_layout()
# Save the parameter sweep figure to disk.
fig_maps.savefig("parameter_sweep.png", dpi=300)

plt.close(fig)
plt.close(fig_gif)
plt.show(block=False)
plt.pause(0.1)
plt.close('all')