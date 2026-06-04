import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import numpy as np
import gpu_solver
from domain import TissueDomain
from domain.vessel import build_vessel_transport_pipeline_from_image
from species import Oxygen


# Convert arrays into a format that the Rust/PyO3 interface can accept safely.
def _rust_array(value, dtype):
	"""Convert a value into a real C-contiguous NumPy array for PyO3/numpy."""
	return np.ascontiguousarray(np.asarray(value, dtype=dtype))


# Normalize a field for visualization only.
def normalize_display_field(values):
	"""Normalize a copy of a concentration field without changing the simulation state."""
	values = np.asarray(values, dtype=np.float32)
	min_value = float(np.min(values))
	max_value = float(np.max(values))

	if np.isclose(max_value, min_value):
		return np.zeros_like(values, dtype=np.float32)

	return np.clip((values - min_value) / (max_value - min_value), 0.0, 1.0)


# Combine oxygen, carbon dioxide, anoxia, and vessels into one RGBA frame.
def make_overlay_frame(o2, co2, vessel_mask, anoxic_threshold=0.2):
	"""Render O2, CO2, anoxic regions, and reconstructed vessels."""
	frame = np.zeros((*o2.shape, 4), dtype=np.float32)

	# Normalize only the rendered frame. The real O2/CO2 arrays stay untouched.
	o2_norm = normalize_display_field(o2)
	co2_norm = normalize_display_field(co2)

	# Oxygen uses a stronger warm perfusion-style orange/red gradient.
	frame[..., 0] += 1.0 * o2_norm
	frame[..., 1] += 0.55 * o2_norm
	frame[..., 2] += 0.10 * o2_norm
	frame[..., 3] += 0.65 * o2_norm

	# Carbon dioxide uses a stronger cyan gradient distinct from oxygen.
	frame[..., 0] += 0.0 * co2_norm
	frame[..., 1] += 0.8 * co2_norm
	frame[..., 2] += 1.0 * co2_norm
	frame[..., 3] += 0.55 * co2_norm

	frame = np.clip(frame, 0.0, 1.0)

	# Make anoxia a transparency gradient: lower O2 -> darker/more opaque.
	anoxic_norm = np.clip(
		(anoxic_threshold - o2) / anoxic_threshold,
		0.0,
		1.0,
	)

	frame[..., 0] *= (1.0 - 0.7 * anoxic_norm)
	frame[..., 1] *= (1.0 - 0.7 * anoxic_norm)
	frame[..., 2] *= (1.0 - 0.7 * anoxic_norm)
	frame[..., 3] += 0.75 * anoxic_norm
	frame = np.clip(frame, 0.0, 1.0)

	# Draw the reconstructed vessel geometry in solid red.
	frame[vessel_mask] = [0.45, 0.0, 0.0, 1.0]

	return frame


# Show exactly which source/sink maps are being passed into the solver.
def plot_solver_source_sink_maps(oxygen_source_map, carbon_dioxide_sink_map, structure_name):
	"""Debug plot for the actual O2 source and CO2 sink maps used by main.py."""
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))

	oxygen_image = axes[0].imshow(
		oxygen_source_map,
		cmap="viridis",
		vmin=0.0,
		vmax=1.0,
	)
	axes[0].set_title(f"{structure_name} O2 Source Map Used")
	axes[0].set_axis_off()
	fig.colorbar(oxygen_image, ax=axes[0], fraction=0.046, pad=0.04)

	carbon_dioxide_image = axes[1].imshow(
		carbon_dioxide_sink_map,
		cmap="coolwarm",
		vmin=-1.0,
		vmax=1.0,
	)
	axes[1].set_title(f"{structure_name} CO2 Sink Map Used")
	axes[1].set_axis_off()
	fig.colorbar(carbon_dioxide_image, ax=axes[1], fraction=0.046, pad=0.04)

	plt.tight_layout()
	plt.show(block=False)


# Prepare simulation arrays that do not change during a run.
def prepare_rust_solver_arrays(domain, species, oxygen_source_map, carbon_dioxide_sink_map):
	"""Prepare static NumPy arrays for the Rust GPU solver once."""
	vessel_source_mask = oxygen_source_map > 0.0

	return {
		"diffusivity": _rust_array(
			species.effective_diffusivity_grid(
				domain.temperature,
				domain.mu,
				domain.epsilon,
				domain.tau,
			),
			np.float32,
		),
		"vmax": _rust_array(domain.consumption_vmax, np.float32),
		"km": _rust_array(domain.consumption_km, np.float32),
		"vessel_mask": _rust_array(vessel_source_mask, bool),
		"vessel_concentration": _rust_array(oxygen_source_map, np.float32),
		"vessel_carbon_dioxide": _rust_array(carbon_dioxide_sink_map, np.float32),
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
		arrays["vessel_carbon_dioxide"],
		arrays["dx"],
		arrays["dy"],
		dt,
		1.0,
		steps,
		True,
	)


# Build one tissue domain from one vessel structure image.
def create_domain_from_structure(structure_path, structure_name, shape, scale):
	"""Create a tissue domain and reconstructed flow-weighted source/sink maps."""
	domain = TissueDomain(shape, scale)
	domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
	domain.set_initial_concentration(0.0)

	transport_result = build_vessel_transport_pipeline_from_image(
		image_path=structure_path,
		shape=shape,
	)

	boundary_sources = transport_result.boundary_sources
	oxygen_source_map = boundary_sources.oxygen
	# The tissue solver now interprets the vessel CO2 map as a source/sink term.
	# Negative values remove CO2 from vessel cells instead of pinning CO2 to a
	# fixed boundary concentration.
	carbon_dioxide_sink_map = -np.abs(boundary_sources.carbon_dioxide)
	vessel_mask = transport_result.geometry.reconstructed_mask.astype(bool)
	domain.set_vessel_mask(vessel_mask, 1.0)
	domain.set_uniform_consumption(vmax=0.05, km=0.05)

	print(f"{structure_name}: vessel transport pipeline")
	print(f"  nodes: {len(transport_result.geometry.network.nodes)}")
	print(f"  segments: {len(transport_result.geometry.network.segments)}")
	print(f"  raw centerline source pixels: {int(np.count_nonzero(boundary_sources.oxygen))}")
	print(f"  raw centerline sink pixels: {int(np.count_nonzero(boundary_sources.carbon_dioxide))}")
	print(f"  reconstructed source pixels: {int(np.count_nonzero(oxygen_source_map))}")
	print(f"  reconstructed sink pixels: {int(np.count_nonzero(boundary_sources.carbon_dioxide))}")
	print(f"  CO2 vessel sink pixels: {int(np.count_nonzero(carbon_dioxide_sink_map))}")
	print()

	return domain, vessel_mask, oxygen_source_map, carbon_dioxide_sink_map


# Run one full gas-exchange simulation for one vessel structure.
def run_structure_simulation(structure_path, structure_name, ax, shape, scale, oxygen):
	"""Run one simulation and draw the final composite frame on one axis."""
	domain, vessel_mask, oxygen_source_map, carbon_dioxide_sink_map = create_domain_from_structure(
		structure_path,
		structure_name,
		shape,
		scale,
	)
	plot_solver_source_sink_maps(
		oxygen_source_map,
		carbon_dioxide_sink_map,
		structure_name,
	)
	rust_arrays = prepare_rust_solver_arrays(
		domain,
		oxygen,
		oxygen_source_map=oxygen_source_map,
		carbon_dioxide_sink_map=carbon_dioxide_sink_map,
	)

	visual_frame = make_overlay_frame(
		domain.concentration,
		np.zeros_like(domain.concentration),
		vessel_mask,
	)
	composite_plot = ax.imshow(visual_frame)
	ax.set_title(f"{structure_name} | step 0")
	ax.axis("off")

	for step in tqdm(
		range(frame_interval_steps, total_steps + 1, frame_interval_steps),
		desc=structure_name,
	):
		o2, co2 = rust_reaction_diffusion_steps(
			domain.concentration,
			rust_arrays,
			steps=frame_interval_steps,
			dt=simulation_dt,
		)

		domain.concentration = o2
		rust_arrays["carbon_dioxide"] = co2

		visual_frame = make_overlay_frame(
			domain.concentration,
			rust_arrays["carbon_dioxide"],
			vessel_mask,
		)
		composite_plot.set_data(visual_frame)
		ax.set_title(f"{structure_name} | step {step}")
		plt.pause(0.001)

	return visual_frame


# Configure the physical simulation time and how often frames are sampled.
shape = (1000, 1000)
scale = (1e-3, 1e-3)
simulation_dt = 0.0001
simulation_time = 10.0
total_steps = int(simulation_time / simulation_dt)
target_frames = 100
frame_interval_steps = max(1, total_steps // target_frames)

oxygen = Oxygen()

structures = [
	("blood_vessel_network_images/structure1.png", "Structure 1"),
]

plt.ion()

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
fig.suptitle(f"Gas exchange simulation | dt = {simulation_dt}", fontsize=14)

structure_path, structure_name = structures[0]
run_structure_simulation(
	structure_path,
	structure_name,
	ax,
	shape,
	scale,
	oxygen,
)

legend_elements = [
	Patch(
		facecolor=(1.0, 0.55, 0.10),
		label="O₂ concentration (frame-normalized, dark → bright)",
	),
	Patch(
		facecolor=(0.0, 0.8, 1.0),
		label="CO₂ concentration (frame-normalized, dark → bright)",
	),
	Patch(
		facecolor=(0.05, 0.05, 0.05),
		label="Anoxic gradient (O₂ < 0.2, more opaque = lower O₂)",
	),
	Patch(
		facecolor=(0.45, 0.0, 0.0),
		label="Reconstructed vessel geometry",
	),
]

fig.legend(
	handles=legend_elements,
	loc="lower center",
	bbox_to_anchor=(0.5, 0.045),
	ncol=2,
	frameon=True,
	fontsize=8,
	title="Visualization legend",
)

plt.ioff()
plt.subplots_adjust(
	left=0.07,
	right=0.97,
	bottom=0.23,
	top=0.88,
	wspace=0.03,
)

plt.show(block=True)
