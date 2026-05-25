import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import numpy as np
import gpu_solver
from domain import TissueDomain
from species import Oxygen
import veseg


# Convert arrays into a format that the Rust/PyO3 interface can accept safely.
def _rust_array(value, dtype):
	"""Convert a value into a real C-contiguous NumPy array for PyO3/numpy."""
	return np.ascontiguousarray(np.asarray(value, dtype=dtype))


# Combine oxygen, carbon dioxide, anoxia, and vessels into one RGBA frame.
def make_overlay_frame(o2, co2, vessel_mask, anoxic_threshold=0.2):
	"""Render O2, CO2, anoxic regions, and vessels as one transparent overlay."""
	frame = np.zeros((*o2.shape, 4), dtype=np.float32)

	# Fixed normalization: O2 = 0.0→1.0 mapped to dark→bright orange.
	o2_norm = o2 / 1.0
	# Fixed normalization: CO2 = 0.0→0.5 mapped to dark→bright cyan.
	co2_norm = co2 / 0.5

	o2_norm = np.clip(o2_norm, 0.0, 1.0)
	co2_norm = np.clip(co2_norm, 0.0, 1.0)

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
		1.0
	)

	frame[..., 0] *= (1.0 - 0.7 * anoxic_norm)
	frame[..., 1] *= (1.0 - 0.7 * anoxic_norm)
	frame[..., 2] *= (1.0 - 0.7 * anoxic_norm)
	frame[..., 3] += 0.75 * anoxic_norm

	frame = np.clip(frame, 0.0, 1.0)

	# Deep blood red for vessels.
	frame[vessel_mask] = [0.45, 0.0, 0.0, 1.0]

	return frame


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


# Build one tissue domain from one vessel structure image.
def create_domain_from_structure(structure_path, shape, scale):
	"""Create a tissue domain and vessel mask from one vessel structure image."""
	domain = TissueDomain(shape, scale)
	domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
	domain.set_initial_concentration(0.0)

	veseg_result = veseg.predict(
		path=structure_path,
		mode=veseg.ENHANCED_INVERTED,
	)
	raw_mask = veseg_result.despeckle(min_neighbors=2).mask.astype(bool)
	mask = veseg.resize_mask(raw_mask, size=(shape[1], shape[0])).astype(bool)

	domain.set_vessel_mask(mask, 1.0)
	domain.set_uniform_consumption(vmax=0.05, km=0.05)

	return domain, mask


# Run one full gas-exchange simulation for one vessel structure.
def run_structure_simulation(structure_path, structure_name, ax, shape, scale, oxygen):
	"""Run one simulation and draw the final composite frame on one axis."""
	domain, mask = create_domain_from_structure(structure_path, shape, scale)
	rust_arrays = prepare_rust_solver_arrays(domain, oxygen)

	visual_frame = make_overlay_frame(
		domain.concentration,
		np.zeros_like(domain.concentration),
		mask
	)
	composite_plot = ax.imshow(visual_frame)
	ax.set_title(f"{structure_name} | step 0")
	ax.axis("off")

	for step in tqdm(
		range(frame_interval_steps, total_steps + 1, frame_interval_steps),
		desc=structure_name
	):
		o2, co2 = rust_reaction_diffusion_steps(
			domain.concentration,
			rust_arrays,
			steps=frame_interval_steps,
			dt=simulation_dt
		)

		domain.concentration = o2
		rust_arrays["carbon_dioxide"] = co2

		visual_frame = make_overlay_frame(
			domain.concentration,
			rust_arrays["carbon_dioxide"],
			mask
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
	("blood_vessel_network_images/structure2.png", "Structure 2"),
	("blood_vessel_network_images/structure3.png", "Structure 3"),
]

plt.ion()

# Draw all three vessel structures side-by-side in one row.
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle(f"Gas exchange comparison | dt = {simulation_dt}", fontsize=14)

for ax, (structure_path, structure_name) in zip(axes, structures):
	run_structure_simulation(
		structure_path,
		structure_name,
		ax,
		shape,
		scale,
		oxygen
	)

legend_elements = [
	Patch(
		facecolor=(1.0, 0.55, 0.10),
		label="O₂ concentration (0 → 1.0, dark → bright)"
	),
	Patch(
		facecolor=(0.0, 0.8, 1.0),
		label="CO₂ concentration (0 → 0.5, dark → bright)"
	),
	Patch(
		facecolor=(0.05, 0.05, 0.05),
		label="Anoxic gradient (O₂ < 0.2, more opaque = lower O₂)"
	),
	Patch(
		facecolor=(0.45, 0.0, 0.0),
		label="Blood vessels (fixed O₂ source)"
	)
]

fig.legend(
	handles=legend_elements,
	loc="lower center",
	ncol=2,
	frameon=True,
	fontsize=8,
	title="Visualization legend",
	bbox_to_anchor=(0.5, 0.05),
)

plt.ioff()
plt.subplots_adjust(
	left=0.03,
	right=0.97,
	bottom=0.18,
	top=0.88,
	wspace=0.03
)

fig.savefig(
	"gas_exchange_structure_comparison.png",
	dpi=300,
	bbox_inches="tight"
)

plt.show(block=True)
