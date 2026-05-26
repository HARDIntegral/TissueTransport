from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
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


# Scale graph node coordinates and edge length/radius values into the simulation grid.
def scale_vessel_geometry(nodes, edges, source_shape, target_shape):
	"""Scale VeSeg graph geometry from segmentation pixels into simulation pixels."""
	source_h, source_w = source_shape
	target_h, target_w = target_shape

	row_scale = target_h / source_h
	col_scale = target_w / source_w
	radius_scale = 0.5 * (row_scale + col_scale)

	scaled_nodes = np.array(nodes, dtype=np.float32, copy=True)
	scaled_edges = np.array(edges, dtype=np.float32, copy=True)

	# Node table columns: [node_id, row, col, node_type].
	if scaled_nodes.size > 0:
		scaled_nodes[:, 1] *= row_scale
		scaled_nodes[:, 2] *= col_scale

	# Edge table columns:
	# [edge_id, start_node, end_node, length_px, mean_radius, min_radius, max_radius, normalized_radius].
	if scaled_edges.size > 0:
		scaled_edges[:, 3] *= radius_scale
		scaled_edges[:, 4] *= radius_scale
		scaled_edges[:, 5] *= radius_scale
		scaled_edges[:, 6] *= radius_scale

	return scaled_nodes, scaled_edges


# Save VeSeg geometry outputs so later flow/transport code can reuse them directly.
def save_vessel_geometry(output_path, raw_mask, skeleton, distance_map, reconstructed_mask, scaled_skeleton, scaled_distance_map, scaled_reconstructed_mask, simulation_mask, nodes, edges, scaled_nodes, scaled_edges):
	"""Save raw and simulation-scaled vessel geometry arrays to one compressed file."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	np.savez_compressed(
		output_path,
		raw_mask=raw_mask.astype(bool),
		skeleton=skeleton.astype(bool),
		distance_map=distance_map.astype(np.float32),
		reconstructed_mask=reconstructed_mask.astype(bool),
		scaled_skeleton=scaled_skeleton.astype(bool),
		scaled_distance_map=scaled_distance_map.astype(np.float32),
		scaled_reconstructed_mask=scaled_reconstructed_mask.astype(bool),
		simulation_mask=simulation_mask.astype(bool),
		nodes=nodes.astype(np.float32),
		edges=edges.astype(np.float32),
		scaled_nodes=scaled_nodes.astype(np.float32),
		scaled_edges=scaled_edges.astype(np.float32),
	)


# Crop or pad a scaled mask so it exactly matches the simulation domain shape.
def fit_mask_to_shape(mask, target_shape):
	"""Fit a binary mask to the target simulation shape without resampling again."""
	target_h, target_w = target_shape
	fitted = np.zeros(target_shape, dtype=bool)

	copy_h = min(mask.shape[0], target_h)
	copy_w = min(mask.shape[1], target_w)

	fitted[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
	return fitted


# Scale the centerline/radius representation first, then reconstruct at simulation resolution.
def scale_skeleton_distance_for_reconstruction(skeleton, distance_map, source_shape, target_shape):
	"""Scale VeSeg skeleton + radius data before reconstruction to avoid blocky mask scaling."""
	source_h, source_w = source_shape
	target_h, target_w = target_shape

	row_scale = target_h / source_h
	col_scale = target_w / source_w
	radius_scale = 0.5 * (row_scale + col_scale)

	scaled_skeleton = np.zeros(target_shape, dtype=bool)
	scaled_distance_map = np.zeros(target_shape, dtype=np.float32)

	rows, cols = np.nonzero(skeleton)

	for row, col in zip(rows, cols):
		target_row = min(int(round(row * row_scale)), target_h - 1)
		target_col = min(int(round(col * col_scale)), target_w - 1)

		scaled_skeleton[target_row, target_col] = True
		scaled_distance_map[target_row, target_col] = max(
			scaled_distance_map[target_row, target_col],
			float(distance_map[row, col]) * radius_scale,
		)

	return scaled_skeleton, scaled_distance_map


# Plot the extracted vessel graph, with each edge colored by mean radius.
def plot_radius_colored_graph(nodes, edges, vessel_mask):
	"""Render the scaled vessel graph and color edges by mean radius."""
	fig, ax = plt.subplots(1, 1, figsize=(7, 7))
	ax.imshow(vessel_mask, cmap="gray", alpha=0.18)

	if nodes.size == 0 or edges.size == 0:
		ax.set_title("Vessel graph + radii | no graph data")
		ax.axis("off")
		plt.close(fig)
		return

	# Node table columns: [node_id, row, col, node_type].
	node_lookup = {
		int(node[0]): (float(node[2]), float(node[1]))
		for node in nodes
	}

	segments = []
	radii = []
	line_widths = []

	for edge in edges:
		start_id = int(edge[1])
		end_id = int(edge[2])

		if start_id not in node_lookup or end_id not in node_lookup:
			continue

		mean_radius = float(edge[4])
		segments.append([node_lookup[start_id], node_lookup[end_id]])
		radii.append(mean_radius)
		line_widths.append(max(1.0, 0.35 * mean_radius))

	if not segments:
		ax.set_title("Vessel graph + radii | no drawable edges")
		ax.axis("off")
		plt.close(fig)
		return

	collection = LineCollection(
		segments,
		array=np.asarray(radii, dtype=np.float32),
		cmap="viridis",
		linewidths=line_widths,
		alpha=0.95,
	)

	ax.add_collection(collection)
	ax.scatter(
		nodes[:, 2],
		nodes[:, 1],
		s=8,
		c="black",
		alpha=0.75,
		label="Graph nodes",
	)

	colorbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
	colorbar.set_label("Mean vessel radius (simulation pixels)")

	ax.set_title("Structure 3 vessel graph colored by radius")
	ax.set_xlim(0, vessel_mask.shape[1])
	ax.set_ylim(vessel_mask.shape[0], 0)
	ax.set_aspect("equal")
	ax.axis("off")


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
def create_domain_from_structure(structure_path, structure_name, shape, scale):
	"""Create a tissue domain, vessel mask, and saved VeSeg graph geometry."""
	domain = TissueDomain(shape, scale)
	domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
	domain.set_initial_concentration(0.0)

	veseg_result = veseg.predict(
		path=structure_path,
		mode=veseg.ENHANCED_INVERTED,
	)

	# Start with VeSeg's postprocessed prediction mask.
	raw_mask = veseg_result.despeckle(min_neighbors=2).mask.astype(bool)

	# Extract reusable graph geometry at the native VeSeg mask resolution.
	geometry_mask, skeleton, distance_map, nodes, edges = veseg.extract_vessel_geometry(raw_mask)

	# Keep the native-resolution reconstruction for saved geometry/debugging.
	reconstructed_mask = veseg.reconstruct_vessel_mask(skeleton, distance_map).astype(bool)

	# For the simulation mask, scale the centerline + radius data first and reconstruct at
	# simulation resolution. This avoids magnifying a low-resolution binary mask, which is
	# what makes vessels look blocky.
	scaled_skeleton, scaled_distance_map = scale_skeleton_distance_for_reconstruction(
		skeleton,
		distance_map,
		source_shape=geometry_mask.shape,
		target_shape=shape,
	)

	scaled_reconstructed_mask = veseg.reconstruct_vessel_mask(
		scaled_skeleton,
		scaled_distance_map,
	).astype(bool)

	simulation_mask = fit_mask_to_shape(
		scaled_reconstructed_mask,
		shape,
	)

	# Scale node coordinates and edge length/radius columns into simulation-pixel units.
	scaled_nodes, scaled_edges = scale_vessel_geometry(
		nodes,
		edges,
		source_shape=geometry_mask.shape,
		target_shape=shape,
	)

	safe_name = structure_name.lower().replace(" ", "_")
	save_vessel_geometry(
		Path("vessel_geometry") / f"{safe_name}_geometry.npz",
		geometry_mask,
		skeleton,
		distance_map,
		reconstructed_mask,
		scaled_skeleton,
		scaled_distance_map,
		scaled_reconstructed_mask,
		simulation_mask,
		nodes,
		edges,
		scaled_nodes,
		scaled_edges,
	)

	domain.set_vessel_mask(simulation_mask, 1.0)
	domain.set_uniform_consumption(vmax=0.05, km=0.05)

	return domain, simulation_mask, scaled_nodes, scaled_edges


# Run one full gas-exchange simulation for one vessel structure.
def run_structure_simulation(structure_path, structure_name, ax, shape, scale, oxygen):
	"""Run one simulation and draw the final composite frame on one axis."""
	domain, mask, nodes, edges = create_domain_from_structure(
		structure_path,
		structure_name,
		shape,
		scale
	)
	print(f"{structure_name}: saved {nodes.shape[0]} nodes and {edges.shape[0]} radius-annotated edges")
	plot_radius_colored_graph(
		nodes,
		edges,
		mask
	)
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
	("blood_vessel_network_images/structure3.png", "Structure 3"),
]

plt.ion()

# Draw only Structure 3 for the current VeSeg geometry workflow.
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
fig.suptitle(f"Gas exchange simulation | dt = {simulation_dt}", fontsize=14)

structure_path, structure_name = structures[0]
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
	wspace=0.03
)

fig.savefig(
	"gas_exchange_structure3.png",
	dpi=300,
	bbox_inches="tight"
)

plt.show(block=True)
