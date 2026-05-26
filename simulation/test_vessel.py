"""
Smoke test for the real VeSeg vessel-to-transport pipeline.

This checks the full path we care about right now:

1. Run VeSeg on a real vessel image.
2. Build a VesselNetwork from the skeleton and radius data.
3. Reconstruct the vessel boundary used for tissue exchange.
4. Solve a simple pressure-driven flow problem.
5. Map O2 source / CO2 sink values back onto the vessel boundary.
6. Plot geometry + transport coupling in one figure.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import veseg

from domain.vessel import (
	FlowSolution,
	VesselNetwork,
	assign_flow_weighted_segment_concentrations,
	build_boundary_source_maps,
	solve_network_flow_mmhg,
	summarize_boundary_sources,
)
from domain.vessel.coupling import (
	build_centerline_mask,
	build_reconstructed_boundary_pixel_map,
	build_segment_pixel_map,
	build_vessel_mask,
)
from domain.vessel.geometry import extract_edge_centerlines_from_skeleton

DEFAULT_REAL_IMAGE_PATH = Path("blood_vessel_network_images/structure1.png")
INLET_PRESSURE_MMHG = 35.0
OUTLET_PRESSURE_MMHG = 15.0

def build_outline_mask(vessel_mask: np.ndarray) -> np.ndarray:
	"""
	Build a hollow outline from a filled vessel mask.
	"""

	vessel_mask = np.asarray(vessel_mask, dtype=bool)
	padded = np.pad(vessel_mask, pad_width=1, mode="constant", constant_values=False)
	eroded = np.ones_like(vessel_mask, dtype=bool)

	for row_offset in (-1, 0, 1):
		for col_offset in (-1, 0, 1):
			window = padded[
				1 + row_offset:1 + row_offset + vessel_mask.shape[0],
				1 + col_offset:1 + col_offset + vessel_mask.shape[1],
			]
			eroded &= window

	return vessel_mask & ~eroded

def build_outline_with_skeleton_image(
	outline_mask: np.ndarray,
	skeleton_mask: np.ndarray,
) -> np.ndarray:
	"""
	Build an RGB debug image with a white outline and red skeleton.
	"""

	outline_mask = np.asarray(outline_mask, dtype=bool)
	skeleton_mask = np.asarray(skeleton_mask, dtype=bool)

	image = np.zeros((*outline_mask.shape, 3), dtype=float)
	image[outline_mask] = [1.0, 1.0, 1.0]
	image[skeleton_mask] = [1.0, 0.0, 0.0]

	return image

def build_real_veseg_network(
	image_path: Path = DEFAULT_REAL_IMAGE_PATH,
) -> tuple[VesselNetwork, tuple[int, int], np.ndarray, np.ndarray] | None:
	"""
	Run VeSeg and convert the result into the simulation vessel representation.
	"""

	if not image_path.exists():
		return None

	result = veseg.predict(image_path)
	mask = result.mask if hasattr(result, "mask") else result

	mask, skeleton, distance_map, nodes, edges = veseg.extract_vessel_geometry(mask)

	try:
		edge_centerlines = extract_edge_centerlines_from_skeleton(
			skeleton=skeleton,
			nodes=nodes,
			edges=edges,
		)
	except Exception as error:
		print("edge centerline tracing failed")
		print(f"  {error}")
		print("  falling back to straight interpolated segment centerlines")
		edge_centerlines = None

	veseg_reconstruction = veseg.reconstruct_vessel_mask(skeleton, distance_map)

	network = VesselNetwork.from_veseg(
		nodes=nodes,
		edges=edges,
		distance_map=distance_map,
		centerlines=edge_centerlines,
		min_length_px=0.0,
		min_radius_px=0.0,
	)

	return network, mask.shape, skeleton, veseg_reconstruction

def build_debug_masks(
	network: VesselNetwork,
	shape: tuple[int, int],
	veseg_skeleton: np.ndarray,
	veseg_reconstruction: np.ndarray,
) -> dict[str, np.ndarray]:
	"""
	Build the masks used for visual sanity checks.
	"""

	centerline_mask = build_centerline_mask(network, shape)
	vessel_mask = build_vessel_mask(network, shape)
	outline_mask = build_outline_mask(veseg_reconstruction)
	outline_with_skeleton = build_outline_with_skeleton_image(outline_mask, veseg_skeleton)
	segment_pixel_map = build_segment_pixel_map(network, shape)

	print("Real VeSeg reconstructed mask stats:")
	print(f"  centerline pixels: {centerline_mask.sum()}")
	print(f"  vessel pixels: {vessel_mask.sum()}")
	print(f"  outline pixels: {outline_mask.sum()}")

	if network.segments:
		first_segment_id = next(iter(network.segments))
		print(f"  segment {first_segment_id} pixels: {len(segment_pixel_map[first_segment_id])}")

	print()

	return {
		"centerline": centerline_mask,
		"vessel": vessel_mask,
		"outline_with_skeleton": outline_with_skeleton,
	}

def summarize_network(label: str, network: VesselNetwork) -> None:
	"""
	Print basic vessel network stats.
	"""

	print(f"{label}:")
	print(f"  nodes: {len(network.nodes)}")
	print(f"  segments: {len(network.segments)}")
	print(f"  total length: {network.total_length_um():.2f} µm")
	print(f"  total volume: {network.total_volume_um3():.2f} µm^3")
	print()


def summarize_boundary_pixel_map(boundary_pixel_map: dict[int, list[tuple[int, int]]]) -> None:
	"""
	Print basic stats for segment-owned exchange boundary pixels.
	"""

	mapped_segments = len(boundary_pixel_map)
	total_boundary_pixels = sum(len(pixels) for pixels in boundary_pixel_map.values())

	print("Reconstructed boundary coupling map:")
	print(f"  mapped segments: {mapped_segments}")
	print(f"  assigned boundary pixels: {total_boundary_pixels}")

	if boundary_pixel_map:
		first_segment_id = next(iter(boundary_pixel_map))
		print(f"  segment {first_segment_id} boundary pixels: {len(boundary_pixel_map[first_segment_id])}")

	print()

def solve_and_summarize_flow(network: VesselNetwork) -> FlowSolution:
	"""
	Solve a quick pressure-driven flow problem for the largest component.
	"""

	inlet_node, outlet_node, largest_component = network.pressure_boundary_nodes_from_largest_component()

	flow_solution = solve_network_flow_mmhg(
		network=network,
		fixed_pressures_mmhg={
			inlet_node: INLET_PRESSURE_MMHG,
			outlet_node: OUTLET_PRESSURE_MMHG,
		},
	)

	flow_values = np.asarray([
		abs(flow.flow_um3_per_s)
		for flow in flow_solution.segment_flows.values()
	], dtype=float)

	print("Network flow smoke test:")
	print(f"  largest connected component nodes: {len(largest_component)}")
	print(f"  inlet node: {inlet_node} ({flow_solution.node_pressure_mmhg(inlet_node):.2f} mmHg)")
	print(f"  outlet node: {outlet_node} ({flow_solution.node_pressure_mmhg(outlet_node):.2f} mmHg)")
	print(f"  solved node pressures: {len(flow_solution.node_pressures_pa)}")
	print(f"  solved segment flows: {len(flow_solution.segment_flows)}")

	if flow_values.size:
		print(f"  min |flow|: {flow_values.min():.4e} µm^3/s")
		print(f"  mean |flow|: {flow_values.mean():.4e} µm^3/s")
		print(f"  max |flow|: {flow_values.max():.4e} µm^3/s")

	print()
	return flow_solution

def build_and_summarize_boundary_sources(
	shape: tuple[int, int],
	boundary_pixel_map: dict[int, list[tuple[int, int]]],
	flow_solution: FlowSolution,
):
	"""
	Build the temporary O2 source / CO2 sink maps from solved flow.
	"""

	# Placeholder chemistry for now. Good enough to check that vessel values
	# actually land on the reconstructed tissue boundary.
	segment_concentrations = assign_flow_weighted_segment_concentrations(
		flow_solution=flow_solution,
		inlet_oxygen=1.0,
		inlet_carbon_dioxide=0.0,
	)

	boundary_sources = build_boundary_source_maps(
		shape=shape,
		boundary_pixel_map=boundary_pixel_map,
		segment_concentrations=segment_concentrations,
	)

	summary = summarize_boundary_sources(boundary_sources)

	print("Boundary transport source maps:")
	print(f"  segment concentrations: {len(segment_concentrations)}")
	print(f"  oxygen total source: {summary['oxygen_total']:.4e}")
	print(f"  oxygen max pixel source: {summary['oxygen_max']:.4e}")
	print(f"  carbon dioxide total sink: {summary['carbon_dioxide_total']:.4e}")
	print(f"  carbon dioxide max pixel sink: {summary['carbon_dioxide_max']:.4e}")
	print()

	return boundary_sources

# Pure visualization. None of this belongs in the actual simulation loop.
def plot_combined_results(mask_set: dict[str, np.ndarray], boundary_sources) -> None:
	"""
	Plot geometry, flow-derived sources, and the final coupling overlay.
	"""

	fig, axes = plt.subplots(2, 3, figsize=(14, 8))

	axes[0, 0].imshow(mask_set["centerline"], cmap="gray")
	axes[0, 0].set_title("Real VeSeg Centerline")

	axes[0, 1].imshow(mask_set["vessel"], cmap="gray")
	axes[0, 1].set_title("Real VeSeg Reconstructed Vessel")

	axes[0, 2].imshow(mask_set["outline_with_skeleton"])
	axes[0, 2].set_title("Reconstructed Outline + Skeleton")

	oxygen_image = axes[1, 0].imshow(boundary_sources.oxygen, cmap="viridis", vmin=0.0, vmax=1.0)
	axes[1, 0].set_title("Oxygen Boundary Source")
	fig.colorbar(oxygen_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

	carbon_dioxide_image = axes[1, 1].imshow(
		boundary_sources.carbon_dioxide,
		cmap="coolwarm",
		vmin=-1.0,
		vmax=1.0,
	)
	axes[1, 1].set_title("Carbon Dioxide Boundary Sink")
	fig.colorbar(carbon_dioxide_image, ax=axes[1, 1], fraction=0.046, pad=0.04)

	axes[1, 2].imshow(mask_set["outline_with_skeleton"])
	axes[1, 2].imshow(boundary_sources.oxygen, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.75)
	axes[1, 2].set_title("Oxygen Source Overlay")

	for axis in axes.flat:
		axis.set_axis_off()

	plt.tight_layout()
	plt.show()

def main() -> None:
	"""
	Run the full real-image vessel coupling smoke test.
	"""

	real_result = build_real_veseg_network()

	if real_result is None:
		print(f"missing image: {DEFAULT_REAL_IMAGE_PATH}")
		return

	real_network, real_shape, veseg_skeleton, veseg_reconstruction = real_result
	summarize_network("real VeSeg-built network", real_network)

	mask_set = build_debug_masks(
		real_network,
		real_shape,
		veseg_skeleton,
		veseg_reconstruction,
	)

	# Geometry-only work. The real simulation should cache this once.
	boundary_pixel_map = build_reconstructed_boundary_pixel_map(
		network=real_network,
		reconstructed_vessel_mask=veseg_reconstruction,
	)
	summarize_boundary_pixel_map(boundary_pixel_map)

	# Flow is static unless geometry, pressure, or vessel radii change.
	flow_solution = solve_and_summarize_flow(real_network)
	boundary_sources = build_and_summarize_boundary_sources(
		shape=real_shape,
		boundary_pixel_map=boundary_pixel_map,
		flow_solution=flow_solution,
	)

	plot_combined_results(mask_set, boundary_sources)

if __name__ == "__main__":
	main()