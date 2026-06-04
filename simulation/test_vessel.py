"""
Smoke test for the real VeSeg -> Rust vessel transport pipeline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domain.vessel import (
	VesselTransportResult,
	build_centerline_mask,
	build_boundary_source_display_maps,
	build_segment_pixel_map,
	build_vessel_mask,
	build_vessel_transport_pipeline_from_image,
	summarize_boundary_sources as summarize_boundary_source_maps,
	summarize_segment_concentrations,
)

DEFAULT_REAL_IMAGE_PATH = Path("blood_vessel_network_images/structure1.png")


 # Build a hollow outline from a filled vessel mask.
def build_outline_mask(vessel_mask: np.ndarray) -> np.ndarray:

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


 # Build an RGB debug image with a white outline and red skeleton.
def build_outline_with_skeleton_image(
	outline_mask: np.ndarray,
	skeleton_mask: np.ndarray,
) -> np.ndarray:

	outline_mask = np.asarray(outline_mask, dtype=bool)
	skeleton_mask = np.asarray(skeleton_mask, dtype=bool)

	image = np.zeros((*outline_mask.shape, 3), dtype=float)
	image[outline_mask] = [1.0, 1.0, 1.0]
	image[skeleton_mask] = [1.0, 0.0, 0.0]

	return image


 # Build the masks used for visual sanity checks.
def build_debug_masks(result: VesselTransportResult) -> dict[str, np.ndarray]:

	geometry = result.geometry
	network = geometry.network
	shape = geometry.reconstructed_mask.shape
	centerline_mask = build_centerline_mask(network, shape)
	vessel_mask = build_vessel_mask(network, shape)
	outline_mask = build_outline_mask(geometry.reconstructed_mask)
	outline_with_skeleton = build_outline_with_skeleton_image(outline_mask, geometry.skeleton)
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


 # Build an RGB image showing solved vs ignored vessel regions.
 # Green segments have solved Rust flow. Red segments were ignored.
def build_perfused_component_overlay(result: VesselTransportResult) -> np.ndarray:

	geometry = result.geometry
	shape = geometry.reconstructed_mask.shape
	segment_pixel_map = build_segment_pixel_map(geometry.network, shape)
	outline_mask = build_outline_mask(geometry.reconstructed_mask)
	perfused_segment_ids = set(result.topology.downstream_connections.keys())

	overlay = np.zeros((*shape, 3), dtype=float)
	overlay[outline_mask] = [1.0, 1.0, 1.0]

	for segment_id, pixels in segment_pixel_map.items():
		segment_pixels = np.asarray(pixels, dtype=int)

		if segment_pixels.size == 0:
			continue

		rows = segment_pixels[:, 0]
		cols = segment_pixels[:, 1]

		if int(segment_id) in perfused_segment_ids:
			overlay[rows, cols] = [0.0, 1.0, 0.0]
		else:
			overlay[rows, cols] = [1.0, 0.0, 0.0]

	return overlay


 # Print basic vessel network stats.
def summarize_network(result: VesselTransportResult) -> None:

	network = result.geometry.network

	print("real VeSeg-built network:")
	print(f"  nodes: {len(network.nodes)}")
	print(f"  segments: {len(network.segments)}")
	print(f"  total length: {network.total_length_um():.2f} µm")
	print(f"  total volume: {network.total_volume_um3():.2f} µm^3")
	print()


 # Print basic stats for segment-owned exchange boundary pixels.
def summarize_boundary_pixel_map(result: VesselTransportResult) -> None:

	boundary_pixel_map = result.boundary_pixel_map
	mapped_segments = len(boundary_pixel_map)
	total_boundary_pixels = sum(len(pixels) for pixels in boundary_pixel_map.values())

	print("Reconstructed boundary coupling map:")
	print(f"  mapped segments: {mapped_segments}")
	print(f"  assigned boundary pixels: {total_boundary_pixels}")

	if boundary_pixel_map:
		first_segment_id = next(iter(boundary_pixel_map))
		print(f"  segment {first_segment_id} boundary pixels: {len(boundary_pixel_map[first_segment_id])}")

	print()


 # Print basic stats for the Rust flow solve and perfusion topology.
def summarize_flow(result: VesselTransportResult) -> None:

	flow_solution = result.flow_solution
	topology = result.topology
	flow_values = np.asarray([
		abs(flow.flow_um3_per_s)
		for flow in flow_solution.segment_flows.values()
	], dtype=float)

	print("Rust network flow smoke test:")
	print(f"  solved node pressures: {len(flow_solution.node_pressures_pa)}")
	print(f"  solved segment flows: {len(flow_solution.segment_flows)}")
	print(f"  perfusion inlet nodes: {len(topology.inlet_nodes)}")
	print(f"  perfusion outlet nodes: {len(topology.outlet_nodes)}")
	print(f"  perfusion traversal segments: {len(topology.traversal_segments)}")
	print(f"  downstream connections: {len(topology.downstream_connections)}")

	if flow_values.size:
		print(f"  min |flow|: {flow_values.min():.4e} µm^3/s")
		print(f"  mean |flow|: {flow_values.mean():.4e} µm^3/s")
		print(f"  max |flow|: {flow_values.max():.4e} µm^3/s")

	print()



 # Print stats for the O2 source / CO2 sink maps.
def summarize_boundary_sources(result: VesselTransportResult) -> None:

	summary = summarize_boundary_source_maps(result.boundary_sources)
	segment_summary = summarize_segment_concentrations(
		result.segment_concentrations,
	)

	print("Boundary transport source maps:")
	print(f"  segment concentrations: {len(result.segment_concentrations)}")
	print(f"  oxygen concentration min: {segment_summary['oxygen_min']:.4f}")
	print(f"  oxygen concentration mean: {segment_summary['oxygen_mean']:.4f}")
	print(f"  oxygen concentration max: {segment_summary['oxygen_max']:.4f}")
	print(f"  oxygen pixels: {summary['oxygen_pixels']:.0f}")
	print(f"  oxygen total source: {summary['oxygen_total']:.4e}")
	print(f"  oxygen min nonzero source: {summary['oxygen_min_nonzero']:.4e}")
	print(f"  oxygen max pixel source: {summary['oxygen_max']:.4e}")
	print(f"  carbon dioxide total sink: {summary['carbon_dioxide_total']:.4e}")
	print(f"  carbon dioxide max pixel sink: {summary['carbon_dioxide_max']:.4e}")
	print()


# Draw pressure and flow-direction diagnostics into the main vessel figure.
def plot_flow_direction_map(result: VesselTransportResult, pressure_axis, direction_axis, fig) -> None:

	network = result.geometry.network
	flow_solution = result.flow_solution
	outline_mask = build_outline_mask(result.geometry.reconstructed_mask)
	pressure_values = np.asarray(list(flow_solution.node_pressures_pa.values()), dtype=float)
	flow_values = np.asarray([
		abs(flow.flow_um3_per_s)
		for flow in flow_solution.segment_flows.values()
	], dtype=float)
	min_pressure = pressure_values.min() if pressure_values.size else 0.0
	max_pressure = pressure_values.max() if pressure_values.size else 1.0
	max_flow = flow_values.max() if flow_values.size else 1.0

	pressure_axis.imshow(outline_mask, cmap="gray")
	pressure_axis.set_title("Pressure Map: Yellow = High, Purple = Low")
	direction_axis.imshow(outline_mask, cmap="gray")
	direction_axis.set_title("Flow Direction: Red Arrows")

	for segment_id, segment in network.segments.items():
		connection = result.topology.downstream_connections.get(segment_id)

		if connection is None:
			continue

		if abs(connection.flow_um3_per_s) <= 0.0:
			print(
				f"segment {segment_id} has topology but zero flow "
				f"({connection.from_node} -> {connection.to_node})"
			)

		start_pressure = flow_solution.node_pressures_pa.get(connection.from_node)
		end_pressure = flow_solution.node_pressures_pa.get(connection.to_node)

		if start_pressure is None or end_pressure is None:
			continue

		centerline = orient_centerline(segment, connection.from_node, connection.to_node)

		if len(centerline) < 2:
			continue

		pressure = 0.5 * (start_pressure + end_pressure)
		pressure_scale = (pressure - min_pressure) / max(max_pressure - min_pressure, 1.0e-12)
		flow_scale = abs(connection.flow_um3_per_s) / max(max_flow, 1.0e-12)
		rows = np.asarray([point[0] for point in centerline], dtype=float)
		cols = np.asarray([point[1] for point in centerline], dtype=float)
		pressure_axis.plot(
			cols,
			rows,
			linewidth=1.0 + 2.0 * flow_scale,
			color=plt.cm.viridis(pressure_scale),
		)
		direction_axis.plot(
			cols,
			rows,
			linewidth=0.75,
			color="white",
			alpha=0.45,
		)
		draw_flow_arrow(direction_axis, centerline)

	pressure_image = pressure_axis.imshow(
		np.full_like(result.geometry.reconstructed_mask, np.nan, dtype=float),
		cmap="viridis",
		vmin=min_pressure,
		vmax=max_pressure,
	)
	fig.colorbar(
		pressure_image,
		ax=pressure_axis,
		fraction=0.046,
		pad=0.04,
		label="Pressure (mmHg)",
	)


# Orient a centerline so arrows use perfusion topology direction.
def orient_centerline(segment, from_node: int, to_node: int):

	centerline = segment.centerline

	if segment.start_node == from_node and segment.end_node == to_node:
		return centerline

	if segment.start_node == to_node and segment.end_node == from_node:
		return list(reversed(centerline))

	return centerline


# Draw one same-sized arrow along the middle of a segment centerline.
def draw_flow_arrow(axis, centerline):

	arrow_distance = max(8, len(centerline) // 7)
	mid_index = len(centerline) // 2
	start_index = max(0, mid_index - arrow_distance)
	end_index = min(len(centerline) - 1, mid_index + arrow_distance)
	start_row, start_col = centerline[start_index]
	end_row, end_col = centerline[end_index]

	axis.annotate(
		"",
		xy=(end_col, end_row),
		xytext=(start_col, start_row),
		arrowprops={
			"arrowstyle": "-|>",
			"linewidth": 1.4,
			"color": "red",
			"mutation_scale": 14.0,
			"shrinkA": 0.0,
			"shrinkB": 0.0,
		},
	)




# Pure visualization. None of this belongs in the actual simulation loop.
def plot_combined_results(mask_set: dict[str, np.ndarray], result: VesselTransportResult) -> None:

	boundary_sources = result.boundary_sources
	oxygen_display, carbon_dioxide_display = build_boundary_source_display_maps(result)
	fig, axes = plt.subplots(2, 5, figsize=(22, 8))

	axes[0, 0].imshow(mask_set["centerline"], cmap="gray")
	axes[0, 0].set_title("Real VeSeg Centerline")

	axes[0, 1].imshow(mask_set["vessel"], cmap="gray")
	axes[0, 1].set_title("Real VeSeg Reconstructed Vessel")

	axes[0, 2].imshow(mask_set["outline_with_skeleton"])
	axes[0, 2].set_title("Reconstructed Outline + Skeleton")

	axes[0, 3].imshow(build_perfused_component_overlay(result))
	axes[0, 3].set_title("Perfusion Topology: Green = Used, Red = Ignored")

	oxygen_image = axes[1, 0].imshow(oxygen_display, cmap="viridis", vmin=0.0, vmax=1.0)
	axes[1, 0].set_title("Oxygen Boundary Source")
	fig.colorbar(oxygen_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

	carbon_dioxide_image = axes[1, 1].imshow(
		carbon_dioxide_display,
		cmap="coolwarm",
		vmin=-1.0,
		vmax=1.0,
	)
	axes[1, 1].set_title("Carbon Dioxide Boundary Sink")
	fig.colorbar(carbon_dioxide_image, ax=axes[1, 1], fraction=0.046, pad=0.04)

	axes[1, 2].imshow(mask_set["outline_with_skeleton"])
	axes[1, 2].imshow(oxygen_display, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.75)
	axes[1, 2].set_title("Oxygen Source Overlay")

	axes[1, 3].imshow(mask_set["outline_with_skeleton"])
	axes[1, 3].imshow(carbon_dioxide_display, cmap="coolwarm", vmin=-1.0, vmax=1.0, alpha=0.75)
	axes[1, 3].set_title("CO2 Sink Overlay")

	plot_flow_direction_map(result, axes[0, 4], axes[1, 4], fig)

	for axis in axes.flat:
		axis.set_axis_off()

	plt.tight_layout()
	plt.show()


def main() -> None:

	if not DEFAULT_REAL_IMAGE_PATH.exists():
		print(f"missing image: {DEFAULT_REAL_IMAGE_PATH}")
		return

	result = build_vessel_transport_pipeline_from_image(DEFAULT_REAL_IMAGE_PATH)

	summarize_network(result)
	mask_set = build_debug_masks(result)
	summarize_boundary_pixel_map(result)
	summarize_flow(result)
	summarize_boundary_sources(result)
	plot_combined_results(mask_set, result)


if __name__ == "__main__":
	main()
