from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import veseg
import vessel_solver as rust_vessel_solver

DEFAULT_BLOOD_VISCOSITY_PA_S = 0.0035
DEFAULT_INLET_PRESSURE_MMHG = 35.0
DEFAULT_OUTLET_PRESSURE_MMHG = 15.0


@dataclass
class VesselGeometryResult:
	raw_mask: np.ndarray
	geometry_mask: np.ndarray
	skeleton: np.ndarray
	distance_map: np.ndarray
	nodes: np.ndarray
	edges: np.ndarray
	reconstructed_mask: np.ndarray
	network: Any | None = None


@dataclass
class VesselTransportResult:
	geometry: VesselGeometryResult
	rust_summary: str
	flow_solution: Any | None = None
	topology: Any | None = None
	boundary_pixel_map: dict[int, list[tuple[int, int]]] | None = None
	segment_concentrations: dict[int, Any] | None = None
	boundary_sources: Any | None = None


@dataclass
class VesselNode:
	id: int
	row: float
	col: float


@dataclass
class VesselSegment:
	id: int
	start_node: int
	end_node: int
	length_um: float
	radius_um: float
	centerline: list[tuple[float, float]]


@dataclass
class VesselNetwork:
	nodes: dict[int, VesselNode]
	segments: dict[int, VesselSegment]

	def total_length_um(self):
		return sum(segment.length_um for segment in self.segments.values())

	def total_volume_um3(self):
		return sum(
			np.pi * segment.radius_um ** 2 * segment.length_um
			for segment in self.segments.values()
		)


@dataclass
class SegmentConcentration:
	segment_id: int
	oxygen: float
	carbon_dioxide: float


@dataclass
class BoundarySourceMaps:
	oxygen: np.ndarray
	carbon_dioxide: np.ndarray


@dataclass
class SegmentFlow:
	segment_id: int
	flow_um3_per_s: float


@dataclass
class FlowSolution:
	node_pressures_pa: dict[int, float]
	segment_flows: dict[int, SegmentFlow]


@dataclass
class DownstreamConnection:
	segment_id: int
	from_node: int
	to_node: int
	flow_um3_per_s: float


@dataclass
class DirectedTopology:
	inlet_nodes: list[int]
	outlet_nodes: list[int]
	traversal_segments: list[int]
	downstream_connections: dict[int, DownstreamConnection]


# VeSeg can return either a raw mask or a wrapper with a mask field.
def prediction_mask(prediction):
	if hasattr(prediction, "mask"):
		return prediction.mask

	return prediction


# Build vessel geometry from a binary mask using VeSeg.
def build_vessel_geometry(raw_mask, shape=None) -> VesselGeometryResult:
	raw_mask = prediction_mask(raw_mask)

	if shape is not None and tuple(shape) != tuple(raw_mask.shape):
		raw_mask = resize_binary_mask_nearest(raw_mask, shape)

	geometry_mask, skeleton, distance_map, nodes, edges = veseg.extract_vessel_geometry(raw_mask)
	reconstructed_mask = veseg.reconstruct_vessel_mask(
		skeleton,
		distance_map,
	).astype(bool)
	network = build_network_from_veseg(nodes, edges, skeleton, distance_map)

	return VesselGeometryResult(
		raw_mask=raw_mask,
		geometry_mask=geometry_mask,
		skeleton=skeleton,
		distance_map=distance_map,
		nodes=nodes,
		edges=edges,
		reconstructed_mask=reconstructed_mask,
		network=network,
	)


# Resize the filled vessel mask first, then derive skeleton/graph at target scale.
def resize_binary_mask_nearest(mask, target_shape):
	mask = np.asarray(mask, dtype=bool)
	target_h, target_w = target_shape
	source_h, source_w = mask.shape
	row_indices = np.clip(
		(np.arange(target_h) * source_h / target_h).astype(int),
		0,
		source_h - 1,
	)
	col_indices = np.clip(
		(np.arange(target_w) * source_w / target_w).astype(int),
		0,
		source_w - 1,
	)

	return mask[row_indices[:, None], col_indices[None, :]]


# Scale VeSeg nodes and edges from one image resolution into another.
def scale_vessel_graph(nodes, edges, source_shape, target_shape):
	source_h, source_w = source_shape
	target_h, target_w = target_shape
	row_scale = target_h / source_h
	col_scale = target_w / source_w
	mean_scale = 0.5 * (row_scale + col_scale)

	scaled_nodes = np.array(nodes, dtype=float, copy=True)
	scaled_edges = np.array(edges, dtype=float, copy=True)

	# Expected node layout: [id, row, col, kind].
	scaled_nodes[:, 1] *= row_scale
	scaled_nodes[:, 2] *= col_scale

	# Expected edge layout: [id, start_node, end_node, length, radius, ...].
	if scaled_edges.shape[1] > 3:
		scaled_edges[:, 3:] *= mean_scale

	return scaled_nodes, scaled_edges


# Scale sparse skeleton and radius map for reconstructed mask generation.
def scale_skeleton_distance(skeleton, distance_map, source_shape, target_shape):
	source_h, source_w = source_shape
	target_h, target_w = target_shape
	row_scale = target_h / source_h
	col_scale = target_w / source_w
	mean_scale = 0.5 * (row_scale + col_scale)

	scaled_skeleton = np.zeros(target_shape, dtype=bool)
	scaled_distance_map = np.zeros(target_shape, dtype=float)
	rows, cols = np.nonzero(skeleton)

	if rows.size == 0:
		return scaled_skeleton, scaled_distance_map

	target_rows = np.clip(np.rint(rows * row_scale).astype(int), 0, target_h - 1)
	target_cols = np.clip(np.rint(cols * col_scale).astype(int), 0, target_w - 1)

	scaled_skeleton[target_rows, target_cols] = True
	scaled_distance_map[target_rows, target_cols] = distance_map[rows, cols] * mean_scale

	return scaled_skeleton, scaled_distance_map


# Convert VeSeg nodes into Rust node objects.
def rust_nodes_from_veseg(nodes):
	return [
		rust_vessel_solver.PyVesselNode(
			int(node[0]),
			float(node[2]),
			float(node[1]),
		)
		for node in nodes
	]


# Convert VeSeg edges into Rust segment objects.
def rust_segments_from_veseg(edges, network=None):
	segments = []

	for edge in edges:
		segment_id = int(edge[0])
		length_um = float(edge[3]) if len(edge) > 3 else 1.0
		radius_um = float(edge[4]) if len(edge) > 4 else 1.0
		centerline = []

		if network is not None and segment_id in network.segments:
			segment = network.segments[segment_id]
			length_um = segment.length_um
			radius_um = segment.radius_um
			centerline = [
				rust_vessel_solver.PyVesselPoint(
					float(col),
					float(row),
				)
				for row, col in segment.centerline
			]

		segments.append(
			rust_vessel_solver.PyVesselSegment(
				segment_id,
				int(edge[1]),
				int(edge[2]),
				length_um,
				radius_um,
				centerline,
			)
		)

	return segments


# Pick pressure boundary pairs for every disconnected vessel component.
def pressure_boundaries_from_graph(nodes, edges):
	components = graph_components(nodes, edges)
	boundaries = []

	for component in components:
		component_nodes = sorted(component["nodes"])

		if len(component_nodes) < 2:
			continue

		inlet_node, outlet_node = component_boundary_nodes(component_nodes, nodes)
		boundaries.append(
			rust_vessel_solver.PyPressureBoundary(
				int(inlet_node),
				DEFAULT_INLET_PRESSURE_MMHG,
			)
		)
		boundaries.append(
			rust_vessel_solver.PyPressureBoundary(
				int(outlet_node),
				DEFAULT_OUTLET_PRESSURE_MMHG,
			)
		)

	if not boundaries:
		raise ValueError("vessel graph needs at least one valid component for pressure boundaries")

	return boundaries


# Split the vessel graph into disconnected components.
def graph_components(nodes, edges):
	adjacency = {int(node[0]): set() for node in nodes}

	for edge in edges:
		start_node = int(edge[1])
		end_node = int(edge[2])
		adjacency.setdefault(start_node, set()).add(end_node)
		adjacency.setdefault(end_node, set()).add(start_node)

	visited = set()
	components = []

	for start_node in adjacency:
		if start_node in visited:
			continue

		queue = [start_node]
		visited.add(start_node)
		component_nodes = set()

		while queue:
			node_id = queue.pop(0)
			component_nodes.add(node_id)

			for neighbor in adjacency.get(node_id, set()):
				if neighbor not in visited:
					visited.add(neighbor)
					queue.append(neighbor)

		components.append({"nodes": component_nodes})

	return components


# Choose inlet/outlet nodes inside one connected component.
def component_boundary_nodes(component_nodes, nodes):
	component_nodes = sorted(component_nodes)
	node_positions = {
		int(node[0]): (float(node[1]), float(node[2]))
		for node in nodes
	}

	if len(component_nodes) < 2:
		raise ValueError("component needs at least two nodes for pressure boundaries")

	best_pair = (component_nodes[0], component_nodes[-1])
	best_distance = -1.0

	for index, first_node in enumerate(component_nodes):
		for second_node in component_nodes[index + 1:]:
			first_row, first_col = node_positions[first_node]
			second_row, second_col = node_positions[second_node]
			distance = (second_row - first_row) ** 2 + (second_col - first_col) ** 2

			if distance > best_distance:
				best_distance = distance
				best_pair = (first_node, second_node)

	return best_pair


# Run the Rust-backed vessel pipeline from an already-built mask.
def build_vessel_transport_pipeline(
	raw_mask,
	shape=None,
	viscosity_pa_s: float = DEFAULT_BLOOD_VISCOSITY_PA_S,
) -> VesselTransportResult:
	geometry = build_vessel_geometry(raw_mask, shape=shape)
	nodes = rust_nodes_from_veseg(geometry.nodes)
	segments = rust_segments_from_veseg(geometry.edges, geometry.network)
	boundaries = pressure_boundaries_from_graph(geometry.nodes, geometry.edges)
	target_shape = tuple(shape) if shape is not None else tuple(geometry.reconstructed_mask.shape)
	boundary_pixel_map = build_segment_pixel_map(geometry.network, target_shape)
	rust_output = rust_vessel_solver.solve_vessel_transport_from_python(
		nodes,
		segments,
		boundaries,
		float(viscosity_pa_s),
		int(target_shape[1]),
		int(target_shape[0]),
	)

	transport_result = VesselTransportResult(
		geometry=geometry,
		rust_summary=rust_output["summary"],
		flow_solution=flow_solution_from_rust_output(rust_output),
		topology=topology_from_rust_output(rust_output),
		boundary_pixel_map=boundary_pixel_map,
		segment_concentrations=segment_concentrations_from_rust_output(rust_output),
		boundary_sources=boundary_sources_from_rust_output(rust_output, target_shape),
	)

	oxygen_source_map, carbon_dioxide_sink_map = build_boundary_source_display_maps(
		transport_result,
		normalize=True,
	)
	transport_result.boundary_sources = BoundarySourceMaps(
		oxygen=oxygen_source_map,
		carbon_dioxide=carbon_dioxide_sink_map,
	)

	return transport_result


# Convenience helper for callers that only have an image path.
def build_vessel_transport_pipeline_from_image(
	image_path: str | Path,
	shape=None,
	threshold: float | None = None,
	**kwargs: Any,
) -> VesselTransportResult:
	image_path = Path(image_path)

	if threshold is None:
		raw_mask = prediction_mask(veseg.predict(image_path))
	else:
		raw_mask = prediction_mask(veseg.predict(image_path, threshold=threshold))

	return build_vessel_transport_pipeline(
		raw_mask=raw_mask,
		shape=shape,
		**kwargs,
	)


# Build the lightweight Python network used by smoke-test visualization only.
def build_network_from_veseg(nodes, edges, skeleton, distance_map):
	node_map = {
		int(node[0]): VesselNode(
			id=int(node[0]),
			row=float(node[1]),
			col=float(node[2]),
		)
		for node in nodes
	}
	centerlines = extract_edge_centerlines_from_skeleton(skeleton, node_map, edges)
	segment_map = {}

	for edge in edges:
		segment_id = int(edge[0])
		start_node = int(edge[1])
		end_node = int(edge[2])
		centerline = centerlines.get(segment_id)

		if not centerline:
			start = node_map[start_node]
			end = node_map[end_node]
			centerline = sample_line_points(start.row, start.col, end.row, end.col)

		length_um = centerline_length_um(centerline)
		radius_um = mean_radius_from_distance_map(centerline, distance_map)

		segment_map[segment_id] = VesselSegment(
			id=segment_id,
			start_node=start_node,
			end_node=end_node,
			length_um=length_um,
			radius_um=radius_um,
			centerline=centerline,
		)

	return VesselNetwork(nodes=node_map, segments=segment_map)


# Trace each graph edge along the real skeleton instead of drawing straight lines.
def extract_edge_centerlines_from_skeleton(skeleton, node_map, edges):
	centerlines = {}

	for edge in edges:
		segment_id = int(edge[0])
		start_node = int(edge[1])
		end_node = int(edge[2])
		start = node_map[start_node]
		end = node_map[end_node]
		centerline = trace_skeleton_path(
			skeleton,
			(start.row, start.col),
			(end.row, end.col),
		)
		centerlines[segment_id] = centerline

	return centerlines


# Find a skeleton path between two graph nodes.
def trace_skeleton_path(skeleton, start_point, end_point):
	start_pixel = nearest_skeleton_pixel(skeleton, start_point)
	end_pixel = nearest_skeleton_pixel(skeleton, end_point)

	if start_pixel is None or end_pixel is None:
		return []

	queue = [start_pixel]
	previous = {start_pixel: None}

	while queue:
		pixel = queue.pop(0)

		if pixel == end_pixel:
			break

		for neighbor in skeleton_neighbors(skeleton, pixel):
			if neighbor not in previous:
				previous[neighbor] = pixel
				queue.append(neighbor)

	if end_pixel not in previous:
		return []

	path = []
	pixel = end_pixel

	while pixel is not None:
		path.append((float(pixel[0]), float(pixel[1])))
		pixel = previous[pixel]

	path.reverse()
	return path


# Find the nearest skeleton pixel to a graph node.
def nearest_skeleton_pixel(skeleton, point):
	rows, cols = np.nonzero(skeleton)

	if rows.size == 0:
		return None

	row, col = point
	distances = (rows - row) ** 2 + (cols - col) ** 2
	index = int(np.argmin(distances))

	return int(rows[index]), int(cols[index])


# Return 8-connected skeleton neighbors for one pixel.
def skeleton_neighbors(skeleton, pixel):
	row, col = pixel
	neighbors = []

	for row_offset in (-1, 0, 1):
		for col_offset in (-1, 0, 1):
			if row_offset == 0 and col_offset == 0:
				continue

			neighbor_row = row + row_offset
			neighbor_col = col + col_offset

			if 0 <= neighbor_row < skeleton.shape[0] and 0 <= neighbor_col < skeleton.shape[1]:
				if skeleton[neighbor_row, neighbor_col]:
					neighbors.append((neighbor_row, neighbor_col))

	return neighbors


# Estimate segment length from its traced centerline.
def centerline_length_um(centerline):
	if len(centerline) < 2:
		return 1.0

	length = 0.0

	for (row_a, col_a), (row_b, col_b) in zip(centerline[:-1], centerline[1:]):
		length += float(np.hypot(row_b - row_a, col_b - col_a))

	return max(length, 1.0)


# Estimate segment radius from the distance map along its centerline.
def mean_radius_from_distance_map(centerline, distance_map):
	radii = []

	for row, col in centerline:
		row = int(round(row))
		col = int(round(col))

		if 0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]:
			radius = float(distance_map[row, col])

			if radius > 0.0:
				radii.append(radius)

	if not radii:
		return 1.0

	return max(float(np.mean(radii)), 1.0)


# Fallback centerline sampler for smoke-test visualization.
def sample_line_points(start_row, start_col, end_row, end_col):
	delta_row = end_row - start_row
	delta_col = end_col - start_col
	steps = max(1, int(np.ceil(max(abs(delta_row), abs(delta_col)))))

	return [
		(
			start_row + delta_row * step / steps,
			start_col + delta_col * step / steps,
		)
		for step in range(steps + 1)
	]


# Build a sparse centerline mask from the vessel network.
def build_centerline_mask(network, shape):
	mask = np.zeros(shape, dtype=bool)

	for segment in network.segments.values():
		for row, col in segment.centerline:
			row = int(round(row))
			col = int(round(col))

			if 0 <= row < shape[0] and 0 <= col < shape[1]:
				mask[row, col] = True

	return mask


# Build a filled vessel mask from centerlines and segment radii.
def build_vessel_mask(network, shape):
	mask = np.zeros(shape, dtype=bool)

	for segment in network.segments.values():
		for row, col in segment.centerline:
			for fill_row, fill_col in disk_pixels(row, col, segment.radius_um, shape):
				mask[fill_row, fill_col] = True

	return mask


# Map each segment to the grid pixels covered by its vessel footprint.
def build_segment_pixel_map(network, shape):
	segment_pixel_map = {}

	for segment_id, segment in network.segments.items():
		pixels = set()

		for row, col in segment.centerline:
			pixels.update(disk_pixels(row, col, segment.radius_um, shape))

		segment_pixel_map[segment_id] = sorted(pixels)

	return segment_pixel_map


# Return integer pixels inside a radius around a centerline point.
def disk_pixels(row, col, radius, shape):
	radius = max(1, int(round(radius)))
	row = int(round(row))
	col = int(round(col))
	row_min = max(0, row - radius)
	row_max = min(shape[0], row + radius + 1)
	col_min = max(0, col - radius)
	col_max = min(shape[1], col + radius + 1)
	pixels = []

	for fill_row in range(row_min, row_max):
		for fill_col in range(col_min, col_max):
			if (fill_row - row) ** 2 + (fill_col - col) ** 2 <= radius ** 2:
				pixels.append((fill_row, fill_col))

	return pixels


# Summarize per-segment oxygen concentrations.
def summarize_segment_concentrations(segment_concentrations):
	oxygen_values = np.asarray([
		concentration.oxygen
		for concentration in segment_concentrations.values()
	], dtype=float)

	if oxygen_values.size == 0:
		return {
			"oxygen_min": 0.0,
			"oxygen_mean": 0.0,
			"oxygen_max": 0.0,
		}

	return {
		"oxygen_min": float(oxygen_values.min()),
		"oxygen_mean": float(oxygen_values.mean()),
		"oxygen_max": float(oxygen_values.max()),
	}


# Summarize oxygen and carbon dioxide source maps.
def summarize_boundary_sources(boundary_sources):
	oxygen = np.asarray(boundary_sources.oxygen, dtype=float)
	carbon_dioxide = np.asarray(boundary_sources.carbon_dioxide, dtype=float)
	oxygen_nonzero = oxygen[oxygen > 0.0]

	return {
		"oxygen_pixels": float(np.count_nonzero(oxygen)),
		"oxygen_total": float(oxygen.sum()),
		"oxygen_min_nonzero": float(oxygen_nonzero.min()) if oxygen_nonzero.size else 0.0,
		"oxygen_max": float(oxygen.max()) if oxygen.size else 0.0,
		"carbon_dioxide_total": float(carbon_dioxide.sum()),
		"carbon_dioxide_max": float(carbon_dioxide.max()) if carbon_dioxide.size else 0.0,
	}


# Build reconstructed-vessel-shaped oxygen source and carbon dioxide sink maps.
def build_boundary_source_display_maps(result: VesselTransportResult, normalize=True):
	shape = result.geometry.reconstructed_mask.shape
	vessel_mask = np.asarray(result.geometry.reconstructed_mask, dtype=bool)
	oxygen_seed_map = np.zeros(shape, dtype=float)
	carbon_dioxide_seed_map = np.zeros(shape, dtype=float)
	seed_mask = np.zeros(shape, dtype=bool)

	for segment_id, boundary_pixels in result.boundary_pixel_map.items():
		concentration = result.segment_concentrations.get(segment_id)

		if concentration is None or not boundary_pixels:
			continue

		boundary_pixels = np.asarray(boundary_pixels, dtype=int)
		boundary_rows = boundary_pixels[:, 0]
		boundary_cols = boundary_pixels[:, 1]
		valid_pixels = vessel_mask[boundary_rows, boundary_cols]

		if not np.any(valid_pixels):
			continue

		boundary_rows = boundary_rows[valid_pixels]
		boundary_cols = boundary_cols[valid_pixels]
		oxygen_seed_map[boundary_rows, boundary_cols] = float(concentration.oxygen)
		carbon_dioxide_seed_map[boundary_rows, boundary_cols] = float(concentration.carbon_dioxide)
		seed_mask[boundary_rows, boundary_cols] = True

	oxygen_display, carbon_dioxide_display = flood_fill_reconstructed_vessel_values(
		vessel_mask,
		seed_mask,
		oxygen_seed_map,
		carbon_dioxide_seed_map,
	)

	oxygen_display = smooth_reconstructed_vessel_values(vessel_mask, oxygen_display, iterations=60)
	carbon_dioxide_display = smooth_reconstructed_vessel_values(vessel_mask, carbon_dioxide_display, iterations=60)

	if normalize:
		oxygen_display = normalize_nonzero_display_values(oxygen_display)
		carbon_dioxide_display = normalize_nonzero_display_values(carbon_dioxide_display)

	return oxygen_display, carbon_dioxide_display


# Fill every reconstructed vessel pixel from nearby seeded segment values.
def flood_fill_reconstructed_vessel_values(vessel_mask, seed_mask, oxygen_seed_map, carbon_dioxide_seed_map):
	vessel_mask = np.asarray(vessel_mask, dtype=bool)
	filled_mask = np.asarray(seed_mask & vessel_mask, dtype=bool)
	oxygen_display = np.where(filled_mask, oxygen_seed_map, 0.0).astype(float)
	carbon_dioxide_display = np.where(filled_mask, carbon_dioxide_seed_map, 0.0).astype(float)
	unfilled_mask = vessel_mask & ~filled_mask

	while np.any(unfilled_mask):
		oxygen_sum, carbon_dioxide_sum, neighbor_count = neighbor_value_sums(
			oxygen_display,
			carbon_dioxide_display,
			filled_mask,
		)
		frontier = unfilled_mask & (neighbor_count > 0)

		if not np.any(frontier):
			break

		oxygen_display[frontier] = oxygen_sum[frontier] / neighbor_count[frontier]
		carbon_dioxide_display[frontier] = carbon_dioxide_sum[frontier] / neighbor_count[frontier]
		filled_mask[frontier] = True
		unfilled_mask = vessel_mask & ~filled_mask

	return oxygen_display, carbon_dioxide_display




# Vectorized 8-neighbor sums for values inside a mask.
def neighbor_value_sums(first_values, second_values, source_mask):
	first_sum = np.zeros_like(first_values, dtype=float)
	second_sum = np.zeros_like(second_values, dtype=float)
	neighbor_count = np.zeros_like(first_values, dtype=float)

	for row_offset in (-1, 0, 1):
		for col_offset in (-1, 0, 1):
			shifted_mask = shifted_array(source_mask, row_offset, col_offset, fill_value=False)
			first_sum += shifted_array(first_values, row_offset, col_offset, fill_value=0.0) * shifted_mask
			second_sum += shifted_array(second_values, row_offset, col_offset, fill_value=0.0) * shifted_mask
			neighbor_count += shifted_mask.astype(float)

	return first_sum, second_sum, neighbor_count


# Shift an array without wraparound.
def shifted_array(values, row_offset, col_offset, fill_value=0.0):
	shifted = np.full_like(values, fill_value)
	row_source_start = max(0, -row_offset)
	row_source_end = values.shape[0] - max(0, row_offset)
	col_source_start = max(0, -col_offset)
	col_source_end = values.shape[1] - max(0, col_offset)
	row_target_start = max(0, row_offset)
	row_target_end = values.shape[0] - max(0, -row_offset)
	col_target_start = max(0, col_offset)
	col_target_end = values.shape[1] - max(0, -col_offset)

	if row_source_start >= row_source_end or col_source_start >= col_source_end:
		return shifted

	shifted[
		row_target_start:row_target_end,
		col_target_start:col_target_end,
	] = values[
		row_source_start:row_source_end,
		col_source_start:col_source_end,
	]

	return shifted


# Normalize only visualization copies. Real source/sink values stay untouched.
def normalize_nonzero_display_values(values: np.ndarray) -> np.ndarray:
	values = np.asarray(values, dtype=float).copy()
	nonzero_mask = np.abs(values) > 0.0

	if not np.any(nonzero_mask):
		return values

	nonzero_values = values[nonzero_mask]
	min_value = nonzero_values.min()
	max_value = nonzero_values.max()

	if np.isclose(max_value, min_value):
		values[nonzero_mask] = 1.0
		return values

	values[nonzero_mask] = (nonzero_values - min_value) / (max_value - min_value)
	return values


# Convert real Rust flow output into the shape expected by the smoke test.
def flow_solution_from_rust_output(rust_output):
	flow = rust_output["flow"]
	node_pressures_pa = {
		int(node_pressure["node_id"]): float(node_pressure["pressure_mmhg"])
		for node_pressure in flow["node_pressures"]
	}
	segment_flows = {
		int(segment_flow["segment_id"]): SegmentFlow(
			segment_id=int(segment_flow["segment_id"]),
			flow_um3_per_s=float(segment_flow["flow_um3_per_s"]),
		)
		for segment_flow in flow["segment_flows"]
	}

	return FlowSolution(
		node_pressures_pa=node_pressures_pa,
		segment_flows=segment_flows,
	)


# Convert Rust perfusion topology output into Python direction data.
def topology_from_rust_output(rust_output):
	topology = rust_output.get("topology", {})
	connections = {}

	for connection in topology.get("downstream_connections", []):
		segment_id = int(connection["segment_id"])
		connections[segment_id] = DownstreamConnection(
			segment_id=segment_id,
			from_node=int(connection["from_node"]),
			to_node=int(connection["to_node"]),
			flow_um3_per_s=float(connection["flow_um3_per_s"]),
		)

	traversal_segments = [
		int(segment_id)
		for segment_id in topology.get("traversal_segments", [])
	]

	# Older Rust bindings may not expose traversal_segments yet.
	# The downstream connection list is the actual perfusion topology, so use it
	# as the Python-side traversal list when the explicit list is missing.
	if not traversal_segments:
		traversal_segments = list(connections.keys())

	return DirectedTopology(
		inlet_nodes=[int(node_id) for node_id in topology.get("inlet_nodes", [])],
		outlet_nodes=[int(node_id) for node_id in topology.get("outlet_nodes", [])],
		traversal_segments=traversal_segments,
		downstream_connections=connections,
	)


# Convert real Rust species transport output into per-segment concentrations.
def segment_concentrations_from_rust_output(rust_output):
	concentrations = {}

	for species_result in rust_output["species_transport"]:
		species = normalized_species_name(species_result["species"])

		for state in species_result["segment_states"]:
			segment_id = int(state["segment_id"])
			concentration = concentrations.setdefault(
				segment_id,
				SegmentConcentration(
					segment_id=segment_id,
					oxygen=0.0,
					carbon_dioxide=0.0,
				),
			)

			if species == "Oxygen":
				concentration.oxygen = float(state["outlet_concentration"])
			elif species == "CarbonDioxide":
				concentration.carbon_dioxide = float(state["outlet_concentration"])

	return concentrations


# Convert real Rust boundary source maps into dense NumPy arrays.
def boundary_sources_from_rust_output(rust_output, shape):
	oxygen = np.zeros(shape, dtype=float)
	carbon_dioxide = np.zeros(shape, dtype=float)

	for source_map in rust_output["boundary_source_maps"]["maps"]:
		species = normalized_species_name(source_map["species"])
		width = int(source_map["width"])
		height = int(source_map["height"])
		values = np.asarray(source_map["values"], dtype=float).reshape((height, width))

		if values.shape != shape:
			raise ValueError(
				f"Rust source map shape {values.shape} does not match expected shape {shape}"
			)

		if species == "Oxygen":
			oxygen = values
		elif species == "CarbonDioxide":
			carbon_dioxide = values

	return BoundarySourceMaps(
		oxygen=oxygen,
		carbon_dioxide=carbon_dioxide,
	)


# Normalize Rust enum debug names into Python adapter names.
def normalized_species_name(species):
	name = str(species).lower()
	name = name.replace("_", "")
	name = name.replace(" ", "")
	name = name.replace("-", "")

	if "oxygen" in name:
		return "Oxygen"

	if "carbondioxide" in name or "co2" in name or "carbon" in name:
		return "CarbonDioxide"

	return str(species)

def smooth_reconstructed_vessel_values(vessel_mask, values, iterations=25):
	vessel_mask = np.asarray(vessel_mask, dtype=bool)
	values = np.asarray(values, dtype=float).copy()

	if not np.any(vessel_mask):
		return values

	for _ in range(iterations):
		value_sum, _, neighbor_count = neighbor_value_sums(values, values, vessel_mask)
		updated_values = np.zeros_like(values, dtype=float)
		valid_mask = vessel_mask & (neighbor_count > 0)
		updated_values[valid_mask] = value_sum[valid_mask] / neighbor_count[valid_mask]
		values = updated_values

	values[~vessel_mask] = 0.0
	return values