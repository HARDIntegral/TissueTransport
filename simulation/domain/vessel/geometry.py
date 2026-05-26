"""
Geometry helpers for vessel centerlines and radii.

This module handles low-level geometry operations used by the vessel network,
flow solver, and tissue coupling code. It does not store a vessel network and it
should not know anything about pressure or transport state.

Main idea:
	The network uses centerlines and radii as its geometric backbone.
	This file validates those arrays, samples radii from masks, traces skeleton
	paths, and computes simple cylindrical vessel approximations.

Coordinate convention:
	centerline[:, 0] -> x coordinate / image column
	centerline[:, 1] -> y coordinate / image row
	skeleton pixels   -> (row, col)

Notes:
	The centerline is the 1D path used for graph flow.
	The reconstructed vessel mask/boundary is the 2D geometry used for tissue
	exchange.

	Most of these functions are preprocessing utilities. They should usually run
	once when the vessel image is loaded, not every timestep.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .constants import DEFAULT_SCALE_UM_PER_PIXEL

Pixel = tuple[int, int]
SkeletonGraph = dict[Pixel, list[Pixel]]


# --- Centerline validation and measurement ---

def as_centerline_array(centerline: np.ndarray) -> np.ndarray:
	"""
	Convert a centerline into a validated ``(n_points, 2)`` float array.

	Parameters:
		centerline -> vessel path coordinates in ``(x, y)`` order

	Returns:
		validated centerline array with shape ``(n_points, 2)``

	Notes:
		Keeping centerlines consistently shaped here prevents flow/coupling code from
		having to repeatedly check input geometry.
	"""

	centerline = np.asarray(centerline, dtype=float)

	if centerline.ndim != 2 or centerline.shape[1] != 2:
		raise ValueError("centerline must have shape (n_points, 2)")

	return centerline


def compute_centerline_length_px(centerline: np.ndarray) -> float:
	"""
	Compute centerline length in pixels.
	"""

	centerline = as_centerline_array(centerline)

	if len(centerline) < 2:
		return 0.0

	steps = np.diff(centerline, axis=0)
	return float(np.linalg.norm(steps, axis=1).sum())


def compute_centerline_length_um(
	centerline: np.ndarray,
	scale_um_per_pixel: float = DEFAULT_SCALE_UM_PER_PIXEL,
) -> float:
	"""
	Compute centerline length in micrometers.
	"""

	return compute_centerline_length_px(centerline) * scale_um_per_pixel


def sample_radii_from_map(centerline: np.ndarray, radius_map: np.ndarray) -> np.ndarray:
	"""
	Sample vessel radii from a radius map along a centerline.

	Parameters:
		centerline -> vessel path in ``(x, y)`` order
		radius_map -> 2D map where each vessel pixel stores local radius

	Returns:
		radius values sampled along the centerline

	Notes:
		The centerline uses ``(x, y)`` coordinates, but images are indexed as
		``[row, col]`` or ``[y, x]``. This function handles that conversion in one
		place so the rest of the code does not mix coordinate systems.
	"""

	centerline = as_centerline_array(centerline)
	radius_map = np.asarray(radius_map, dtype=float)

	if radius_map.ndim != 2:
		raise ValueError("radius_map must be a 2D array")

	x_coords = np.rint(centerline[:, 0]).astype(int)
	y_coords = np.rint(centerline[:, 1]).astype(int)

	x_coords = np.clip(x_coords, 0, radius_map.shape[1] - 1)
	y_coords = np.clip(y_coords, 0, radius_map.shape[0] - 1)

	return radius_map[y_coords, x_coords]


# --- Skeleton tracing and edge centerline extraction ---

def extract_edge_centerlines_from_skeleton(
	skeleton: np.ndarray,
	nodes: np.ndarray,
	edges: np.ndarray,
) -> dict[int, np.ndarray]:
	"""
	Trace per-edge centerline paths from a skeleton image.

	Parameters:
		skeleton -> global vessel skeleton mask
		nodes    -> VeSeg node table, usually ``[id, row, col, kind]``
		edges    -> VeSeg edge table, usually ``[id, start_node, end_node, ...]``

	Returns:
		dict mapping edge id to centerline array in ``(x, y)`` order

	Notes:
		VeSeg gives a compact graph and one global skeleton. For flow, each vessel
		segment needs its own ordered path. This function maps each edge back onto
		the skeleton and extracts that path.

		The skeleton graph is built once and reused for all edges, instead of scanning
		image neighbors from scratch during every BFS step.
	"""

	skeleton = _validate_skeleton(skeleton)
	nodes = _validate_geometry_table(nodes, "nodes", min_columns=3)
	edges = _validate_geometry_table(edges, "edges", min_columns=3)

	node_pixels = _build_node_pixel_lookup(nodes)
	skeleton_graph = _build_skeleton_graph(skeleton)
	centerlines: dict[int, np.ndarray] = {}

	for edge in edges:
		edge_id = int(edge[0])
		start_node = int(edge[1])
		end_node = int(edge[2])

		path_pixels = trace_skeleton_path(
			skeleton=skeleton,
			start_pixel=node_pixels[start_node],
			end_pixel=node_pixels[end_node],
			skeleton_graph=skeleton_graph,
		)

		centerlines[edge_id] = _pixels_to_centerline(path_pixels)

	return centerlines


def trace_skeleton_path(
	skeleton: np.ndarray,
	start_pixel: Pixel,
	end_pixel: Pixel,
	skeleton_graph: SkeletonGraph | None = None,
) -> list[Pixel]:
	"""
	Trace a shortest path between two skeleton pixels.

	Parameters:
		skeleton       -> global vessel skeleton mask
		start_pixel    -> expected start pixel in ``(row, col)`` order
		end_pixel      -> expected end pixel in ``(row, col)`` order
		skeleton_graph -> optional precomputed skeleton adjacency map

	Returns:
		ordered path of pixels in ``(row, col)`` order

	Notes:
		Nodes do not always land exactly on skeleton pixels, so the start/end points
		are snapped to the nearest nearby skeleton pixel first.
	"""

	skeleton = _validate_skeleton(skeleton)
	skeleton_graph = skeleton_graph or _build_skeleton_graph(skeleton)

	start_pixel = _nearest_skeleton_pixel(skeleton, start_pixel)
	end_pixel = _nearest_skeleton_pixel(skeleton, end_pixel)

	queue: deque[Pixel] = deque([start_pixel])
	parents: dict[Pixel, Pixel | None] = {start_pixel: None}

	while queue:
		current = queue.popleft()

		if current == end_pixel:
			return _reconstruct_pixel_path(parents, end_pixel)

		for neighbor in skeleton_graph.get(current, []):
			if neighbor in parents:
				continue

			parents[neighbor] = current
			queue.append(neighbor)

	raise ValueError("no skeleton path found between the requested nodes")


def _validate_skeleton(skeleton: np.ndarray) -> np.ndarray:
	"""
	Validate and convert a skeleton mask.
	"""

	skeleton = np.asarray(skeleton, dtype=bool)

	if skeleton.ndim != 2:
		raise ValueError("skeleton must be a 2D array")

	return skeleton


def _validate_geometry_table(
	table: np.ndarray,
	name: str,
	min_columns: int,
) -> np.ndarray:
	"""
	Validate a VeSeg node/edge table.
	"""

	table = np.asarray(table)

	if table.ndim != 2 or table.shape[1] < min_columns:
		raise ValueError(f"{name} must have shape (n_rows, {min_columns}) or wider")

	return table


def _build_node_pixel_lookup(nodes: np.ndarray) -> dict[int, Pixel]:
	"""
	Build ``node_id -> (row, col)`` lookup from a VeSeg node table.
	"""

	return {
		int(node[0]): (int(round(node[1])), int(round(node[2])))
		for node in nodes
	}


def _build_skeleton_graph(skeleton: np.ndarray) -> SkeletonGraph:
	"""
	Build an 8-connected graph from all skeleton pixels.

	Notes:
		This is faster than repeatedly checking image neighborhoods during every path
		trace, especially when many edges are being traced from the same skeleton.
	"""

	skeleton_pixels = np.argwhere(skeleton)
	pixel_set = {
		(int(row), int(col))
		for row, col in skeleton_pixels
	}
	graph: SkeletonGraph = {}

	for pixel in pixel_set:
		graph[pixel] = _pixel_neighbors_from_set(pixel, pixel_set)

	return graph


def _pixel_neighbors_from_set(pixel: Pixel, pixel_set: set[Pixel]) -> list[Pixel]:
	"""
	Return 8-connected neighbors using a precomputed skeleton pixel set.
	"""

	row, col = pixel
	neighbors: list[Pixel] = []

	for row_offset in (-1, 0, 1):
		for col_offset in (-1, 0, 1):
			if row_offset == 0 and col_offset == 0:
				continue

			neighbor = (row + row_offset, col + col_offset)

			if neighbor in pixel_set:
				neighbors.append(neighbor)

	return neighbors


def _nearest_skeleton_pixel(
	skeleton: np.ndarray,
	pixel: Pixel,
	search_radius: int = 3,
) -> Pixel:
	"""
	Find the closest skeleton pixel near an expected node location.
	"""

	height, width = skeleton.shape
	row, col = pixel

	row = int(np.clip(row, 0, height - 1))
	col = int(np.clip(col, 0, width - 1))

	if skeleton[row, col]:
		return (row, col)

	row_min = max(0, row - search_radius)
	row_max = min(height, row + search_radius + 1)
	col_min = max(0, col - search_radius)
	col_max = min(width, col + search_radius + 1)

	candidate_rows, candidate_cols = np.nonzero(skeleton[row_min:row_max, col_min:col_max])

	if candidate_rows.size == 0:
		raise ValueError("could not find a nearby skeleton pixel")

	candidate_rows = candidate_rows + row_min
	candidate_cols = candidate_cols + col_min
	distances = (candidate_rows - row) ** 2 + (candidate_cols - col) ** 2
	best_index = int(np.argmin(distances))

	return (int(candidate_rows[best_index]), int(candidate_cols[best_index]))


def _reconstruct_pixel_path(
	parents: dict[Pixel, Pixel | None],
	end_pixel: Pixel,
) -> list[Pixel]:
	"""
	Reconstruct a traced skeleton path from parent pointers.
	"""

	path: list[Pixel] = []
	current: Pixel | None = end_pixel

	while current is not None:
		path.append(current)
		current = parents[current]

	path.reverse()
	return path


def _pixels_to_centerline(pixels: list[Pixel]) -> np.ndarray:
	"""
	Convert ``(row, col)`` pixels into ``(x, y)`` centerline coordinates.
	"""

	return np.asarray(
		[
			[col, row]
			for row, col in pixels
		],
		dtype=float,
	)


# --- Radius and cylindrical vessel approximations ---

def mean_radius_px(radii: np.ndarray) -> float:
	"""
	Return the mean vessel radius in pixels.
	"""

	radii = np.asarray(radii, dtype=float)

	if radii.size == 0:
		raise ValueError("radii cannot be empty")

	return float(np.mean(radii))


def mean_radius_um(
	radii: np.ndarray,
	scale_um_per_pixel: float = DEFAULT_SCALE_UM_PER_PIXEL,
) -> float:
	"""
	Return the mean vessel radius in micrometers.
	"""

	return mean_radius_px(radii) * scale_um_per_pixel


def cross_sectional_area(radius_um: float) -> float:
	"""
	Compute cross-sectional area from vessel radius.
	"""

	if radius_um < 0:
		raise ValueError("radius_um must be non-negative")

	return float(np.pi * radius_um**2)


def cylindrical_surface_area(radius_um: float, length_um: float) -> float:
	"""
	Compute vessel wall surface area using a cylindrical approximation.
	"""

	if radius_um < 0:
		raise ValueError("radius_um must be non-negative")

	if length_um < 0:
		raise ValueError("length_um must be non-negative")

	return float(2.0 * np.pi * radius_um * length_um)


def cylindrical_volume(radius_um: float, length_um: float) -> float:
	"""
	Compute vessel volume using a cylindrical approximation.
	"""

	return cross_sectional_area(radius_um) * length_um


def estimate_segment_geometry(
	centerline: np.ndarray,
	radii: np.ndarray,
	scale_um_per_pixel: float = DEFAULT_SCALE_UM_PER_PIXEL,
) -> dict[str, float]:
	"""
	Estimate common geometric values for one vessel segment.

	Parameters:
		centerline         -> ordered vessel path in ``(x, y)`` coordinates
		radii              -> local radius values along the centerline
		scale_um_per_pixel -> physical scale used to convert pixels to µm

	Returns:
		dictionary containing length, radius, area, surface area, and volume estimates

	Notes:
		The vessel is approximated as a cylinder using mean radius and centerline
		length. This is not perfect for irregular vessels, but it is good enough for
		flow resistance estimates and first-pass exchange calculations.
	"""

	length_px = compute_centerline_length_px(centerline)
	length_um = length_px * scale_um_per_pixel
	radius_px = mean_radius_px(radii)
	radius_um = radius_px * scale_um_per_pixel

	return {
		"length_px": length_px,
		"length_um": length_um,
		"mean_radius_px": radius_px,
		"mean_radius_um": radius_um,
		"cross_sectional_area_um2": cross_sectional_area(radius_um),
		"surface_area_um2": cylindrical_surface_area(radius_um, length_um),
		"volume_um3": cylindrical_volume(radius_um, length_um),
	}