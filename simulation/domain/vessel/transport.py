"""
Transport coupling between vessel flow and tissue-grid source maps.

This module converts 1D vessel-segment values into 2D tissue-grid source/sink
maps. It does not solve diffusion or update the tissue concentration fields by
itself. It only builds the vessel-side inputs that the tissue timestep can use.

Main idea:
	Flow solver gives values per vessel segment.
	Boundary coupling gives pixels owned by each segment.
	This file turns those segment values into grid maps.

Sign convention:
	O2 source      -> positive value added to tissue oxygen
	CO2 sink       -> negative value removed from tissue carbon dioxide

Notes:
	The boundary pixel map is geometry-only and should be cached.
	The source maps only need to be rebuilt when segment concentrations/flows
	change, not every time the vessel geometry is reused.
"""

from dataclasses import dataclass

import numpy as np

from .flow import FlowSolution
from .network import VesselNetwork

Pixel = tuple[int, int]
GridShape = tuple[int, int]


@dataclass(frozen=True)
class SegmentConcentration:
	"""
	Stores blood-side concentration/exchange values for one vessel segment.

	Parameters:
		segment_id     -> vessel segment id that owns these values
		oxygen         -> oxygen source strength for tissue coupling
		carbon_dioxide -> carbon dioxide sink/source strength for tissue coupling

	Notes:
		For now, these values are source/sink strengths used for coupling, not a full
		blood chemistry state. Later this can be replaced with real advective blood
		concentration updates along the flow graph.
	"""

	segment_id: int
	oxygen: float
	carbon_dioxide: float


@dataclass(frozen=True)
class BoundarySourceMaps:
	"""
	Stores vessel-generated source/sink maps on the tissue grid.

	Parameters:
		oxygen         -> grid map of tissue oxygen source values
		carbon_dioxide -> grid map of tissue carbon dioxide sink/source values

	Notes:
		These maps have the same shape as the tissue grid. The tissue solver can add
		them directly into the timestep update after scaling by timestep/exchange rate.
	"""

	oxygen: np.ndarray
	carbon_dioxide: np.ndarray


# --- Segment concentration assignment ---

def assign_uniform_segment_concentrations(
	network: VesselNetwork,
	oxygen: float = 1.0,
	carbon_dioxide: float = 0.0,
) -> dict[int, SegmentConcentration]:
	"""
	Assign the same coupling value to every vessel segment.

	Parameters:
		network        -> vessel network whose segments receive values
		oxygen         -> oxygen value assigned to each segment
		carbon_dioxide -> carbon dioxide value assigned to each segment

	Returns:
		dict[segment_id, SegmentConcentration]

	Notes:
		This is mostly useful for debugging because every vessel segment should show
		up with the same strength.
	"""

	return {
		segment.id: SegmentConcentration(
			segment_id=segment.id,
			oxygen=oxygen,
			carbon_dioxide=carbon_dioxide,
		)
		for segment in network.iter_segments()
	}


def assign_flow_weighted_segment_concentrations(
	flow_solution: FlowSolution,
	inlet_oxygen: float = 1.0,
	inlet_carbon_dioxide: float = 0.0,
	minimum_flow_fraction: float = 0.15,
) -> dict[int, SegmentConcentration]:
	"""
	Assign temporary segment values from relative flow magnitude.

	Parameters:
		flow_solution          -> solved vessel pressures and segment flows
		inlet_oxygen           -> max oxygen source value at high-flow segments
		inlet_carbon_dioxide   -> baseline carbon dioxide value
		minimum_flow_fraction  -> lowest fraction kept for low-flow segments

	Returns:
		dict[segment_id, SegmentConcentration]

	Notes:
		This is still placeholder chemistry. It makes high-flow segments stronger O2
		sources and keeps low-flow segments visible instead of dropping them to zero.

		CO2 is currently treated as a mirrored sink term, so high-O2 regions also act
		as stronger CO2 removal regions. Later this should be replaced with real
		blood/tissue exchange equations.
	"""

	if not flow_solution.segment_flows:
		return {}

	max_flow = _max_abs_flow(flow_solution)

	if max_flow <= 0.0:
		return _assign_minimum_flow_concentrations(
			flow_solution=flow_solution,
			inlet_oxygen=inlet_oxygen,
			inlet_carbon_dioxide=inlet_carbon_dioxide,
			minimum_flow_fraction=minimum_flow_fraction,
		)

	concentrations: dict[int, SegmentConcentration] = {}

	for segment_id, flow in flow_solution.segment_flows.items():
		flow_fraction = abs(flow.flow_um3_per_s) / max_flow
		retained_fraction = minimum_flow_fraction + (1.0 - minimum_flow_fraction) * flow_fraction
		concentrations[segment_id] = _make_placeholder_concentration(
			segment_id=segment_id,
			retained_fraction=retained_fraction,
			inlet_oxygen=inlet_oxygen,
			inlet_carbon_dioxide=inlet_carbon_dioxide,
		)

	return concentrations


def _max_abs_flow(flow_solution: FlowSolution) -> float:
	"""
	Return the largest absolute segment flow.
	"""

	flow_values = np.asarray([
		abs(flow.flow_um3_per_s)
		for flow in flow_solution.segment_flows.values()
	], dtype=float)

	return float(np.max(flow_values)) if flow_values.size else 0.0


def _assign_minimum_flow_concentrations(
	flow_solution: FlowSolution,
	inlet_oxygen: float,
	inlet_carbon_dioxide: float,
	minimum_flow_fraction: float,
) -> dict[int, SegmentConcentration]:
	"""
	Assign fallback values when all solved segment flows are zero.
	"""

	return {
		segment_id: _make_placeholder_concentration(
			segment_id=segment_id,
			retained_fraction=minimum_flow_fraction,
			inlet_oxygen=inlet_oxygen,
			inlet_carbon_dioxide=inlet_carbon_dioxide,
		)
		for segment_id in flow_solution.segment_flows
	}


def _make_placeholder_concentration(
	segment_id: int,
	retained_fraction: float,
	inlet_oxygen: float,
	inlet_carbon_dioxide: float,
) -> SegmentConcentration:
	"""
	Create the temporary O2 source / CO2 sink pair for one segment.
	"""

	return SegmentConcentration(
		segment_id=segment_id,
		oxygen=inlet_oxygen * retained_fraction,
		carbon_dioxide=inlet_carbon_dioxide - retained_fraction,
	)


# --- Boundary source-map construction ---

def build_boundary_source_maps(
	shape: GridShape,
	boundary_pixel_map: dict[int, list[Pixel]],
	segment_concentrations: dict[int, SegmentConcentration],
	oxygen_scale: float = 1.0,
	carbon_dioxide_scale: float = 1.0,
	remap_missing_segments: bool = True,
	normalize_by_segment_boundary: bool = False,
) -> BoundarySourceMaps:
	"""
	Build O2 source and CO2 sink/source maps on the tissue grid.

	Parameters:
		shape                         -> tissue grid shape as ``(height, width)``
		boundary_pixel_map            -> segment-owned vessel boundary pixels
		segment_concentrations        -> segment values to write onto the grid
		oxygen_scale                  -> multiplier for oxygen map values
		carbon_dioxide_scale          -> multiplier for carbon dioxide map values
		remap_missing_segments        -> use nearest solved segment for unsolved owners
		normalize_by_segment_boundary -> divide each segment value over its pixels

	Returns:
		BoundarySourceMaps with oxygen and carbon dioxide arrays

	Notes:
		By default, every boundary pixel gets the full local segment source strength.
		Set ``normalize_by_segment_boundary=True`` if the value should be split over
		the whole boundary of a segment.
	"""

	_validate_grid_shape(shape)

	oxygen_source = np.zeros(shape, dtype=float)
	carbon_dioxide_source = np.zeros(shape, dtype=float)
	segment_lookup = _build_segment_concentration_lookup(
		boundary_pixel_map=boundary_pixel_map,
		segment_concentrations=segment_concentrations,
		remap_missing_segments=remap_missing_segments,
	)

	for segment_id, pixels in boundary_pixel_map.items():
		concentration = segment_lookup.get(segment_id)

		if concentration is None or not pixels:
			continue

		pixel_weight = _pixel_weight(
			pixels=pixels,
			normalize_by_segment_boundary=normalize_by_segment_boundary,
		)
		valid_pixels = _valid_pixels(pixels, shape)

		if valid_pixels.size == 0:
			continue

		y_coords = valid_pixels[:, 0]
		x_coords = valid_pixels[:, 1]
		oxygen_source[y_coords, x_coords] += concentration.oxygen * oxygen_scale * pixel_weight
		carbon_dioxide_source[y_coords, x_coords] += concentration.carbon_dioxide * carbon_dioxide_scale * pixel_weight

	return BoundarySourceMaps(
		oxygen=oxygen_source,
		carbon_dioxide=carbon_dioxide_source,
	)


def build_boundary_concentration_maps(
	shape: GridShape,
	boundary_pixel_map: dict[int, list[Pixel]],
	segment_concentrations: dict[int, SegmentConcentration],
	remap_missing_segments: bool = True,
) -> BoundarySourceMaps:
	"""
	Build direct boundary concentration maps for visualization/debugging.

	Parameters:
		shape                  -> tissue grid shape as ``(height, width)``
		boundary_pixel_map     -> segment-owned vessel boundary pixels
		segment_concentrations -> segment values to write onto the grid
		remap_missing_segments -> use nearest solved segment for unsolved owners

	Returns:
		BoundarySourceMaps where boundary pixels directly equal segment values

	Notes:
		Unlike source maps, this does not normalize or scale values. It is mainly for
		checking that segment values are landing on the expected vessel boundary.
	"""

	_validate_grid_shape(shape)

	oxygen = np.zeros(shape, dtype=float)
	carbon_dioxide = np.zeros(shape, dtype=float)
	segment_lookup = _build_segment_concentration_lookup(
		boundary_pixel_map=boundary_pixel_map,
		segment_concentrations=segment_concentrations,
		remap_missing_segments=remap_missing_segments,
	)

	for segment_id, pixels in boundary_pixel_map.items():
		concentration = segment_lookup.get(segment_id)

		if concentration is None:
			continue

		valid_pixels = _valid_pixels(pixels, shape)

		if valid_pixels.size == 0:
			continue

		y_coords = valid_pixels[:, 0]
		x_coords = valid_pixels[:, 1]
		oxygen[y_coords, x_coords] = concentration.oxygen
		carbon_dioxide[y_coords, x_coords] = concentration.carbon_dioxide

	return BoundarySourceMaps(
		oxygen=oxygen,
		carbon_dioxide=carbon_dioxide,
	)


def summarize_boundary_sources(source_maps: BoundarySourceMaps) -> dict[str, float]:
	"""
	Summarize generated vessel source/sink maps.

	Parameters:
		source_maps -> generated oxygen and carbon dioxide maps

	Returns:
		dictionary of total and max source/sink values
	"""

	return {
		"oxygen_total": float(np.sum(source_maps.oxygen)),
		"oxygen_max": _safe_max(source_maps.oxygen),
		"carbon_dioxide_total": float(np.sum(source_maps.carbon_dioxide)),
		"carbon_dioxide_max": _safe_max(source_maps.carbon_dioxide),
	}


# --- Missing segment remapping ---

def _build_segment_concentration_lookup(
	boundary_pixel_map: dict[int, list[Pixel]],
	segment_concentrations: dict[int, SegmentConcentration],
	remap_missing_segments: bool,
) -> dict[int, SegmentConcentration]:
	"""
	Build a concentration lookup for every boundary-owning segment.

	Parameters:
		boundary_pixel_map     -> segment-owned boundary pixels
		segment_concentrations -> known segment concentration values
		remap_missing_segments -> if True, fill missing owners from nearest solved owner

	Returns:
		dict[segment_id, SegmentConcentration]

	Notes:
		Missing segments happen when a boundary segment exists but flow was only solved
		for part of the network. For visualization and first-pass coupling, the nearest
		solved segment is a reasonable fallback.
	"""

	lookup = dict(segment_concentrations)

	if not remap_missing_segments or not segment_concentrations:
		return lookup

	solved_segment_ids, solved_centroids = _solved_segment_centroids(
		boundary_pixel_map=boundary_pixel_map,
		segment_concentrations=segment_concentrations,
	)

	if solved_centroids.size == 0:
		return lookup

	for segment_id, pixels in boundary_pixel_map.items():
		if segment_id in lookup or not pixels:
			continue

		nearest_segment_id = _nearest_centroid_segment_id(
			centroid=_segment_boundary_centroid(pixels),
			solved_segment_ids=solved_segment_ids,
			solved_centroids=solved_centroids,
		)
		lookup[segment_id] = segment_concentrations[nearest_segment_id]

	return lookup


def _solved_segment_centroids(
	boundary_pixel_map: dict[int, list[Pixel]],
	segment_concentrations: dict[int, SegmentConcentration],
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Return ids and centroids for solved segments that own boundary pixels.
	"""

	segment_ids: list[int] = []
	centroids: list[np.ndarray] = []

	for segment_id in segment_concentrations:
		pixels = boundary_pixel_map.get(segment_id)

		if not pixels:
			continue

		segment_ids.append(segment_id)
		centroids.append(_segment_boundary_centroid(pixels))

	if not centroids:
		return np.asarray([], dtype=int), np.empty((0, 2), dtype=float)

	return np.asarray(segment_ids, dtype=int), np.vstack(centroids)


def _segment_boundary_centroid(pixels: list[Pixel]) -> np.ndarray:
	"""
	Return the centroid of a segment's boundary pixels as ``(y, x)``.
	"""

	return np.mean(np.asarray(pixels, dtype=float), axis=0)


def _nearest_centroid_segment_id(
	centroid: np.ndarray,
	solved_segment_ids: np.ndarray,
	solved_centroids: np.ndarray,
) -> int:
	"""
	Find the solved segment with the nearest boundary centroid.
	"""

	distances = np.sum((solved_centroids - centroid) ** 2, axis=1)
	nearest_index = int(np.argmin(distances))
	return int(solved_segment_ids[nearest_index])


# --- Grid helpers ---

def _valid_pixels(pixels: list[Pixel], shape: GridShape) -> np.ndarray:
	"""
	Return in-bounds pixels as a ``(n_pixels, 2)`` integer array.
	"""

	if not pixels:
		return np.empty((0, 2), dtype=int)

	pixel_array = np.asarray(pixels, dtype=int)
	height, width = shape
	in_bounds = (
		(pixel_array[:, 0] >= 0)
		& (pixel_array[:, 0] < height)
		& (pixel_array[:, 1] >= 0)
		& (pixel_array[:, 1] < width)
	)

	return pixel_array[in_bounds]


def _pixel_weight(pixels: list[Pixel], normalize_by_segment_boundary: bool) -> float:
	"""
	Return the weight applied to each boundary pixel for one segment.
	"""

	if normalize_by_segment_boundary:
		return 1.0 / len(pixels)

	return 1.0


def _safe_max(values: np.ndarray) -> float:
	"""
	Return max value for an array, with a safe empty-array fallback.
	"""

	return float(np.max(values)) if values.size else 0.0


def _validate_grid_shape(shape: GridShape) -> None:
	"""
	Validate a 2D grid shape.
	"""

	if len(shape) != 2:
		raise ValueError("shape must have exactly two dimensions")

	if shape[0] <= 0 or shape[1] <= 0:
		raise ValueError("shape dimensions must be positive")