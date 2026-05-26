"""
Coupling helpers between vessel networks and tissue grids.

This module maps 1D vessel geometry onto the 2D tissue simulation grid. It does
not solve flow, diffusion, or blood chemistry directly. It only builds masks and
lookup tables that later solvers can reuse.

Main idea:
	VesselNetwork stores graph/centerline geometry.
	The tissue solver works on pixels.
	This file connects those two worlds.

Coordinate convention:
	Grid pixels use ``(row, col)`` or ``(y, x)``.
	Vessel centerlines store ``(x, y)`` coordinates.

Notes:
	The expensive geometry maps should be built once and cached. The simulation
	loop should reuse these masks/maps unless the vessel geometry changes.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .network import VesselNetwork, VesselSegment

GridShape = tuple[int, int]
Pixel = tuple[int, int]
RadiusPixel = tuple[int, int, float]


# --- Basic vessel masks ---

def build_centerline_mask(network: VesselNetwork, shape: GridShape) -> np.ndarray:
	"""
	Build a boolean mask containing vessel centerline pixels.

	Parameters:
		network -> vessel graph containing segment centerlines
		shape   -> tissue grid shape as ``(height, width)``

	Returns:
		boolean mask where centerline pixels are True

	Notes:
		This is mostly a debug/visualization mask. Flow uses the segment centerlines
		directly, not this rasterized mask.
	"""

	mask = np.zeros(shape, dtype=bool)

	for segment in network.iter_segments():
		for y, x in _centerline_pixels(segment, shape):
			mask[y, x] = True

	return mask


def build_vessel_mask(
	network: VesselNetwork,
	shape: GridShape,
	radius_scale: float = 1.0,
) -> np.ndarray:
	"""
	Build an approximate filled vessel mask from centerlines and radii.

	Parameters:
		network      -> vessel graph containing centerlines and radii
		shape        -> tissue grid shape as ``(height, width)``
		radius_scale -> multiplier applied to each local vessel radius

	Returns:
		boolean mask where vessel interior pixels are True

	Notes:
		This draws disks along each segment centerline. It is useful when the network
		itself is the source of geometry. For final VeSeg reconstructions, prefer the
		reconstructed vessel mask from VeSeg instead of rebuilding it here.
	"""

	mask = np.zeros(shape, dtype=bool)

	for segment in network.iter_segments():
		for y, x, radius_px in _segment_radius_pixels(segment, shape):
			_draw_disk(mask, y, x, radius_px * radius_scale)

	return mask


def build_exchange_mask(
	network: VesselNetwork,
	shape: GridShape,
	radius_padding: float = 1.0,
) -> np.ndarray:
	"""
	Build a shell of tissue pixels around the vessel interior.

	Parameters:
		network        -> vessel graph containing centerlines and radii
		shape          -> tissue grid shape as ``(height, width)``
		radius_padding -> extra pixel distance around each vessel radius

	Returns:
		boolean mask where near-vessel exchange pixels are True

	Notes:
		This is a rough local exchange region. For the current reconstructed-vessel
		coupling path, boundary pixel maps are more precise than this shell mask.
	"""

	vessel_mask = build_vessel_mask(network, shape)
	exchange_mask = np.zeros(shape, dtype=bool)

	for segment in network.iter_segments():
		for y, x, radius_px in _segment_radius_pixels(segment, shape):
			_draw_disk(exchange_mask, y, x, radius_px + radius_padding)

	return exchange_mask & ~vessel_mask


def build_source_mask(
	network: VesselNetwork,
	shape: GridShape,
	use_exchange_region: bool = False,
) -> np.ndarray:
	"""
	Build a simple source/sink mask from the vessel geometry.

	Parameters:
		network             -> vessel graph to rasterize
		shape               -> tissue grid shape as ``(height, width)``
		use_exchange_region -> if True, use near-vessel shell instead of vessel interior

	Returns:
		boolean source/sink mask
	"""

	if use_exchange_region:
		return build_exchange_mask(network, shape)

	return build_vessel_mask(network, shape)


# --- Segment-owned pixel maps ---

def build_segment_pixel_map(
	network: VesselNetwork,
	shape: GridShape,
	radius_scale: float = 1.0,
) -> dict[int, list[Pixel]]:
	"""
	Map each vessel segment to the grid pixels occupied by that segment.

	Parameters:
		network      -> vessel graph containing centerlines and radii
		shape        -> tissue grid shape as ``(height, width)``
		radius_scale -> multiplier applied to each local vessel radius

	Returns:
		dict[segment_id, list[(y, x)]]

	Notes:
		This map is useful for debugging segment ownership. For actual tissue exchange,
		boundary ownership is usually better than filled-interior ownership.
	"""

	segment_pixels: dict[int, set[Pixel]] = defaultdict(set)

	for segment in network.iter_segments():
		for y, x, radius_px in _segment_radius_pixels(segment, shape):
			segment_pixels[segment.id].update(_disk_pixels(shape, y, x, radius_px * radius_scale))

	return _sorted_pixel_map(segment_pixels)


def build_exchange_pixel_map(
	network: VesselNetwork,
	shape: GridShape,
	radius_padding: float = 1.0,
) -> dict[int, list[Pixel]]:
	"""
	Map each vessel segment to nearby tissue exchange pixels.

	Parameters:
		network        -> vessel graph containing centerlines and radii
		shape          -> tissue grid shape as ``(height, width)``
		radius_padding -> extra pixel distance around each local vessel radius

	Returns:
		dict[segment_id, list[(y, x)]]
	"""

	exchange_pixels: dict[int, set[Pixel]] = defaultdict(set)

	for segment in network.iter_segments():
		for y, x, radius_px in _segment_radius_pixels(segment, shape):
			outer_pixels = set(_disk_pixels(shape, y, x, radius_px + radius_padding))
			inner_pixels = set(_disk_pixels(shape, y, x, radius_px))
			exchange_pixels[segment.id].update(outer_pixels - inner_pixels)

	return _sorted_pixel_map(exchange_pixels)


# --- Reconstructed vessel boundary maps ---

def build_vessel_boundary_mask(vessel_mask: np.ndarray) -> np.ndarray:
	"""
	Build boundary pixels from a filled vessel mask.

	Parameters:
		vessel_mask -> filled 2D vessel geometry mask

	Returns:
		boolean mask where vessel boundary pixels are True

	Notes:
		This keeps vessel pixels that touch background and removes interior pixels.
		For VeSeg reconstruction, this is the wall-like geometry used for exchange
		coupling.
	"""

	vessel_mask = _validate_mask(vessel_mask, "vessel_mask")
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


def build_boundary_pixel_map(
	network: VesselNetwork,
	boundary_mask: np.ndarray,
) -> dict[int, list[Pixel]]:
	"""
	Map reconstructed vessel boundary pixels to nearest vessel segments.

	Parameters:
		network       -> vessel graph that owns the flow segments
		boundary_mask -> boundary pixels from the final vessel geometry

	Returns:
		dict[segment_id, list[(y, x)]]

	Notes:
		This is the important bridge from 2D tissue pixels back to 1D vessel flow.
		Each boundary pixel gets assigned to the nearest segment centerline pixel.

		The nearest-segment lookup is vectorized across all boundary pixels so we do
		not loop over every boundary pixel and every centerline pixel manually.
	"""

	boundary_mask = _validate_mask(boundary_mask, "boundary_mask")
	boundary_pixels = _mask_pixels(boundary_mask)

	if not boundary_pixels:
		return {}

	centerline_pixels, centerline_segment_ids = _flatten_segment_centerline_pixels(
		network=network,
		shape=boundary_mask.shape,
	)

	if centerline_pixels.size == 0:
		return {}

	owner_ids = _nearest_segment_ids_for_pixels(
		pixels=np.asarray(boundary_pixels, dtype=float),
		centerline_pixels=centerline_pixels,
		centerline_segment_ids=centerline_segment_ids,
	)

	boundary_pixel_map: dict[int, list[Pixel]] = defaultdict(list)

	for pixel, segment_id in zip(boundary_pixels, owner_ids):
		boundary_pixel_map[int(segment_id)].append(pixel)

	return dict(boundary_pixel_map)


def build_reconstructed_boundary_pixel_map(
	network: VesselNetwork,
	reconstructed_vessel_mask: np.ndarray,
) -> dict[int, list[Pixel]]:
	"""
	Build segment-owned boundary pixels from reconstructed vessel geometry.

	Parameters:
		network                   -> vessel graph that owns flow segments
		reconstructed_vessel_mask -> filled vessel geometry from VeSeg reconstruction

	Returns:
		dict[segment_id, list[(y, x)]]

	Notes:
		Use this for the current realistic coupling path. The reconstructed mask gives
		the actual grid vessel shape, and the network centerlines decide which segment
		owns each boundary pixel.
	"""

	boundary_mask = build_vessel_boundary_mask(reconstructed_vessel_mask)
	return build_boundary_pixel_map(network, boundary_mask)


# --- Segment centerline lookup helpers ---

def _flatten_segment_centerline_pixels(
	network: VesselNetwork,
	shape: GridShape,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Flatten all segment centerline pixels into arrays for vectorized lookup.
	"""

	all_pixels: list[Pixel] = []
	all_segment_ids: list[int] = []

	for segment in network.iter_segments():
		pixels = _centerline_pixels(segment, shape)

		if not pixels:
			continue

		all_pixels.extend(pixels)
		all_segment_ids.extend([segment.id] * len(pixels))

	return (
		np.asarray(all_pixels, dtype=float),
		np.asarray(all_segment_ids, dtype=int),
	)


def _nearest_segment_ids_for_pixels(
	pixels: np.ndarray,
	centerline_pixels: np.ndarray,
	centerline_segment_ids: np.ndarray,
	chunk_size: int = 512,
) -> np.ndarray:
	"""
	Find nearest segment ids for many pixels using chunked vectorized distance.

	Notes:
		Chunking keeps memory reasonable while still avoiding slow nested Python loops.
	"""

	nearest_ids = np.empty(len(pixels), dtype=int)

	for start in range(0, len(pixels), chunk_size):
		end = min(start + chunk_size, len(pixels))
		pixel_chunk = pixels[start:end]
		deltas = pixel_chunk[:, None, :] - centerline_pixels[None, :, :]
		distances = np.sum(deltas * deltas, axis=2)
		nearest_indices = np.argmin(distances, axis=1)
		nearest_ids[start:end] = centerline_segment_ids[nearest_indices]

	return nearest_ids


# --- Pixel geometry helpers ---

def _centerline_pixels(segment: VesselSegment, shape: GridShape) -> list[Pixel]:
	"""
	Return rounded centerline pixels for a segment.
	"""

	height, width = shape
	rounded = np.rint(segment.centerline).astype(int)
	pixels: list[Pixel] = []

	for x, y in rounded:
		if 0 <= y < height and 0 <= x < width:
			pixels.append((int(y), int(x)))

	return pixels


def _segment_radius_pixels(segment: VesselSegment, shape: GridShape) -> list[RadiusPixel]:
	"""
	Return centerline pixels paired with local radius values.
	"""

	height, width = shape
	centerline = np.asarray(segment.centerline, dtype=float)
	radii = _radii_for_centerline(segment)
	points: list[RadiusPixel] = []

	for (x_value, y_value), radius_px in zip(centerline, radii):
		x = int(round(x_value))
		y = int(round(y_value))

		if 0 <= y < height and 0 <= x < width:
			points.append((y, x, float(radius_px)))

	return points


def _radii_for_centerline(segment: VesselSegment) -> np.ndarray:
	"""
	Return one radius value for each centerline point.
	"""

	if len(segment.radii) == len(segment.centerline):
		return np.asarray(segment.radii, dtype=float)

	if len(segment.radii) == 1:
		return np.full(len(segment.centerline), float(segment.radii[0]), dtype=float)

	# If the radii are mismatched, interpolate them along the segment path.
	old_positions = np.linspace(0.0, 1.0, len(segment.radii))
	new_positions = np.linspace(0.0, 1.0, len(segment.centerline))
	return np.interp(new_positions, old_positions, np.asarray(segment.radii, dtype=float))


def _draw_disk(mask: np.ndarray, y_center: int, x_center: int, radius: float) -> None:
	"""
	Draw a filled disk into a boolean mask.
	"""

	for y, x in _disk_pixels(mask.shape, y_center, x_center, radius):
		mask[y, x] = True


def _disk_pixels(shape: GridShape, y_center: int, x_center: int, radius: float) -> list[Pixel]:
	"""
	Return all pixels inside a disk clipped to the grid shape.
	"""

	height, width = shape
	radius = max(float(radius), 0.0)
	radius_int = int(np.ceil(radius))
	radius_squared = radius**2

	y_min = max(0, y_center - radius_int)
	y_max = min(height - 1, y_center + radius_int)
	x_min = max(0, x_center - radius_int)
	x_max = min(width - 1, x_center + radius_int)

	pixels: list[Pixel] = []

	for y in range(y_min, y_max + 1):
		for x in range(x_min, x_max + 1):
			if (y - y_center) ** 2 + (x - x_center) ** 2 <= radius_squared:
				pixels.append((y, x))

	return pixels


def _mask_pixels(mask: np.ndarray) -> list[Pixel]:
	"""
	Return all True pixels from a mask in ``(y, x)`` order.
	"""

	return [
		(int(y), int(x))
		for y, x in np.argwhere(mask)
	]


def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
	"""
	Validate and convert a 2D boolean mask.
	"""

	mask = np.asarray(mask, dtype=bool)

	if mask.ndim != 2:
		raise ValueError(f"{name} must be a 2D array")

	return mask


def _sorted_pixel_map(pixel_map: dict[int, set[Pixel]]) -> dict[int, list[Pixel]]:
	"""
	Convert a segment-to-pixel-set map into a stable sorted list map.
	"""

	return {
		segment_id: sorted(pixels)
		for segment_id, pixels in pixel_map.items()
	}
