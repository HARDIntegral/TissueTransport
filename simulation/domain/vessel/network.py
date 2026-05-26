"""
Core vessel network data structures.

This module stores the physical vessel graph used by the flow and transport
solvers. It does not solve blood flow or tissue diffusion directly. It only
stores the geometry that those solvers need.

Main idea:
	VeSeg / graph extraction gives us nodes, edges, centerlines, and radii.
	This file turns that raw geometry into a clean VesselNetwork object.

Coordinate convention:
	node.x / centerline[:, 0] -> image column / x coordinate
	node.y / centerline[:, 1] -> image row / y coordinate

Notes:
	Most of this geometry should be built once and cached. The tissue simulation
	should not rebuild the vessel graph every timestep unless the vessel structure
	actually changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .constants import DEFAULT_SCALE_UM_PER_PIXEL
from .geometry import sample_radii_from_map


@dataclass(slots=True)
class VesselNode:
	"""
	Represents one point in the vessel graph.

	Parameters:
		id          -> unique integer identifier for this node
		x           -> x-position in image/pixel coordinates
		y           -> y-position in image/pixel coordinates
		pressure    -> optional pressure value used later by the flow solver
		is_inlet    -> marks this node as a known inlet
		is_outlet   -> marks this node as a known outlet

	Notes:
		A node can be a branch point, endpoint, or connection point. The node itself
		does not store flow. Flow belongs to the segments connected to nodes.
	"""

	id: int
	x: float
	y: float
	pressure: float | None = None
	is_inlet: bool = False
	is_outlet: bool = False

	@property
	def position(self) -> tuple[float, float]:
		"""Return the node position as an ``(x, y)`` tuple."""

		return (self.x, self.y)


@dataclass(slots=True)
class VesselSegment:
	"""
	Represents one vessel segment between two nodes.

	Parameters:
		id                         -> unique integer identifier for this segment
		start_node                 -> id of the upstream/first endpoint node
		end_node                   -> id of the downstream/second endpoint node
		centerline                 -> ordered pixel path through the vessel segment
		radii                      -> local vessel radius values along the centerline
		scale_um_per_pixel          -> physical scale used to convert pixels to µm
		flow_rate                  -> optional solved flow value
		oxygen_concentration        -> optional blood-side oxygen concentration
		carbon_dioxide_concentration -> optional blood-side carbon dioxide concentration
		metadata                   -> extra source data from VeSeg or graph extraction

	Attributes:
		length_px                  -> centerline length in pixels
		length_um                  -> centerline length in micrometers
		mean_radius_px             -> average vessel radius in pixels
		mean_radius_um             -> average vessel radius in micrometers
		cross_sectional_area_um2   -> approximate circular cross-sectional area
		surface_area_um2           -> approximate cylindrical wall surface area
		volume_um3                 -> approximate cylindrical vessel volume

	Notes:
		The centerline is the 1D flow path. The radius tells us how thick the vessel
		is around that path. Together they let the flow solver estimate resistance
		and let the coupling code reconstruct vessel/tissue exchange boundaries.
	"""

	id: int
	start_node: int
	end_node: int
	centerline: np.ndarray
	radii: np.ndarray
	scale_um_per_pixel: float = DEFAULT_SCALE_UM_PER_PIXEL
	flow_rate: float | None = None
	oxygen_concentration: float | None = None
	carbon_dioxide_concentration: float | None = None
	metadata: dict[str, object] = field(default_factory=dict)

	def __post_init__(self) -> None:
		self.centerline = np.asarray(self.centerline, dtype=float)
		self.radii = np.asarray(self.radii, dtype=float)

		if self.centerline.ndim != 2 or self.centerline.shape[1] != 2:
			raise ValueError("centerline must have shape (n_points, 2)")

		if self.radii.ndim != 1:
			raise ValueError("radii must be a 1D array")

		if len(self.radii) not in {1, len(self.centerline)}:
			raise ValueError("radii must contain either one value or one value per centerline point")

	@property
	def length_px(self) -> float:
		"""Return the segment length in pixels."""

		if "length_px" in self.metadata:
			return float(self.metadata["length_px"])

		if len(self.centerline) < 2:
			return 0.0

		steps = np.diff(self.centerline, axis=0)
		return float(np.linalg.norm(steps, axis=1).sum())

	@property
	def length_um(self) -> float:
		"""Return the segment length in micrometers."""

		return self.length_px * self.scale_um_per_pixel

	@property
	def mean_radius_px(self) -> float:
		"""Return the average segment radius in pixels."""

		return float(np.mean(self.radii))

	@property
	def mean_radius_um(self) -> float:
		"""Return the average segment radius in micrometers."""

		return self.mean_radius_px * self.scale_um_per_pixel

	@property
	def cross_sectional_area_um2(self) -> float:
		"""Return the estimated cross-sectional area using the mean radius."""

		return float(np.pi * self.mean_radius_um**2)

	@property
	def surface_area_um2(self) -> float:
		"""Return the estimated cylindrical wall surface area."""

		return float(2.0 * np.pi * self.mean_radius_um * self.length_um)

	@property
	def volume_um3(self) -> float:
		"""Return the estimated cylindrical vessel volume."""

		return self.cross_sectional_area_um2 * self.length_um


@dataclass(slots=True)
class VesselNetwork:
	"""
	Represents the full extracted vessel graph.

	Parameters:
		nodes              -> dictionary of node id to VesselNode
		segments           -> dictionary of segment id to VesselSegment
		scale_um_per_pixel -> physical scale used to convert pixels to µm

	Attributes:
		nodes              -> all graph nodes in the vascular network
		segments           -> all vessel segments connecting nodes
		scale_um_per_pixel -> default physical pixel size for this network

	Notes:
		This object is the bridge between image-space vessels and physics-space
		transport. The image processing code creates the network once, then flow,
		transport, and coupling modules reuse the same graph.

		In the actual simulation loop, this should mostly be treated as static data.
		Only rebuild it if the vessel mask, vessel radius, or angiogenesis structure
		changes.
	"""

	nodes: dict[int, VesselNode] = field(default_factory=dict)
	segments: dict[int, VesselSegment] = field(default_factory=dict)
	scale_um_per_pixel: float = DEFAULT_SCALE_UM_PER_PIXEL

	@classmethod
	def from_veseg(
		cls,
		nodes: np.ndarray,
		edges: np.ndarray,
		distance_map: np.ndarray | None = None,
		centerlines: dict[int, np.ndarray] | None = None,
		scale_um_per_pixel: float = DEFAULT_SCALE_UM_PER_PIXEL,
		min_length_px: float = 10.0,
		min_radius_px: float = 0.5,
	) -> "VesselNetwork":
		"""
		Build a VesselNetwork from VeSeg geometry arrays.

		Parameters:
			nodes              -> VeSeg node table, usually [id, row, col, kind]
			edges              -> VeSeg edge table with ids, endpoints, length, and radius stats
			distance_map       -> radius/distance map from the reconstructed vessel mask
			centerlines        -> optional traced centerline path for each segment id
			scale_um_per_pixel -> physical scale used to convert pixels to µm
			min_length_px      -> filter cutoff for tiny vessel fragments
			min_radius_px      -> filter cutoff for very thin/noisy vessel fragments

		Notes:
			If a traced centerline exists, the segment follows the real skeleton path.
			If it does not exist, the segment falls back to a straight line between its
			two endpoint nodes while keeping VeSeg's measured length/radius metadata.

			The filtering happens before segments are added. This keeps obvious image
			noise from entering the flow and transport code.
		"""

		nodes = np.asarray(nodes)
		edges = np.asarray(edges)
		centerlines = centerlines or {}

		if nodes.ndim != 2 or nodes.shape[1] < 4:
			raise ValueError("nodes must have shape (n_nodes, 4) or wider")

		if edges.ndim != 2 or edges.shape[1] < 8:
			raise ValueError("edges must have shape (n_edges, 8) or wider")

		network = cls(scale_um_per_pixel=scale_um_per_pixel)

		for node_row in nodes:
			node_id = int(node_row[0])
			y = float(node_row[1])
			x = float(node_row[2])

			network.add_node(
				VesselNode(
					id=node_id,
					x=x,
					y=y,
				)
			)

		for edge_row in edges:
			segment_id = int(edge_row[0])
			start_node = int(edge_row[1])
			end_node = int(edge_row[2])
			length_px = float(edge_row[3])
			mean_radius_px = float(edge_row[4])
			edge_min_radius_px = float(edge_row[5])
			max_radius_px = float(edge_row[6])
			normalized_radius = float(edge_row[7])

			if length_px < min_length_px:
				continue

			if mean_radius_px < min_radius_px:
				continue

			start = network.nodes[start_node]
			end = network.nodes[end_node]

			if segment_id in centerlines:
				centerline = np.asarray(centerlines[segment_id], dtype=float)
			else:
				centerline = cls._interpolate_centerline(
					start_node=start,
					end_node=end,
					length_px=length_px,
				)

			network.add_segment(
				VesselSegment(
					id=segment_id,
					start_node=start_node,
					end_node=end_node,
					centerline=centerline,
					radii=np.asarray([mean_radius_px], dtype=float),
					scale_um_per_pixel=scale_um_per_pixel,
					metadata={
						"length_px": length_px,
						"mean_radius_px": mean_radius_px,
						"min_radius_px": edge_min_radius_px,
						"max_radius_px": max_radius_px,
						"normalized_radius": normalized_radius,
						"source": "veseg",
					},
				)
			)

		return network

	@classmethod
	def from_graph(
		cls,
		graph,
		radius_map: np.ndarray,
		scale_um_per_pixel: float = DEFAULT_SCALE_UM_PER_PIXEL,
	) -> "VesselNetwork":
		"""
		Build a vessel network from a graph and radius map.

		The graph is expected to store node positions and edge centerlines from a
		skeleton graph extraction step. This method does not assign inlets, outlets,
		pressures, or flow directions.
		"""

		network = cls(scale_um_per_pixel=scale_um_per_pixel)

		for node_id, node_data in graph.nodes(data=True):
			x, y = cls._extract_node_position(node_data)

			network.add_node(
				VesselNode(
					id=int(node_id),
					x=x,
					y=y,
				)
			)

		for segment_id, (start_node, end_node, edge_data) in enumerate(graph.edges(data=True)):
			centerline = cls._extract_edge_centerline(
				edge_data=edge_data,
				start_node=network.nodes[int(start_node)],
				end_node=network.nodes[int(end_node)],
			)
			radii = sample_radii_from_map(centerline, radius_map)

			network.add_segment(
				VesselSegment(
					id=segment_id,
					start_node=int(start_node),
					end_node=int(end_node),
					centerline=centerline,
					radii=radii,
					scale_um_per_pixel=scale_um_per_pixel,
					metadata=dict(edge_data),
				)
			)

		return network

	@staticmethod
	def _interpolate_centerline(
		start_node: VesselNode,
		end_node: VesselNode,
		length_px: float,
	) -> np.ndarray:
		"""
		Build a straight fallback centerline between two nodes.

		VeSeg's compact edge table stores the measured edge length but not the full
		path coordinates, so this is only a geometric fallback for masks and quick
		visualization.
		"""

		point_count = max(int(round(length_px)) + 1, 2)

		x_values = np.linspace(start_node.x, end_node.x, point_count)
		y_values = np.linspace(start_node.y, end_node.y, point_count)

		return np.column_stack((x_values, y_values))

	@staticmethod
	def _extract_node_position(node_data: dict[str, object]) -> tuple[float, float]:
		"""
		Extract an ``(x, y)`` position from graph node metadata.
		"""

		if "x" in node_data and "y" in node_data:
			return float(node_data["x"]), float(node_data["y"])

		if "pos" in node_data:
			position = node_data["pos"]
			return float(position[0]), float(position[1])

		if "position" in node_data:
			position = node_data["position"]
			return float(position[0]), float(position[1])

		if "coord" in node_data:
			coord = node_data["coord"]
			return float(coord[0]), float(coord[1])

		raise ValueError("node metadata must contain x/y, pos, position, or coord")

	@staticmethod
	def _extract_edge_centerline(
		edge_data: dict[str, object],
		start_node: VesselNode,
		end_node: VesselNode,
	) -> np.ndarray:
		"""
		Extract an ordered centerline from graph edge metadata.
		"""

		for key in ("centerline", "pixels", "points", "path", "coords"):
			if key in edge_data:
				centerline = np.asarray(edge_data[key], dtype=float)

				if centerline.ndim != 2 or centerline.shape[1] != 2:
					raise ValueError(f"edge {key} must have shape (n_points, 2)")

				return centerline

		return np.asarray(
			[
				[start_node.x, start_node.y],
				[end_node.x, end_node.y],
			],
			dtype=float,
		)

	def add_node(self, node: VesselNode) -> None:
		"""Add a node to the network."""

		if node.id in self.nodes:
			raise ValueError(f"node with id {node.id} already exists")

		self.nodes[node.id] = node

	def add_segment(self, segment: VesselSegment) -> None:
		"""Add a segment to the network."""

		if segment.id in self.segments:
			raise ValueError(f"segment with id {segment.id} already exists")

		if segment.start_node not in self.nodes:
			raise ValueError(f"start node {segment.start_node} does not exist")

		if segment.end_node not in self.nodes:
			raise ValueError(f"end node {segment.end_node} does not exist")

		self.segments[segment.id] = segment

	def connected_segments(self, node_id: int) -> list[VesselSegment]:
		"""Return all segments connected to a node."""

		if node_id not in self.nodes:
			raise ValueError(f"node {node_id} does not exist")

		return [
			segment
			for segment in self.segments.values()
			if segment.start_node == node_id or segment.end_node == node_id
		]

	def inlet_nodes(self) -> list[VesselNode]:
		"""Return all nodes marked as inlets."""

		return [node for node in self.nodes.values() if node.is_inlet]

	def outlet_nodes(self) -> list[VesselNode]:
		"""Return all nodes marked as outlets."""

		return [node for node in self.nodes.values() if node.is_outlet]

	def endpoint_nodes(self) -> list[VesselNode]:
		"""Return nodes connected to only one vessel segment."""

		degree_by_node = {
			node_id: 0
			for node_id in self.nodes
		}

		for segment in self.segments.values():
			degree_by_node[segment.start_node] = degree_by_node.get(segment.start_node, 0) + 1
			degree_by_node[segment.end_node] = degree_by_node.get(segment.end_node, 0) + 1

		return [
			node
			for node in self.nodes.values()
			if degree_by_node.get(node.id, 0) == 1
		]

	def node_neighbor_map(self) -> dict[int, set[int]]:
		"""
		Build a fast lookup of which nodes touch each other.

		Returns:
			dict[node_id, set[neighbor_node_id]]

		Notes:
			This is useful for graph traversal, connected components, and picking simple
			flow boundary nodes without keeping that logic inside test files.
		"""

		neighbors: dict[int, set[int]] = {
			node_id: set()
			for node_id in self.nodes
		}

		for segment in self.segments.values():
			neighbors.setdefault(segment.start_node, set()).add(segment.end_node)
			neighbors.setdefault(segment.end_node, set()).add(segment.start_node)

		return neighbors

	def largest_connected_component(self) -> set[int]:
		"""
		Find the largest connected group of vessel nodes.

		Returns:
			set of node ids belonging to the largest connected component

		Notes:
			Real segmented vessel images often contain disconnected fragments. The flow
			solver needs a connected graph with pressure boundary conditions, so this
			helper finds the main network instead of tiny isolated pieces.
		"""

		if not self.nodes:
			return set()

		neighbors = self.node_neighbor_map()
		unvisited = set(self.nodes)
		largest_component: set[int] = set()

		while unvisited:
			start_node = unvisited.pop()
			component = {start_node}
			stack = [start_node]

			while stack:
				node_id = stack.pop()

				for neighbor_id in neighbors.get(node_id, set()):
					if neighbor_id in unvisited:
						unvisited.remove(neighbor_id)
						component.add(neighbor_id)
						stack.append(neighbor_id)

			if len(component) > len(largest_component):
				largest_component = component

		return largest_component

	def pressure_boundary_nodes_from_largest_component(self) -> tuple[int, int, set[int]]:
		"""
		Pick simple inlet/outlet candidates from the largest connected component.

		Returns:
			inlet_node_id       -> leftmost node in the largest connected component
			outlet_node_id      -> rightmost node in the largest connected component
			largest_component   -> node ids included in the selected component

		Notes:
			This is a smoke-test helper, not a final physiological boundary condition
			model. For now, leftmost/rightmost pressure boundaries are enough to verify
			that the extracted graph can support a pressure-driven flow solve.
		"""

		largest_component = self.largest_connected_component()

		if len(largest_component) < 2:
			raise ValueError("at least two connected nodes are required to solve flow")

		component_nodes = [
			self.nodes[node_id]
			for node_id in largest_component
		]

		inlet_node = min(component_nodes, key=lambda node: node.x)
		outlet_node = max(component_nodes, key=lambda node: node.x)

		return inlet_node.id, outlet_node.id, largest_component

	def total_length_um(self) -> float:
		"""Return the total vessel length in micrometers."""

		return sum(segment.length_um for segment in self.segments.values())

	def total_volume_um3(self) -> float:
		"""Return the total estimated vessel volume in cubic micrometers."""

		return sum(segment.volume_um3 for segment in self.segments.values())

	def iter_segments(self) -> Iterable[VesselSegment]:
		"""Iterate over vessel segments."""

		return self.segments.values()

	def iter_nodes(self) -> Iterable[VesselNode]:
		"""Iterate over vessel nodes."""

		return self.nodes.values()
