

"""
Flow calculations for vessel networks.

This module treats the vessel graph as a 1D hydraulic network. Each segment is
modeled as a cylindrical resistor, node pressures are solved from conservation
of flow, and segment flow rates are computed from pressure differences.
"""

from dataclasses import dataclass

import numpy as np

from .network import VesselNetwork


DEFAULT_BLOOD_VISCOSITY_PA_S = 3.5e-3
PA_PER_MMHG = 133.322
METER_PER_MICROMETER = 1.0e-6
M3_PER_SECOND_TO_UM3_PER_SECOND = 1.0e18


@dataclass(frozen=True)
class SegmentFlow:
	"""
	Flow result for one vessel segment.
	"""

	segment_id: int
	start_node: int
	end_node: int
	start_pressure_pa: float
	end_pressure_pa: float
	resistance_pa_s_per_m3: float
	flow_m3_per_s: float
	flow_um3_per_s: float


@dataclass(frozen=True)
class FlowSolution:
	"""
	Pressure and flow solution for a vessel network.
	"""

	node_pressures_pa: dict[int, float]
	segment_flows: dict[int, SegmentFlow]

	def node_pressure_mmhg(self, node_id: int) -> float:
		"""
		Return a solved node pressure in mmHg.
		"""

		return self.node_pressures_pa[node_id] / PA_PER_MMHG

	def segment_flow_um3_per_s(self, segment_id: int) -> float:
		"""
		Return a solved segment flow in ``µm^3 / s``.
		"""

		return self.segment_flows[segment_id].flow_um3_per_s


def compute_segment_resistance(
	length_um: float,
	radius_um: float,
	viscosity_pa_s: float = DEFAULT_BLOOD_VISCOSITY_PA_S,
) -> float:
	"""
	Compute hydraulic resistance from Hagen-Poiseuille flow.

	Length and radius are provided in micrometers and converted to SI units
	internally. The returned resistance has units ``Pa s / m^3``.
	"""

	if length_um <= 0:
		raise ValueError("length_um must be positive")

	if radius_um <= 0:
		raise ValueError("radius_um must be positive")

	if viscosity_pa_s <= 0:
		raise ValueError("viscosity_pa_s must be positive")

	length_m = length_um * METER_PER_MICROMETER
	radius_m = radius_um * METER_PER_MICROMETER

	return (8.0 * viscosity_pa_s * length_m) / (np.pi * radius_m**4)


def solve_network_flow(
	network: VesselNetwork,
	fixed_pressures_pa: dict[int, float],
	viscosity_pa_s: float = DEFAULT_BLOOD_VISCOSITY_PA_S,
) -> FlowSolution:
	"""
	Solve pressures and segment flows for a vessel network.

	``fixed_pressures_pa`` should contain pressure boundary conditions for at
	least two nodes, usually one inlet and one outlet. Unknown internal node
	pressures are solved using conservation of flow.
	"""

	if len(fixed_pressures_pa) < 2:
		raise ValueError("at least two fixed-pressure nodes are required")

	resistances = compute_segment_resistances(
		network=network,
		viscosity_pa_s=viscosity_pa_s,
	)
	node_segments = _build_node_segment_lookup(network)
	active_nodes = _find_pressure_connected_nodes(
		network=network,
		fixed_nodes=set(fixed_pressures_pa),
		node_segments=node_segments,
	)
	active_fixed_pressures = {
		node_id: pressure
		for node_id, pressure in fixed_pressures_pa.items()
		if node_id in active_nodes
	}

	if len(active_fixed_pressures) < 2:
		raise ValueError("flow solve requires at least two fixed-pressure nodes in the same active network")

	unknown_nodes = [
		node_id
		for node_id in active_nodes
		if node_id not in active_fixed_pressures
	]

	if not unknown_nodes:
		node_pressures = dict(active_fixed_pressures)
		segment_flows = compute_segment_flows(network, node_pressures, resistances)
		return FlowSolution(node_pressures, segment_flows)

	matrix, rhs = _build_pressure_system(
		network=network,
		unknown_nodes=unknown_nodes,
		fixed_pressures_pa=active_fixed_pressures,
		segment_resistances=resistances,
		node_segments=node_segments,
	)

	solved_pressures = np.linalg.solve(matrix, rhs)
	node_pressures = dict(active_fixed_pressures)

	for node_id, pressure in zip(unknown_nodes, solved_pressures):
		node_pressures[node_id] = float(pressure)

	segment_flows = compute_segment_flows(network, node_pressures, resistances)
	return FlowSolution(node_pressures, segment_flows)

def _build_node_segment_lookup(network: VesselNetwork) -> dict[int, list[int]]:
	"""
	Build a node-to-segment adjacency lookup for a vessel network.
	"""

	node_segments: dict[int, list[int]] = {
		node_id: []
		for node_id in network.nodes
	}

	for segment in network.iter_segments():
		node_segments.setdefault(segment.start_node, []).append(segment.id)
		node_segments.setdefault(segment.end_node, []).append(segment.id)

	return node_segments


def _find_pressure_connected_nodes(
	network: VesselNetwork,
	fixed_nodes: set[int],
	node_segments: dict[int, list[int]],
) -> set[int]:
	"""
	Find the connected component anchored by the fixed-pressure nodes.

	Disconnected fragments without pressure boundary conditions make the pressure
	matrix singular, so the flow solve only includes nodes reachable from the
	fixed-pressure component.
	"""

	if not fixed_nodes:
		return set()

	visited: set[int] = set()
	queue = list(fixed_nodes)

	while queue:
		node_id = queue.pop()

		if node_id in visited or node_id not in network.nodes:
			continue

		visited.add(node_id)

		for neighbor_id in _neighbor_nodes(network, node_id, node_segments):
			if neighbor_id not in visited:
				queue.append(neighbor_id)

	return visited


def _neighbor_nodes(
	network: VesselNetwork,
	node_id: int,
	node_segments: dict[int, list[int]],
) -> list[int]:
	"""
	Return graph neighbors for a node.
	"""

	neighbors: list[int] = []

	for segment_id in node_segments.get(node_id, []):
		segment = network.segments[segment_id]
		neighbors.append(_other_node(segment.start_node, segment.end_node, node_id))

	return neighbors


def solve_network_flow_mmhg(
	network: VesselNetwork,
	fixed_pressures_mmhg: dict[int, float],
	viscosity_pa_s: float = DEFAULT_BLOOD_VISCOSITY_PA_S,
) -> FlowSolution:
	"""
	Solve network flow using pressure boundary conditions in mmHg.
	"""

	fixed_pressures_pa = {
		node_id: pressure_mmhg * PA_PER_MMHG
		for node_id, pressure_mmhg in fixed_pressures_mmhg.items()
	}

	return solve_network_flow(
		network=network,
		fixed_pressures_pa=fixed_pressures_pa,
		viscosity_pa_s=viscosity_pa_s,
	)


def compute_segment_resistances(
	network: VesselNetwork,
	viscosity_pa_s: float = DEFAULT_BLOOD_VISCOSITY_PA_S,
) -> dict[int, float]:
	"""
	Compute hydraulic resistance for every segment in a network.
	"""

	resistances: dict[int, float] = {}

	for segment in network.iter_segments():
		resistances[segment.id] = compute_segment_resistance(
			length_um=segment.length_um,
			radius_um=float(np.mean(segment.radii)) * network.scale_um_per_pixel,
			viscosity_pa_s=viscosity_pa_s,
		)

	return resistances


def compute_segment_flows(
	network: VesselNetwork,
	node_pressures_pa: dict[int, float],
	segment_resistances: dict[int, float],
) -> dict[int, SegmentFlow]:
	"""
	Compute oriented segment flows from solved node pressures.

	Positive flow means flow from ``segment.start_node`` to ``segment.end_node``.
	Negative flow means the actual flow direction is reversed.
	"""

	flows: dict[int, SegmentFlow] = {}

	for segment in network.iter_segments():
		if segment.start_node not in node_pressures_pa or segment.end_node not in node_pressures_pa:
			continue

		start_pressure = node_pressures_pa[segment.start_node]
		end_pressure = node_pressures_pa[segment.end_node]
		resistance = segment_resistances[segment.id]
		flow_m3_per_s = (start_pressure - end_pressure) / resistance

		flows[segment.id] = SegmentFlow(
			segment_id=segment.id,
			start_node=segment.start_node,
			end_node=segment.end_node,
			start_pressure_pa=start_pressure,
			end_pressure_pa=end_pressure,
			resistance_pa_s_per_m3=resistance,
			flow_m3_per_s=flow_m3_per_s,
			flow_um3_per_s=flow_m3_per_s * M3_PER_SECOND_TO_UM3_PER_SECOND,
		)

	return flows


def _build_pressure_system(
	network: VesselNetwork,
	unknown_nodes: list[int],
	fixed_pressures_pa: dict[int, float],
	segment_resistances: dict[int, float],
	node_segments: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Build the linear system for unknown node pressures.
	"""

	unknown_index = {
		node_id: index
		for index, node_id in enumerate(unknown_nodes)
	}

	matrix = np.zeros((len(unknown_nodes), len(unknown_nodes)), dtype=float)
	rhs = np.zeros(len(unknown_nodes), dtype=float)

	for node_id in unknown_nodes:
		row = unknown_index[node_id]

		for segment_id in node_segments.get(node_id, []):
			segment = network.segments[segment_id]
			neighbor_id = _other_node(segment.start_node, segment.end_node, node_id)
			conductance = 1.0 / segment_resistances[segment_id]

			matrix[row, row] += conductance

			if neighbor_id in unknown_index:
				matrix[row, unknown_index[neighbor_id]] -= conductance
			else:
				rhs[row] += conductance * fixed_pressures_pa[neighbor_id]

	return matrix, rhs


def _other_node(start_node: int, end_node: int, node_id: int) -> int:
	"""
	Return the opposite node on a segment.
	"""

	if node_id == start_node:
		return end_node

	if node_id == end_node:
		return start_node

	raise ValueError("node_id is not connected to the provided segment nodes")