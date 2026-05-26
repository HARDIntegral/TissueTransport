# domain/vessel/__init__.py

"""Vessel domain models and vascular transport helpers."""

from .network import (
	VesselNetwork,
	VesselNode,
	VesselSegment,
)

# Re-export common vascular constants for convenience.
from .constants import *
from .flow import (
	FlowSolution,
	SegmentFlow,
	solve_network_flow,
	solve_network_flow_mmhg,
)
from .transport import (
	BoundarySourceMaps,
	SegmentConcentration,
	assign_flow_weighted_segment_concentrations,
	assign_uniform_segment_concentrations,
	build_boundary_concentration_maps,
	build_boundary_source_maps,
	summarize_boundary_sources,
)

__all__ = [
	# models
	"VesselNetwork",
	"VesselNode",
	"VesselSegment",

	# flow
	"FlowSolution",
	"SegmentFlow",
	"solve_network_flow",
	"solve_network_flow_mmhg",

	# transport
	"BoundarySourceMaps",
	"SegmentConcentration",
	"assign_uniform_segment_concentrations",
	"assign_flow_weighted_segment_concentrations",
	"build_boundary_source_maps",
	"build_boundary_concentration_maps",
	"summarize_boundary_sources",
]