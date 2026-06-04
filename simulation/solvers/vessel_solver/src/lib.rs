// Vessel-side transport solver.
//
// This crate owns vessel graph storage, hydraulic flow, directed topology,
// blood species transport, vessel-tissue exchange, and boundary/source map
// generation.
//
// Tissue diffusion and metabolism live outside this crate.

pub mod exchange;
pub mod flow;
pub mod maps;
pub mod network;
pub mod transport;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

pub use exchange::{compute_exchange, compute_exchange_set};
pub use flow::{poiseuille_resistance, solve_flow, solve_pressure_flow};
pub use maps::{build_boundary_source_maps, empty_boundary_source_maps, rasterize_segments};
pub use network::{
    build_topology_from_perfusion, classify_perfusion_paths, components_with_min_segments,
    connected_components, largest_component, source_nodes_from_pressure_boundaries,
    NetworkComponent, PerfusionCollision, PerfusionPath, PerfusionSegment, VesselNetwork,
};
pub use transport::{
    default_carbon_dioxide_parameters, default_oxygen_parameters, default_species_parameters,
    propagate_species, propagate_species_set,
};
pub use types::{
    BoundarySourceMaps, DirectedTopology, DownstreamConnection, ExchangeResult, FlowSolution,
    NodeKind, NodePressure, PressureBoundary, SegmentExchange, SegmentFlow, SegmentSpeciesState,
    SpeciesKind, SpeciesParameters, SpeciesSourceMap, SpeciesTransportResult, VesselNode,
    VesselPoint, VesselSegment, VesselTransportOutput,
};
