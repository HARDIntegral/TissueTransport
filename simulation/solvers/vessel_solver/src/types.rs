// Kind of node extracted from the vessel graph.
//
// This is mostly useful for debugging and later boundary selection. The flow
// solver only needs connectivity, but knowing whether a node is an endpoint or
// branch point helps when picking inlet/outlet candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    EndPoint,
    BranchPoint,
    Unknown,
}

// One node in the vessel graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VesselNode {
    pub id: usize,
    pub x_um: f32,
    pub y_um: f32,
    pub kind: NodeKind,
}

// One sampled point along a vessel centerline.
//
// Centerline points preserve the real curved path extracted from the skeleton
// instead of forcing each vessel segment to behave like a straight line between
// two graph nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VesselPoint {
    pub x_um: f32,
    pub y_um: f32,
}

// One vessel segment connecting two nodes.
//
// Segments are geometric vessel centerline pieces. They do not store flow,
// pressure, oxygen, carbon dioxide, or tissue exchange state. Those values are
// produced by later solver stages.
//
// centerline stores the actual skeleton path for this segment, including
// intermediate curved points when available. This allows source-map
// rasterization to follow the real vessel shape instead of drawing a straight
// line from start_node to end_node.
#[derive(Debug, Clone, PartialEq)]
pub struct VesselSegment {
    pub id: usize,
    pub start_node: usize,
    pub end_node: usize,
    pub length_um: f32,
    pub radius_um: f32,
    pub centerline: Vec<VesselPoint>,
}

// Pressure boundary condition assigned to a vessel node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureBoundary {
    pub node_id: usize,
    pub pressure_mmhg: f32,
}

// Solved pressure at a node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodePressure {
    pub node_id: usize,
    pub pressure_mmhg: f32,
}

// Solved signed hydraulic flow value for one vessel segment.
//
// The sign is relative to the segment's stored orientation:
//
//     start_node -> end_node
//
// A positive value means flow follows that orientation. A negative value means
// flow moves from end_node to start_node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentFlow {
    pub segment_id: usize,
    pub flow_um3_per_s: f32,
}

// Complete hydraulic solution for a vessel network.
//
// This is the output of the flow stage only. It should not contain oxygen,
// carbon dioxide, vessel-tissue exchange, or rasterized source maps.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowSolution {
    pub node_pressures: Vec<NodePressure>,
    pub segment_flows: Vec<SegmentFlow>,
}

// One directed outgoing connection from a node to a downstream segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DownstreamConnection {
    pub segment_id: usize,
    pub from_node: usize,
    pub to_node: usize,
    pub flow_um3_per_s: f32,
}

// Directed graph view used by blood chemistry transport.
//
// Segment direction comes from perfusion-front classification. Flow magnitudes
// come from the pressure-flow solve.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectedTopology {
    pub inlet_nodes: Vec<usize>,
    pub outlet_nodes: Vec<usize>,
    pub traversal_segments: Vec<usize>,
    pub downstream_connections: Vec<DownstreamConnection>,
}

// One segment assigned by the perfusion-front classifier.
//
// This is not the pressure solver output. It is a graph-level guess for how
// blood should move through the vessel tree before the pressure solve cleans up
// magnitudes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerfusionSegment {
    pub segment_id: usize,
    pub from_node: usize,
    pub to_node: usize,
    pub parent_node: usize,
    pub path_distance: f32,
}

// One collision between two flow fronts during perfusion classification.
//
// The first front to claim a node or segment keeps the direction. Later fronts
// are recorded here instead of flipping the existing direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerfusionCollision {
    pub node_id: usize,
    pub incoming_from_node: usize,
    pub existing_parent_node: usize,
}

// Full perfusion-front classifier result.
#[derive(Debug, Clone, PartialEq)]
pub struct PerfusionPath {
    pub source_nodes: Vec<usize>,
    pub perfused_segments: Vec<PerfusionSegment>,
    pub collisions: Vec<PerfusionCollision>,
}

// Species identifier for transported blood chemistry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeciesKind {
    Oxygen,
    CarbonDioxide,
}

// Parameters for one transported species.
//
// Oxygen and carbon dioxide should share the same transport algorithm when
// possible. Their differences should mostly live in these parameters and in
// exchange/production rules, not in duplicated traversal code.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesParameters {
    pub kind: SpeciesKind,
    pub inlet_concentration: f32,
    pub diffusivity_um2_per_s: f32,
    pub permeability_um_per_s: f32,
}

// Blood concentration state for one species along one segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentSpeciesState {
    pub segment_id: usize,
    pub inlet_concentration: f32,
    pub outlet_concentration: f32,
    pub exchanged_amount_per_s: f32,
}

// Transport result for one species inside the directed vessel network.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesTransportResult {
    pub species: SpeciesKind,
    pub segment_states: Vec<SegmentSpeciesState>,
}

// Vessel-tissue exchange result for one segment and one species.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentExchange {
    pub segment_id: usize,
    pub species: SpeciesKind,
    pub source_amount_per_s: f32,
}

// Collection of vessel-tissue exchange outputs before rasterization.
#[derive(Debug, Clone, PartialEq)]
pub struct ExchangeResult {
    pub segment_exchanges: Vec<SegmentExchange>,
}

// Rasterized source map for one species on the tissue grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesSourceMap {
    pub species: SpeciesKind,
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

// Final boundary/source maps sent back to the tissue diffusion solver.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundarySourceMaps {
    pub maps: Vec<SpeciesSourceMap>,
}

// Full high-level vessel transport output.
//
// This is the final output of the Rust vessel solver pipeline before Python
// visualizes the result or passes the source maps into the tissue solver.
#[derive(Debug, Clone, PartialEq)]
pub struct VesselTransportOutput {
    pub flow: FlowSolution,
    pub topology: DirectedTopology,
    pub species_transport: Vec<SpeciesTransportResult>,
    pub exchange: ExchangeResult,
    pub boundary_source_maps: BoundarySourceMaps,
}
