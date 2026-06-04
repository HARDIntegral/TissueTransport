use std::collections::{HashMap, HashSet};

use crate::types::{
    DirectedTopology, DownstreamConnection, FlowSolution, PerfusionPath, PressureBoundary,
};

// Build directed topology from perfusion-front classification.
//
// Segment direction comes from the perfusion path. The pressure solve only
// contributes flow magnitude values for transport strength.
pub fn build_topology_from_perfusion(
    perfusion: &PerfusionPath,
    flow: &FlowSolution,
) -> DirectedTopology {
    let flow_lookup = segment_flow_lookup(flow);
    let mean_flow = mean_nonzero_flow(flow);
    let mut downstream_connections = perfusion
        .perfused_segments
        .iter()
        .map(|segment| {
            let flow_um3_per_s = flow_lookup
                .get(&segment.segment_id)
                .copied()
                .filter(|flow| *flow > 0.0)
                .unwrap_or(mean_flow);

            DownstreamConnection {
                segment_id: segment.segment_id,
                from_node: segment.from_node,
                to_node: segment.to_node,
                flow_um3_per_s,
            }
        })
        .collect::<Vec<_>>();

    downstream_connections.sort_by_key(|connection| {
        (
            connection.from_node,
            connection.to_node,
            connection.segment_id,
        )
    });

    let traversal_segments = perfusion
        .perfused_segments
        .iter()
        .map(|segment| segment.segment_id)
        .collect::<Vec<_>>();
    let outlet_nodes = find_outlet_nodes(&downstream_connections);

    DirectedTopology {
        inlet_nodes: perfusion.source_nodes.clone(),
        outlet_nodes,
        traversal_segments,
        downstream_connections,
    }
}

// Pick the high-pressure boundary nodes as perfusion source nodes.
pub fn source_nodes_from_pressure_boundaries(boundaries: &[PressureBoundary]) -> Vec<usize> {
    let Some(max_pressure) = boundaries
        .iter()
        .map(|boundary| boundary.pressure_mmhg)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Vec::new();
    };

    boundaries
        .iter()
        .filter(|boundary| (boundary.pressure_mmhg - max_pressure).abs() < 1.0e-6)
        .map(|boundary| boundary.node_id)
        .collect()
}

fn segment_flow_lookup(flow: &FlowSolution) -> HashMap<usize, f32> {
    flow.segment_flows
        .iter()
        .map(|segment_flow| (segment_flow.segment_id, segment_flow.flow_um3_per_s.abs()))
        .collect()
}

fn mean_nonzero_flow(flow: &FlowSolution) -> f32 {
    let nonzero_flows = flow
        .segment_flows
        .iter()
        .map(|segment_flow| segment_flow.flow_um3_per_s.abs())
        .filter(|flow| *flow > 0.0)
        .collect::<Vec<_>>();

    if nonzero_flows.is_empty() {
        return 1.0;
    }

    nonzero_flows.iter().sum::<f32>() / nonzero_flows.len() as f32
}

fn find_outlet_nodes(connections: &[DownstreamConnection]) -> Vec<usize> {
    let from_nodes: HashSet<usize> = connections
        .iter()
        .map(|connection| connection.from_node)
        .collect();
    let to_nodes: HashSet<usize> = connections
        .iter()
        .map(|connection| connection.to_node)
        .collect();

    let mut outlet_nodes: Vec<usize> = to_nodes.difference(&from_nodes).copied().collect();

    outlet_nodes.sort_unstable();
    outlet_nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        FlowSolution, PerfusionPath, PerfusionSegment, PressureBoundary, SegmentFlow,
    };

    fn flow(segment_flows: Vec<SegmentFlow>) -> FlowSolution {
        FlowSolution {
            node_pressures: Vec::new(),
            segment_flows,
        }
    }

    fn segment_flow(segment_id: usize, flow_um3_per_s: f32) -> SegmentFlow {
        SegmentFlow {
            segment_id,
            flow_um3_per_s,
        }
    }

    #[test]
    fn builds_topology_from_perfusion_path() {
        let solution = flow(vec![segment_flow(0, -5.0), segment_flow(1, 3.0)]);
        let perfusion = PerfusionPath {
            source_nodes: vec![0],
            perfused_segments: vec![
                PerfusionSegment {
                    segment_id: 0,
                    from_node: 0,
                    to_node: 1,
                    parent_node: 0,
                    path_distance: 1.0,
                },
                PerfusionSegment {
                    segment_id: 1,
                    from_node: 1,
                    to_node: 2,
                    parent_node: 0,
                    path_distance: 2.0,
                },
            ],
            collisions: Vec::new(),
        };

        let topology = build_topology_from_perfusion(&perfusion, &solution);

        assert_eq!(topology.inlet_nodes, vec![0]);
        assert_eq!(topology.outlet_nodes, vec![2]);
        assert_eq!(topology.traversal_segments, vec![0, 1]);
        assert_eq!(topology.downstream_connections[0].from_node, 0);
        assert_eq!(topology.downstream_connections[0].to_node, 1);
        assert_eq!(topology.downstream_connections[0].flow_um3_per_s, 5.0);
        assert_eq!(topology.downstream_connections[1].from_node, 1);
        assert_eq!(topology.downstream_connections[1].to_node, 2);
        assert_eq!(topology.downstream_connections[1].flow_um3_per_s, 3.0);
    }

    #[test]
    fn source_nodes_use_highest_pressure_boundaries() {
        let sources = source_nodes_from_pressure_boundaries(&[
            PressureBoundary {
                node_id: 0,
                pressure_mmhg: 15.0,
            },
            PressureBoundary {
                node_id: 1,
                pressure_mmhg: 35.0,
            },
            PressureBoundary {
                node_id: 2,
                pressure_mmhg: 35.0,
            },
        ]);

        assert_eq!(sources, vec![1, 2]);
    }
}
