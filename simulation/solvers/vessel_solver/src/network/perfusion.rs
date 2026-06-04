use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::types::VesselSegment;
pub use crate::types::{PerfusionCollision, PerfusionPath, PerfusionSegment};
use crate::VesselNetwork;

#[derive(Debug, Clone, Copy, PartialEq)]
struct FrontState {
    node_id: usize,
    parent_node: usize,
    distance: f32,
}

impl Eq for FrontState {}

impl Ord for FrontState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for FrontState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy)]
struct NeighborSegment {
    segment_id: usize,
    to_node: usize,
    cost: f32,
}

// Classify likely blood paths by growing flow fronts from source nodes.
//
// This does not replace the pressure solver. It gives the solver and the
// diagnostics a graph-level direction guess that avoids arbitrary inlet/outlet
// flips from node-id ordering.
pub fn classify_perfusion_paths(network: &VesselNetwork, source_nodes: &[usize]) -> PerfusionPath {
    let adjacency = build_adjacency(network);
    let sources = valid_source_nodes(network, source_nodes);
    let mut queue = BinaryHeap::new();
    let mut best_node_distance: HashMap<usize, f32> = HashMap::new();
    let mut node_parent: HashMap<usize, usize> = HashMap::new();
    let mut claimed_segments: HashSet<usize> = HashSet::new();
    let mut perfused_segments = Vec::new();
    let mut collisions = Vec::new();

    for source in &sources {
        best_node_distance.insert(*source, 0.0);
        node_parent.insert(*source, *source);
        queue.push(FrontState {
            node_id: *source,
            parent_node: *source,
            distance: 0.0,
        });
    }

    while let Some(state) = queue.pop() {
        let Some(current_best) = best_node_distance.get(&state.node_id) else {
            continue;
        };

        if state.distance > *current_best {
            continue;
        }

        let Some(neighbors) = adjacency.get(&state.node_id) else {
            continue;
        };

        for neighbor in neighbors {
            let next_distance = state.distance + neighbor.cost;

            if best_node_distance.contains_key(&neighbor.to_node) {
                record_collision(
                    &mut collisions,
                    neighbor.to_node,
                    state.node_id,
                    &node_parent,
                );
                continue;
            }

            claimed_segments.insert(neighbor.segment_id);
            best_node_distance.insert(neighbor.to_node, next_distance);
            node_parent.insert(neighbor.to_node, state.node_id);
            perfused_segments.push(PerfusionSegment {
                segment_id: neighbor.segment_id,
                from_node: state.node_id,
                to_node: neighbor.to_node,
                parent_node: state.parent_node,
                path_distance: next_distance,
            });
            queue.push(FrontState {
                node_id: neighbor.to_node,
                parent_node: state.node_id,
                distance: next_distance,
            });
        }
    }

    append_unclaimed_segments(
        network,
        &mut claimed_segments,
        &mut best_node_distance,
        &mut node_parent,
        &mut perfused_segments,
    );

    PerfusionPath {
        source_nodes: sources,
        perfused_segments,
        collisions,
    }
}

fn build_adjacency(network: &VesselNetwork) -> HashMap<usize, Vec<NeighborSegment>> {
    let mut adjacency: HashMap<usize, Vec<NeighborSegment>> = HashMap::new();

    for segment in &network.segments {
        let cost = segment_cost(segment);
        adjacency
            .entry(segment.start_node)
            .or_default()
            .push(NeighborSegment {
                segment_id: segment.id,
                to_node: segment.end_node,
                cost,
            });
        adjacency
            .entry(segment.end_node)
            .or_default()
            .push(NeighborSegment {
                segment_id: segment.id,
                to_node: segment.start_node,
                cost,
            });
    }

    adjacency
}

fn segment_cost(segment: &VesselSegment) -> f32 {
    let length = segment.length_um.max(1.0);
    let radius = segment.radius_um.max(1.0e-3);

    length / radius.powi(4)
}

fn valid_source_nodes(network: &VesselNetwork, source_nodes: &[usize]) -> Vec<usize> {
    let network_nodes: HashSet<usize> = network.nodes.iter().map(|node| node.id).collect();
    let mut sources = source_nodes
        .iter()
        .copied()
        .filter(|node_id| network_nodes.contains(node_id))
        .collect::<Vec<_>>();

    if sources.is_empty() {
        if let Some(first_node) = network.nodes.first() {
            sources.push(first_node.id);
        }
    }

    sources.sort_unstable();
    sources.dedup();
    sources
}

fn append_unclaimed_segments(
    network: &VesselNetwork,
    claimed_segments: &mut HashSet<usize>,
    reached_nodes: &mut HashMap<usize, f32>,
    node_parent: &mut HashMap<usize, usize>,
    perfused_segments: &mut Vec<PerfusionSegment>,
) {
    let mut added_segment = true;

    while added_segment {
        added_segment = false;

        for segment in &network.segments {
            if claimed_segments.contains(&segment.id) {
                continue;
            }

            let start_reached = reached_nodes.contains_key(&segment.start_node);
            let end_reached = reached_nodes.contains_key(&segment.end_node);

            match (start_reached, end_reached) {
                (true, false) => {
                    append_unclaimed_segment(
                        segment,
                        segment.start_node,
                        segment.end_node,
                        claimed_segments,
                        reached_nodes,
                        node_parent,
                        perfused_segments,
                    );
                    added_segment = true;
                }
                (false, true) => {
                    append_unclaimed_segment(
                        segment,
                        segment.end_node,
                        segment.start_node,
                        claimed_segments,
                        reached_nodes,
                        node_parent,
                        perfused_segments,
                    );
                    added_segment = true;
                }
                (true, true) => {
                    claimed_segments.insert(segment.id);
                }
                (false, false) => {}
            }
        }
    }
}

fn append_unclaimed_segment(
    segment: &VesselSegment,
    from_node: usize,
    to_node: usize,
    claimed_segments: &mut HashSet<usize>,
    reached_nodes: &mut HashMap<usize, f32>,
    node_parent: &mut HashMap<usize, usize>,
    perfused_segments: &mut Vec<PerfusionSegment>,
) {
    let parent_distance = reached_nodes.get(&from_node).copied().unwrap_or(0.0);
    let next_distance = parent_distance + segment_cost(segment);

    claimed_segments.insert(segment.id);
    reached_nodes.insert(to_node, next_distance);
    node_parent.insert(to_node, from_node);
    perfused_segments.push(PerfusionSegment {
        segment_id: segment.id,
        from_node,
        to_node,
        parent_node: from_node,
        path_distance: next_distance,
    });
}

fn record_collision(
    collisions: &mut Vec<PerfusionCollision>,
    node_id: usize,
    incoming_from_node: usize,
    node_parent: &HashMap<usize, usize>,
) {
    let existing_parent_node = node_parent.get(&node_id).copied().unwrap_or(node_id);

    if existing_parent_node == incoming_from_node {
        return;
    }

    collisions.push(PerfusionCollision {
        node_id,
        incoming_from_node,
        existing_parent_node,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeKind, VesselNode, VesselSegment};

    #[test]
    fn classifies_linear_network_from_source() {
        let network = linear_network();
        let path = classify_perfusion_paths(&network, &[0]);

        assert_eq!(path.source_nodes, vec![0]);
        assert_eq!(path.perfused_segments.len(), 2);
        assert_eq!(path.perfused_segments[0].from_node, 0);
        assert_eq!(path.perfused_segments[0].to_node, 1);
        assert_eq!(path.perfused_segments[1].from_node, 1);
        assert_eq!(path.perfused_segments[1].to_node, 2);
    }

    #[test]
    fn classifies_forked_network_from_source() {
        let network = forked_network();
        let path = classify_perfusion_paths(&network, &[0]);
        let segment_ids = path
            .perfused_segments
            .iter()
            .map(|segment| segment.segment_id)
            .collect::<HashSet<_>>();

        assert_eq!(segment_ids.len(), 3);
        assert!(segment_ids.contains(&0));
        assert!(segment_ids.contains(&1));
        assert!(segment_ids.contains(&2));
    }

    #[test]
    fn records_collision_without_flipping_existing_path() {
        let network = loop_network();
        let path = classify_perfusion_paths(&network, &[0, 3]);

        assert!(!path.collisions.is_empty());
        assert!(path.perfused_segments.len() <= network.segments.len());
    }

    #[test]
    fn collision_does_not_claim_segment_into_existing_node() {
        let network = loop_network();
        let path = classify_perfusion_paths(&network, &[0, 3]);
        let mut directed_edges = path
            .perfused_segments
            .iter()
            .map(|segment| (segment.from_node, segment.to_node))
            .collect::<Vec<_>>();

        directed_edges.sort_unstable();

        assert!(!directed_edges.contains(&(1, 2)) || !directed_edges.contains(&(3, 2)));
        assert!(!directed_edges.contains(&(2, 1)) || !directed_edges.contains(&(0, 1)));
    }

    #[test]
    fn appends_unclaimed_branch_away_from_reached_tree() {
        let network = delayed_branch_network();
        let path = classify_perfusion_paths(&network, &[0, 3]);
        let branch = path
            .perfused_segments
            .iter()
            .find(|segment| segment.segment_id == 3)
            .expect("branch segment should be appended");

        assert_eq!(branch.from_node, 1);
        assert_eq!(branch.to_node, 4);
    }

    fn linear_network() -> VesselNetwork {
        VesselNetwork {
            nodes: vec![node(0), node(1), node(2)],
            segments: vec![segment(0, 0, 1), segment(1, 1, 2)],
        }
    }

    fn forked_network() -> VesselNetwork {
        VesselNetwork {
            nodes: vec![node(0), node(1), node(2), node(3)],
            segments: vec![segment(0, 0, 1), segment(1, 1, 2), segment(2, 1, 3)],
        }
    }

    fn loop_network() -> VesselNetwork {
        VesselNetwork {
            nodes: vec![node(0), node(1), node(2), node(3)],
            segments: vec![
                segment(0, 0, 1),
                segment(1, 1, 2),
                segment(2, 2, 3),
                segment(3, 0, 3),
            ],
        }
    }

    fn delayed_branch_network() -> VesselNetwork {
        VesselNetwork {
            nodes: vec![node(0), node(1), node(2), node(3), node(4)],
            segments: vec![
                segment(0, 0, 1),
                segment(1, 1, 2),
                segment(2, 2, 3),
                segment(3, 1, 4),
            ],
        }
    }

    fn node(id: usize) -> VesselNode {
        VesselNode {
            id,
            x_um: id as f32,
            y_um: 0.0,
            kind: NodeKind::Unknown,
        }
    }

    fn segment(id: usize, start_node: usize, end_node: usize) -> VesselSegment {
        VesselSegment {
            id,
            start_node,
            end_node,
            length_um: 1.0,
            radius_um: 1.0,
            centerline: Vec::new(),
        }
    }
}
