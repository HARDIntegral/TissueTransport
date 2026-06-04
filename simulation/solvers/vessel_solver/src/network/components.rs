use std::collections::{HashMap, HashSet, VecDeque};

use crate::network::model::VesselNetwork;

// One undirected connected component of the vessel graph.
//
// Components only describe graph connectivity. They do not know whether a
// component is perfused, pressurized, oxygenated, or biologically valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkComponent {
    pub id: usize,
    pub node_ids: Vec<usize>,
    pub segment_ids: Vec<usize>,
}

// Build all undirected connected components in the vessel network.
//
// A segment connects its start and end nodes bidirectionally for this stage.
// Flow direction is handled later by the flow/topology stages.
pub fn connected_components(network: &VesselNetwork) -> Vec<NetworkComponent> {
    let node_to_segments = build_node_to_segments(network);
    let mut visited_nodes = HashSet::new();
    let mut components = Vec::new();

    for node in &network.nodes {
        if visited_nodes.contains(&node.id) {
            continue;
        }

        let component = walk_component(
            components.len(),
            node.id,
            network,
            &node_to_segments,
            &mut visited_nodes,
        );

        components.push(component);
    }

    components
}

// Return the largest connected component by segment count.
//
// Node count is used as a tiebreaker. This is useful for debugging and for
// quick visual sanity checks, but it should not be treated as a perfusion
// classifier.
pub fn largest_component(components: &[NetworkComponent]) -> Option<&NetworkComponent> {
    components
        .iter()
        .max_by_key(|component| (component.segment_ids.len(), component.node_ids.len()))
}

// Return all components with at least `min_segments` vessel segments.
pub fn components_with_min_segments(
    components: &[NetworkComponent],
    min_segments: usize,
) -> Vec<&NetworkComponent> {
    components
        .iter()
        .filter(|component| component.segment_ids.len() >= min_segments)
        .collect()
}

fn build_node_to_segments(network: &VesselNetwork) -> HashMap<usize, Vec<usize>> {
    let mut node_to_segments: HashMap<usize, Vec<usize>> = HashMap::new();

    for segment in &network.segments {
        node_to_segments
            .entry(segment.start_node)
            .or_default()
            .push(segment.id);

        node_to_segments
            .entry(segment.end_node)
            .or_default()
            .push(segment.id);
    }

    node_to_segments
}

fn walk_component(
    component_id: usize,
    start_node: usize,
    network: &VesselNetwork,
    node_to_segments: &HashMap<usize, Vec<usize>>,
    visited_nodes: &mut HashSet<usize>,
) -> NetworkComponent {
    let segment_lookup = network.segment_lookup();
    let mut queue = VecDeque::new();
    let mut node_ids = Vec::new();
    let mut segment_ids = HashSet::new();

    visited_nodes.insert(start_node);
    queue.push_back(start_node);

    while let Some(node_id) = queue.pop_front() {
        node_ids.push(node_id);

        let Some(attached_segments) = node_to_segments.get(&node_id) else {
            continue;
        };

        for segment_id in attached_segments {
            let Some(segment) = segment_lookup.get(segment_id) else {
                continue;
            };

            segment_ids.insert(*segment_id);

            let neighbor_id = if segment.start_node == node_id {
                segment.end_node
            } else {
                segment.start_node
            };

            if visited_nodes.insert(neighbor_id) {
                queue.push_back(neighbor_id);
            }
        }
    }

    node_ids.sort_unstable();

    let mut segment_ids: Vec<usize> = segment_ids.into_iter().collect();
    segment_ids.sort_unstable();

    NetworkComponent {
        id: component_id,
        node_ids,
        segment_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeKind, VesselNode, VesselSegment};

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
            length_um: 10.0,
            radius_um: 2.0,
            centerline: Vec::new(),
        }
    }

    #[test]
    fn finds_single_connected_component() {
        let network = VesselNetwork::from_parts(
            vec![node(0), node(1), node(2)],
            vec![segment(0, 0, 1), segment(1, 1, 2)],
        );

        let components = connected_components(&network);

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].node_ids, vec![0, 1, 2]);
        assert_eq!(components[0].segment_ids, vec![0, 1]);
    }

    #[test]
    fn finds_multiple_connected_components() {
        let network = VesselNetwork::from_parts(
            vec![node(0), node(1), node(2), node(3)],
            vec![segment(0, 0, 1), segment(1, 2, 3)],
        );

        let components = connected_components(&network);

        assert_eq!(components.len(), 2);
        assert_eq!(components[0].node_ids, vec![0, 1]);
        assert_eq!(components[0].segment_ids, vec![0]);
        assert_eq!(components[1].node_ids, vec![2, 3]);
        assert_eq!(components[1].segment_ids, vec![1]);
    }

    #[test]
    fn largest_component_prefers_more_segments() {
        let network = VesselNetwork::from_parts(
            vec![node(0), node(1), node(2), node(3), node(4)],
            vec![segment(0, 0, 1), segment(1, 2, 3), segment(2, 3, 4)],
        );

        let components = connected_components(&network);
        let largest = largest_component(&components).unwrap();

        assert_eq!(largest.node_ids, vec![2, 3, 4]);
        assert_eq!(largest.segment_ids, vec![1, 2]);
    }
}
