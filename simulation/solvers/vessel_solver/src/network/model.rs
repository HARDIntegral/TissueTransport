use std::collections::HashMap;

use crate::types::{VesselNode, VesselSegment};

// Graph-level vessel network model.
//
// This type owns the geometric vessel graph only:
//
//     nodes + segments
//
// It should not store pressure, flow, oxygen, carbon dioxide, exchange, or
// rasterized tissue source maps. Those are outputs of later solver stages.
#[derive(Debug, Clone, PartialEq)]
pub struct VesselNetwork {
    pub nodes: Vec<VesselNode>,
    pub segments: Vec<VesselSegment>,
}

impl VesselNetwork {
    /// Create an empty vessel network.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            segments: Vec::new(),
        }
    }

    /// Create a vessel network from already-built node and segment arrays.
    pub fn from_parts(nodes: Vec<VesselNode>, segments: Vec<VesselSegment>) -> Self {
        Self { nodes, segments }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of vessel segments in the graph.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// True when the graph has no nodes and no segments.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.segments.is_empty()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: VesselNode) {
        self.nodes.push(node);
    }

    /// Add a vessel segment to the graph.
    pub fn add_segment(&mut self, segment: VesselSegment) {
        self.segments.push(segment);
    }

    /// Get a node by id.
    pub fn node(&self, node_id: usize) -> Option<&VesselNode> {
        self.nodes.iter().find(|node| node.id == node_id)
    }

    /// Get a vessel segment by id.
    pub fn segment(&self, segment_id: usize) -> Option<&VesselSegment> {
        self.segments
            .iter()
            .find(|segment| segment.id == segment_id)
    }

    /// Build a lookup table from node id to node reference.
    pub fn node_lookup(&self) -> HashMap<usize, &VesselNode> {
        self.nodes.iter().map(|node| (node.id, node)).collect()
    }

    /// Build a lookup table from segment id to segment reference.
    pub fn segment_lookup(&self) -> HashMap<usize, &VesselSegment> {
        self.segments
            .iter()
            .map(|segment| (segment.id, segment))
            .collect()
    }
}

impl Default for VesselNetwork {
    fn default() -> Self {
        Self::new()
    }
}
