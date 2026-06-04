use std::collections::HashMap;

use crate::flow::resistance::{poiseuille_resistance, M3_TO_UM3};
use crate::network::model::VesselNetwork;
use crate::types::{FlowSolution, NodePressure, PressureBoundary, SegmentFlow};

const MMHG_TO_PA: f32 = 133.322;

// Solve vessel pressure and signed segment flow for a vessel network.
//
// This function owns the pressure stage only.
//
// pressure boundaries
//     ->
// linear pressure solve
//     ->
// node pressures
//     ->
// signed segment flows
//
// It does not reconstruct directed flow topology and does not perform any
// blood chemistry transport.
pub fn solve_pressure_flow(
    network: &VesselNetwork,
    boundaries: &[PressureBoundary],
    viscosity_pa_s: f32,
) -> FlowSolution {
    let boundary_lookup = boundary_lookup(boundaries);
    let node_pressures = solve_node_pressures(network, &boundary_lookup, viscosity_pa_s);
    let pressure_lookup = pressure_lookup(&node_pressures);
    let segment_flows = build_segment_flows(network, &pressure_lookup, viscosity_pa_s);

    FlowSolution {
        node_pressures,
        segment_flows,
    }
}

fn boundary_lookup(boundaries: &[PressureBoundary]) -> HashMap<usize, f32> {
    boundaries
        .iter()
        .map(|boundary| (boundary.node_id, boundary.pressure_mmhg))
        .collect()
}

fn solve_node_pressures(
    network: &VesselNetwork,
    boundary_lookup: &HashMap<usize, f32>,
    viscosity_pa_s: f32,
) -> Vec<NodePressure> {
    let node_index_lookup = node_index_lookup(network);
    let node_count = network.node_count();
    let mut matrix = vec![vec![0.0; node_count]; node_count];
    let mut rhs = vec![0.0; node_count];

    for node in &network.nodes {
        let row = node_index_lookup[&node.id];

        if let Some(boundary_pressure_mmhg) = boundary_lookup.get(&node.id) {
            matrix[row][row] = 1.0;
            rhs[row] = *boundary_pressure_mmhg;
            continue;
        }

        for segment in network
            .segments
            .iter()
            .filter(|segment| segment.start_node == node.id || segment.end_node == node.id)
        {
            let neighbor_id = if segment.start_node == node.id {
                segment.end_node
            } else {
                segment.start_node
            };

            let neighbor_col = node_index_lookup[&neighbor_id];
            let resistance =
                poiseuille_resistance(segment.length_um, segment.radius_um, viscosity_pa_s);

            if !resistance.is_finite() || resistance <= 0.0 {
                continue;
            }

            let conductance = 1.0 / resistance;
            matrix[row][row] += conductance;
            matrix[row][neighbor_col] -= conductance;
        }
    }

    let pressures_mmhg = solve_linear_system(matrix, rhs);

    network
        .nodes
        .iter()
        .map(|node| {
            let index = node_index_lookup[&node.id];

            NodePressure {
                node_id: node.id,
                pressure_mmhg: pressures_mmhg[index],
            }
        })
        .collect()
}

fn build_segment_flows(
    network: &VesselNetwork,
    pressure_lookup: &HashMap<usize, f32>,
    viscosity_pa_s: f32,
) -> Vec<SegmentFlow> {
    network
        .segments
        .iter()
        .map(|segment| {
            let start_pressure_pa = pressure_lookup[&segment.start_node] * MMHG_TO_PA;
            let end_pressure_pa = pressure_lookup[&segment.end_node] * MMHG_TO_PA;
            let resistance =
                poiseuille_resistance(segment.length_um, segment.radius_um, viscosity_pa_s);

            let flow_m3_per_s = if resistance.is_finite() && resistance > 0.0 {
                (start_pressure_pa - end_pressure_pa) / resistance
            } else {
                0.0
            };

            SegmentFlow {
                segment_id: segment.id,
                flow_um3_per_s: flow_m3_per_s * M3_TO_UM3,
            }
        })
        .collect()
}

fn node_index_lookup(network: &VesselNetwork) -> HashMap<usize, usize> {
    network
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect()
}

fn pressure_lookup(node_pressures: &[NodePressure]) -> HashMap<usize, f32> {
    node_pressures
        .iter()
        .map(|pressure| (pressure.node_id, pressure.pressure_mmhg))
        .collect()
}

fn solve_linear_system(mut matrix: Vec<Vec<f32>>, mut rhs: Vec<f32>) -> Vec<f32> {
    let n = rhs.len();

    for pivot_index in 0..n {
        let best_row = find_best_pivot_row(&matrix, pivot_index);

        if best_row != pivot_index {
            matrix.swap(pivot_index, best_row);
            rhs.swap(pivot_index, best_row);
        }

        let pivot = matrix[pivot_index][pivot_index];

        if pivot == 0.0 {
            continue;
        }

        for col in pivot_index..n {
            matrix[pivot_index][col] /= pivot;
        }
        rhs[pivot_index] /= pivot;

        for row in 0..n {
            if row == pivot_index {
                continue;
            }

            let factor = matrix[row][pivot_index];

            for col in pivot_index..n {
                matrix[row][col] -= factor * matrix[pivot_index][col];
            }
            rhs[row] -= factor * rhs[pivot_index];
        }
    }

    rhs
}

fn find_best_pivot_row(matrix: &[Vec<f32>], pivot_index: usize) -> usize {
    let mut best_row = pivot_index;
    let mut best_value = matrix[pivot_index][pivot_index].abs();

    for (row, values) in matrix.iter().enumerate().skip(pivot_index + 1) {
        let value = values[pivot_index].abs();

        if value > best_value {
            best_row = row;
            best_value = value;
        }
    }

    best_row
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
            length_um: 100.0,
            radius_um: 5.0,
            centerline: Vec::new(),
        }
    }

    #[test]
    fn solves_single_segment_pressure_boundaries() {
        let network = VesselNetwork::from_parts(vec![node(0), node(1)], vec![segment(0, 0, 1)]);
        let boundaries = vec![
            PressureBoundary {
                node_id: 0,
                pressure_mmhg: 50.0,
            },
            PressureBoundary {
                node_id: 1,
                pressure_mmhg: 20.0,
            },
        ];

        let solution = solve_pressure_flow(&network, &boundaries, 0.004);

        assert_eq!(solution.node_pressures.len(), 2);
        assert_eq!(solution.segment_flows.len(), 1);
        assert!(solution.segment_flows[0].flow_um3_per_s > 0.0);
    }

    #[test]
    fn solves_middle_pressure_in_symmetric_chain() {
        let network = VesselNetwork::from_parts(
            vec![node(0), node(1), node(2)],
            vec![segment(0, 0, 1), segment(1, 1, 2)],
        );
        let boundaries = vec![
            PressureBoundary {
                node_id: 0,
                pressure_mmhg: 60.0,
            },
            PressureBoundary {
                node_id: 2,
                pressure_mmhg: 20.0,
            },
        ];

        let solution = solve_pressure_flow(&network, &boundaries, 0.004);
        let middle_pressure = solution
            .node_pressures
            .iter()
            .find(|pressure| pressure.node_id == 1)
            .unwrap()
            .pressure_mmhg;

        assert!((middle_pressure - 40.0).abs() < 1.0e-3);
    }

    #[test]
    fn signed_flow_follows_segment_orientation() {
        let network = VesselNetwork::from_parts(vec![node(0), node(1)], vec![segment(0, 0, 1)]);
        let boundaries = vec![
            PressureBoundary {
                node_id: 0,
                pressure_mmhg: 20.0,
            },
            PressureBoundary {
                node_id: 1,
                pressure_mmhg: 50.0,
            },
        ];

        let solution = solve_pressure_flow(&network, &boundaries, 0.004);

        assert!(solution.segment_flows[0].flow_um3_per_s < 0.0);
    }
}
