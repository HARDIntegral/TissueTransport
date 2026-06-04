use std::collections::HashMap;

use crate::types::{
    DirectedTopology, DownstreamConnection, SegmentSpeciesState, SpeciesKind, SpeciesParameters,
    SpeciesTransportResult,
};

// Propagate one species through the directed vessel network.
//
// This is still a simplified transport model, but it is no longer a zero-exchange
// placeholder. Oxygen and carbon dioxide both create positive tissue source
// terms; their different tissue behavior is handled by the tissue solver.
pub fn propagate_species(
    topology: &DirectedTopology,
    species: &SpeciesParameters,
) -> SpeciesTransportResult {
    let mut node_concentrations = inlet_concentrations(topology, species.inlet_concentration);
    let mut segment_states = Vec::new();

    for connection in ordered_connections(topology) {
        let inlet_concentration = node_concentrations
            .get(&connection.from_node)
            .copied()
            .unwrap_or(species.inlet_concentration);
        let exchanged_amount_per_s =
            segment_exchange_amount(inlet_concentration, connection.flow_um3_per_s, species);
        let outlet_concentration = outlet_concentration_after_exchange(
            inlet_concentration,
            connection.flow_um3_per_s,
            exchanged_amount_per_s,
        );

        segment_states.push(SegmentSpeciesState {
            segment_id: connection.segment_id,
            inlet_concentration,
            outlet_concentration,
            exchanged_amount_per_s,
        });

        node_concentrations
            .entry(connection.to_node)
            .and_modify(|existing| {
                *existing = mix_equal_weight(*existing, outlet_concentration);
            })
            .or_insert(outlet_concentration);
    }

    SpeciesTransportResult {
        species: species.kind,
        segment_states,
    }
}

// Run vessel transport for every species.
pub fn propagate_species_set(
    topology: &DirectedTopology,
    species_set: &[SpeciesParameters],
) -> Vec<SpeciesTransportResult> {
    species_set
        .iter()
        .map(|species| propagate_species(topology, species))
        .collect()
}

fn ordered_connections(topology: &DirectedTopology) -> Vec<DownstreamConnection> {
    let lookup = connection_lookup(&topology.downstream_connections);
    let mut ordered = Vec::new();
    let mut seen = HashMap::new();

    for segment_id in &topology.traversal_segments {
        let Some(connection) = lookup.get(segment_id) else {
            continue;
        };

        ordered.push(*connection);
        seen.insert(*segment_id, true);
    }

    for connection in &topology.downstream_connections {
        if seen.contains_key(&connection.segment_id) {
            continue;
        }

        ordered.push(*connection);
    }

    ordered
}

fn connection_lookup(connections: &[DownstreamConnection]) -> HashMap<usize, DownstreamConnection> {
    connections
        .iter()
        .map(|connection| (connection.segment_id, *connection))
        .collect()
}

fn inlet_concentrations(
    topology: &DirectedTopology,
    inlet_concentration: f32,
) -> HashMap<usize, f32> {
    topology
        .inlet_nodes
        .iter()
        .map(|node_id| (*node_id, inlet_concentration))
        .collect()
}

fn segment_exchange_amount(
    inlet_concentration: f32,
    flow_um3_per_s: f32,
    species: &SpeciesParameters,
) -> f32 {
    let flow = flow_um3_per_s.abs().max(1.0e-6);
    let concentration = inlet_concentration.max(0.0);
    let exchange_fraction = exchange_fraction(flow, species.permeability_um_per_s);
    let amount = flow * concentration * exchange_fraction;

    match species.kind {
        SpeciesKind::Oxygen => amount,
        SpeciesKind::CarbonDioxide => amount,
    }
}

fn exchange_fraction(flow_um3_per_s: f32, permeability_um_per_s: f32) -> f32 {
    let permeability = permeability_um_per_s.max(0.0);
    let flow_scale = flow_um3_per_s.abs().sqrt().max(1.0);
    let fraction = permeability / (permeability + flow_scale);

    fraction.clamp(0.0, 0.25)
}

fn outlet_concentration_after_exchange(
    inlet_concentration: f32,
    flow_um3_per_s: f32,
    exchanged_amount_per_s: f32,
) -> f32 {
    let flow = flow_um3_per_s.abs().max(1.0e-6);
    let outlet = inlet_concentration - exchanged_amount_per_s / flow;

    outlet.max(0.0)
}

fn mix_equal_weight(existing: f32, incoming: f32) -> f32 {
    0.5 * (existing + incoming)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DirectedTopology, DownstreamConnection, SpeciesKind};

    fn oxygen(inlet_concentration: f32) -> SpeciesParameters {
        SpeciesParameters {
            kind: SpeciesKind::Oxygen,
            inlet_concentration,
            diffusivity_um2_per_s: 2_000.0,
            permeability_um_per_s: 1.0,
        }
    }

    fn connection(segment_id: usize, from_node: usize, to_node: usize) -> DownstreamConnection {
        DownstreamConnection {
            segment_id,
            from_node,
            to_node,
            flow_um3_per_s: 1.0,
        }
    }

    #[test]
    fn propagates_species_through_linear_topology() {
        let topology = DirectedTopology {
            inlet_nodes: vec![0],
            outlet_nodes: vec![2],
            traversal_segments: vec![0, 1],
            downstream_connections: vec![connection(0, 0, 1), connection(1, 1, 2)],
        };

        let result = propagate_species(&topology, &oxygen(1.0));

        assert_eq!(result.species, SpeciesKind::Oxygen);
        assert_eq!(result.segment_states.len(), 2);
        assert_eq!(result.segment_states[0].inlet_concentration, 1.0);
        assert!(result.segment_states[0].outlet_concentration < 1.0);
        assert!(result.segment_states[0].exchanged_amount_per_s > 0.0);
        assert_eq!(
            result.segment_states[1].inlet_concentration,
            result.segment_states[0].outlet_concentration
        );
        assert!(
            result.segment_states[1].outlet_concentration
                < result.segment_states[1].inlet_concentration
        );
    }

    #[test]
    fn propagates_species_through_branching_topology() {
        let topology = DirectedTopology {
            inlet_nodes: vec![0],
            outlet_nodes: vec![2, 3],
            traversal_segments: vec![0, 1, 2],
            downstream_connections: vec![
                connection(0, 0, 1),
                connection(1, 1, 2),
                connection(2, 1, 3),
            ],
        };

        let result = propagate_species(&topology, &oxygen(0.75));

        assert_eq!(result.segment_states.len(), 3);
        assert_eq!(result.segment_states[0].inlet_concentration, 0.75);
        assert!(result
            .segment_states
            .iter()
            .all(|state| state.exchanged_amount_per_s > 0.0));
    }

    #[test]
    fn propagates_multiple_species() {
        let topology = DirectedTopology {
            inlet_nodes: vec![0],
            outlet_nodes: vec![1],
            traversal_segments: vec![0],
            downstream_connections: vec![connection(0, 0, 1)],
        };
        let species = vec![
            oxygen(1.0),
            SpeciesParameters {
                kind: SpeciesKind::CarbonDioxide,
                inlet_concentration: 0.25,
                diffusivity_um2_per_s: 2_500.0,
                permeability_um_per_s: 1.0,
            },
        ];

        let results = propagate_species_set(&topology, &species);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].species, SpeciesKind::Oxygen);
        assert_eq!(results[1].species, SpeciesKind::CarbonDioxide);
        assert!(results[0].segment_states[0].exchanged_amount_per_s > 0.0);
        assert!(results[1].segment_states[0].exchanged_amount_per_s > 0.0);
    }

    #[test]
    fn propagates_unreached_directed_connections() {
        let topology = DirectedTopology {
            inlet_nodes: vec![0],
            outlet_nodes: vec![2, 11],
            traversal_segments: vec![0, 1],
            downstream_connections: vec![
                connection(0, 0, 1),
                connection(1, 1, 2),
                connection(10, 10, 11),
            ],
        };

        let result = propagate_species(&topology, &oxygen(1.0));

        assert_eq!(result.segment_states.len(), 3);
        assert!(result
            .segment_states
            .iter()
            .any(|state| state.segment_id == 10));
    }
}
