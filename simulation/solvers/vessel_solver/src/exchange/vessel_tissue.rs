use crate::types::{ExchangeResult, SegmentExchange, SpeciesTransportResult};

// Compute vessel-to-tissue exchange for a transported species.
//
// This stage sits between blood transport and source-map generation:
//
//     blood transport
//         ->
//     vessel-tissue exchange
//         ->
//     source maps
//
// For now, exchange is a simple passthrough placeholder. The transport stage
// already stores `exchanged_amount_per_s` for each segment state. Later, this
// file will own permeability models, membrane transport, and tissue-coupled
// exchange kinetics.
pub fn compute_exchange(transport: &SpeciesTransportResult) -> ExchangeResult {
    let segment_exchanges = transport
        .segment_states
        .iter()
        .map(|state| SegmentExchange {
            segment_id: state.segment_id,
            species: transport.species,
            source_amount_per_s: state.exchanged_amount_per_s,
        })
        .collect();

    ExchangeResult { segment_exchanges }
}

// Compute exchange for multiple transported species.
pub fn compute_exchange_set(transports: &[SpeciesTransportResult]) -> ExchangeResult {
    let mut segment_exchanges = Vec::new();

    for transport in transports {
        segment_exchanges.extend(compute_exchange(transport).segment_exchanges);
    }

    ExchangeResult { segment_exchanges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SegmentSpeciesState, SpeciesKind, SpeciesTransportResult};

    #[test]
    fn converts_transport_state_into_exchange_result() {
        let transport = SpeciesTransportResult {
            species: SpeciesKind::Oxygen,
            segment_states: vec![SegmentSpeciesState {
                segment_id: 7,
                inlet_concentration: 1.0,
                outlet_concentration: 0.9,
                exchanged_amount_per_s: 0.1,
            }],
        };

        let exchange = compute_exchange(&transport);

        assert_eq!(exchange.segment_exchanges.len(), 1);
        assert_eq!(exchange.segment_exchanges[0].segment_id, 7);
        assert_eq!(exchange.segment_exchanges[0].species, SpeciesKind::Oxygen);
        assert_eq!(exchange.segment_exchanges[0].source_amount_per_s, 0.1);
    }

    #[test]
    fn combines_multiple_species_exchange_results() {
        let oxygen = SpeciesTransportResult {
            species: SpeciesKind::Oxygen,
            segment_states: vec![SegmentSpeciesState {
                segment_id: 1,
                inlet_concentration: 1.0,
                outlet_concentration: 1.0,
                exchanged_amount_per_s: 0.1,
            }],
        };

        let carbon_dioxide = SpeciesTransportResult {
            species: SpeciesKind::CarbonDioxide,
            segment_states: vec![SegmentSpeciesState {
                segment_id: 2,
                inlet_concentration: 0.0,
                outlet_concentration: 0.0,
                exchanged_amount_per_s: 0.2,
            }],
        };

        let exchange = compute_exchange_set(&[oxygen, carbon_dioxide]);

        assert_eq!(exchange.segment_exchanges.len(), 2);
    }
}
