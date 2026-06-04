use std::collections::HashMap;

use crate::types::{BoundarySourceMaps, ExchangeResult, SpeciesKind, SpeciesSourceMap};

// Build empty boundary/source maps for the requested species.
//
// This file owns source-map allocation and aggregation. It does not decide
// which grid cells belong to a vessel segment. Segment-to-cell mapping belongs
// in maps::rasterize.
pub fn empty_boundary_source_maps(
    width: usize,
    height: usize,
    species: &[SpeciesKind],
) -> BoundarySourceMaps {
    let maps = species
        .iter()
        .map(|species| SpeciesSourceMap {
            species: *species,
            width,
            height,
            values: vec![0.0; width * height],
        })
        .collect();

    BoundarySourceMaps { maps }
}

// Build source maps from exchange results and segment rasterization data.
//
// segment_cells maps:
//
//     segment_id -> grid cell indices
//
// Each segment exchange amount is distributed evenly across the cells assigned
// to that vessel segment.
pub fn build_boundary_source_maps(
    width: usize,
    height: usize,
    exchange: &ExchangeResult,
    segment_cells: &HashMap<usize, Vec<usize>>,
) -> BoundarySourceMaps {
    let species = species_in_exchange(exchange);
    let mut maps = empty_boundary_source_maps(width, height, &species);

    for segment_exchange in &exchange.segment_exchanges {
        let Some(cells) = segment_cells.get(&segment_exchange.segment_id) else {
            continue;
        };

        if cells.is_empty() {
            continue;
        }

        let source_per_cell = segment_exchange.source_amount_per_s / cells.len() as f32;

        let Some(map) = maps
            .maps
            .iter_mut()
            .find(|map| map.species == segment_exchange.species)
        else {
            continue;
        };

        for cell_index in cells {
            if *cell_index < map.values.len() {
                map.values[*cell_index] += source_per_cell;
            }
        }
    }

    maps
}

fn species_in_exchange(exchange: &ExchangeResult) -> Vec<SpeciesKind> {
    let mut species = Vec::new();

    for segment_exchange in &exchange.segment_exchanges {
        if !species.contains(&segment_exchange.species) {
            species.push(segment_exchange.species);
        }
    }

    species
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SegmentExchange, SpeciesKind};

    #[test]
    fn creates_empty_maps_for_species() {
        let maps =
            empty_boundary_source_maps(3, 2, &[SpeciesKind::Oxygen, SpeciesKind::CarbonDioxide]);

        assert_eq!(maps.maps.len(), 2);
        assert_eq!(maps.maps[0].values.len(), 6);
        assert_eq!(maps.maps[1].values.len(), 6);
        assert!(maps.maps[0].values.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn distributes_segment_exchange_across_cells() {
        let exchange = ExchangeResult {
            segment_exchanges: vec![SegmentExchange {
                segment_id: 4,
                species: SpeciesKind::Oxygen,
                source_amount_per_s: 9.0,
            }],
        };
        let mut segment_cells = HashMap::new();
        segment_cells.insert(4, vec![0, 1, 2]);

        let maps = build_boundary_source_maps(3, 3, &exchange, &segment_cells);

        assert_eq!(maps.maps.len(), 1);
        assert_eq!(maps.maps[0].values[0], 3.0);
        assert_eq!(maps.maps[0].values[1], 3.0);
        assert_eq!(maps.maps[0].values[2], 3.0);
    }

    #[test]
    fn ignores_out_of_bounds_cell_indices() {
        let exchange = ExchangeResult {
            segment_exchanges: vec![SegmentExchange {
                segment_id: 4,
                species: SpeciesKind::Oxygen,
                source_amount_per_s: 9.0,
            }],
        };
        let mut segment_cells = HashMap::new();
        segment_cells.insert(4, vec![0, 100]);

        let maps = build_boundary_source_maps(2, 2, &exchange, &segment_cells);

        assert_eq!(maps.maps[0].values[0], 4.5);
        assert_eq!(maps.maps[0].values.iter().sum::<f32>(), 4.5);
    }
}
