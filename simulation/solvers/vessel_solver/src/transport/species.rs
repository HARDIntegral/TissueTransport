use crate::types::{SpeciesKind, SpeciesParameters};

// Build default transport parameters for oxygen.
//
// These are solver defaults, not final biological constants. They should be
// overridden from Python or a higher-level config once physical units are
// calibrated against the tissue diffusion grid.
pub fn default_oxygen_parameters() -> SpeciesParameters {
    SpeciesParameters {
        kind: SpeciesKind::Oxygen,
        inlet_concentration: 1.0,
        diffusivity_um2_per_s: 2_000.0,
        permeability_um_per_s: 1.0,
    }
}

// Build default transport parameters for carbon dioxide.
//
// CO2 uses the same generic blood transport machinery as oxygen. Differences
// between gases should live in parameter values and exchange/production rules,
// not duplicated graph traversal code.
pub fn default_carbon_dioxide_parameters() -> SpeciesParameters {
    SpeciesParameters {
        kind: SpeciesKind::CarbonDioxide,
        inlet_concentration: 0.25,
        diffusivity_um2_per_s: 2_500.0,
        permeability_um_per_s: 1.0,
    }
}

// Return the built-in species set used by the default vessel transport
// pipeline.
pub fn default_species_parameters() -> Vec<SpeciesParameters> {
    vec![
        default_oxygen_parameters(),
        default_carbon_dioxide_parameters(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oxygen_defaults_use_oxygen_kind() {
        let oxygen = default_oxygen_parameters();

        assert_eq!(oxygen.kind, SpeciesKind::Oxygen);
        assert!(oxygen.inlet_concentration > 0.0);
        assert!(oxygen.diffusivity_um2_per_s > 0.0);
        assert!(oxygen.permeability_um_per_s > 0.0);
    }

    #[test]
    fn carbon_dioxide_defaults_use_carbon_dioxide_kind() {
        let carbon_dioxide = default_carbon_dioxide_parameters();

        assert_eq!(carbon_dioxide.kind, SpeciesKind::CarbonDioxide);
        assert!(carbon_dioxide.diffusivity_um2_per_s > 0.0);
        assert!(carbon_dioxide.permeability_um_per_s > 0.0);
    }

    #[test]
    fn default_species_contains_oxygen_and_carbon_dioxide() {
        let species = default_species_parameters();

        assert_eq!(species.len(), 2);
        assert_eq!(species[0].kind, SpeciesKind::Oxygen);
        assert_eq!(species[1].kind, SpeciesKind::CarbonDioxide);
    }
}
