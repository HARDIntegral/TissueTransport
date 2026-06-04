pub mod propagation;
pub mod species;

pub use propagation::{propagate_species, propagate_species_set};
pub use species::{
    default_carbon_dioxide_parameters, default_oxygen_parameters, default_species_parameters,
};
