// tests/test_timestep.rs
use gpu_solver::{explicit_gas_exchange_step, GasExchangeStepInput, Grid2D};

#[test]
fn uniform_concentration_without_consumption_stays_constant() {
    let grid = Grid2D::new(5, 5, 1.0, 1.0);

    let oxygen = vec![1.0; grid.len()];
    let carbon_dioxide = vec![0.0; grid.len()];
    let diffusivity = vec![1.0; grid.len()];
    let vmax = vec![0.0; grid.len()];
    let km = vec![1.0; grid.len()];
    let vessel_mask = vec![false; grid.len()];
    let vessel_oxygen = vec![0.0; grid.len()];
    let vessel_carbon_dioxide = vec![0.0; grid.len()];
    let mut oxygen_next = vec![0.0; grid.len()];
    let mut carbon_dioxide_next = vec![0.0; grid.len()];

    let (oxygen_output, carbon_dioxide_output) = explicit_gas_exchange_step(GasExchangeStepInput {
        grid,
        oxygen_current: &oxygen,
        carbon_dioxide_current: &carbon_dioxide,
        oxygen_next: &mut oxygen_next,
        carbon_dioxide_next: &mut carbon_dioxide_next,
        oxygen_diffusivity: &diffusivity,
        carbon_dioxide_diffusivity: &diffusivity,
        vmax: &vmax,
        km: &km,
        vessel_mask: &vessel_mask,
        vessel_oxygen_concentration: &vessel_oxygen,
        vessel_carbon_dioxide_concentration: &vessel_carbon_dioxide,
        dt: 0.1,
        co2_yield: 1.0,
        reset_vessels: false,
    });

    assert!(oxygen_output.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    assert!(carbon_dioxide_output.iter().all(|v| v.abs() < 1e-6));
}
