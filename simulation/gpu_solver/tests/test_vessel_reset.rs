// tests/test_vessel_reset.rs
use gpu_solver::{explicit_gas_exchange_step, GasExchangeStepInput, Grid2D};

#[test]
fn vessel_cells_reset_to_fixed_oxygen_and_carbon_dioxide_concentration() {
    let grid = Grid2D::new(3, 3, 1.0, 1.0);
    let center = grid.idx(1, 1);

    let mut vessel_mask = vec![false; grid.len()];
    vessel_mask[center] = true;

    let mut vessel_oxygen = vec![0.0; grid.len()];
    vessel_oxygen[center] = 5.0;

    let mut vessel_carbon_dioxide = vec![0.0; grid.len()];
    vessel_carbon_dioxide[center] = 0.0;

    let mut oxygen_next = vec![0.0; grid.len()];
    let mut carbon_dioxide_next = vec![0.0; grid.len()];

    let (oxygen_output, carbon_dioxide_output) = explicit_gas_exchange_step(GasExchangeStepInput {
        grid,
        oxygen_current: &vec![1.0; grid.len()],
        carbon_dioxide_current: &vec![2.0; grid.len()],
        oxygen_next: &mut oxygen_next,
        carbon_dioxide_next: &mut carbon_dioxide_next,
        oxygen_diffusivity: &vec![1.0; grid.len()],
        carbon_dioxide_diffusivity: &vec![1.0; grid.len()],
        vmax: &vec![0.0; grid.len()],
        km: &vec![1.0; grid.len()],
        vessel_mask: &vessel_mask,
        vessel_oxygen_concentration: &vessel_oxygen,
        vessel_carbon_dioxide_concentration: &vessel_carbon_dioxide,
        dt: 0.1,
        co2_yield: 1.0,
        reset_vessels: true,
    });

    assert!((oxygen_output[center] - 5.0).abs() < 1e-6);
    assert!(carbon_dioxide_output[center].abs() < 1e-6);
}
