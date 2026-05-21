use gpu_solver::{explicit_gas_exchange_step, GasExchangeStepInput, Grid2D};
use serde::Deserialize;

#[derive(Deserialize)]
struct ReferenceCase {
    width: usize,
    height: usize,
    dx: f32,
    dy: f32,
    dt: f32,
    oxygen_concentration: Vec<f32>,
    carbon_dioxide_concentration: Vec<f32>,

    diffusivity: Vec<f32>,
    vmax: Vec<f32>,
    km: Vec<f32>,

    vessel_mask: Vec<bool>,
    vessel_oxygen_concentration: Vec<f32>,
    vessel_carbon_dioxide_concentration: Vec<f32>,
    co2_yield: f32,
    expected_oxygen: Vec<f32>,
    expected_carbon_dioxide: Vec<f32>,
}

#[test]
fn rust_matches_python_gas_exchange_reference() {
    let raw = include_str!("../test_data/reference_case.json");

    let case: ReferenceCase = serde_json::from_str(raw).unwrap();

    let grid = Grid2D::new(case.width, case.height, case.dx, case.dy);
    let mut oxygen_next = vec![0.0; grid.len()];
    let mut carbon_dioxide_next = vec![0.0; grid.len()];

    let (oxygen_output, carbon_dioxide_output) = explicit_gas_exchange_step(GasExchangeStepInput {
        grid,
        oxygen_current: &case.oxygen_concentration,
        carbon_dioxide_current: &case.carbon_dioxide_concentration,
        oxygen_next: &mut oxygen_next,
        carbon_dioxide_next: &mut carbon_dioxide_next,
        oxygen_diffusivity: &case.diffusivity,
        carbon_dioxide_diffusivity: &case.diffusivity,
        vmax: &case.vmax,
        km: &case.km,
        vessel_mask: &case.vessel_mask,
        vessel_oxygen_concentration: &case.vessel_oxygen_concentration,
        vessel_carbon_dioxide_concentration: &case.vessel_carbon_dioxide_concentration,
        dt: case.dt,
        co2_yield: case.co2_yield,
        reset_vessels: true,
    });

    for (rust, py) in oxygen_output.iter().zip(case.expected_oxygen.iter()) {
        assert!(
            (rust - py).abs() < 1e-5,
            "O2 Rust: {}, Python: {}",
            rust,
            py
        );
    }

    for (rust, py) in carbon_dioxide_output
        .iter()
        .zip(case.expected_carbon_dioxide.iter())
    {
        assert!(
            (rust - py).abs() < 1e-5,
            "CO2 Rust: {}, Python: {}",
            rust,
            py
        );
    }
}
