use gpu_solver::{explicit_step, Grid2D, StepInput};
use serde::Deserialize;

#[derive(Deserialize)]
struct ReferenceCase {
    width: usize,
    height: usize,
    dx: f32,
    dy: f32,
    dt: f32,

    concentration: Vec<f32>,
    diffusivity: Vec<f32>,
    vmax: Vec<f32>,
    km: Vec<f32>,

    vessel_mask: Vec<bool>,
    vessel_concentration: Vec<f32>,

    expected: Vec<f32>,
}

#[test]
fn rust_matches_python_reference() {
    let raw = include_str!("../test_data/reference_case.json");

    let case: ReferenceCase = serde_json::from_str(raw).unwrap();

    let grid = Grid2D::new(case.width, case.height, case.dx, case.dy);

    let input = StepInput {
        grid,
        concentration: case.concentration,
        diffusivity: case.diffusivity,
        vmax: case.vmax,
        km: case.km,
        vessel_mask: case.vessel_mask,
        vessel_concentration: case.vessel_concentration,
        dt: case.dt,
        reset_vessels: true,
    };

    let output = explicit_step(&input);

    for (rust, py) in output.iter().zip(case.expected.iter()) {
        assert!((rust - py).abs() < 1e-5, "Rust: {}, Python: {}", rust, py);
    }
}
