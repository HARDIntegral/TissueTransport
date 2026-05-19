// tests/test_consumption.rs
use gpu_solver::{explicit_step, Grid2D, StepInput};

#[test]
fn michaelis_menten_consumption_lowers_concentration() {
    let grid = Grid2D::new(3, 3, 1.0, 1.0);

    let input = StepInput {
        grid,
        concentration: vec![1.0; grid.len()],
        diffusivity: vec![0.0; grid.len()],
        vmax: vec![0.5; grid.len()],
        km: vec![1.0; grid.len()],
        vessel_mask: vec![false; grid.len()],
        vessel_concentration: vec![0.0; grid.len()],
        dt: 0.1,
        reset_vessels: true,
    };

    let output = explicit_step(&input);

    assert!(output.iter().all(|v| *v < 1.0));
}
