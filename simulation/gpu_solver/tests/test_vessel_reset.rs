// tests/test_vessel_reset.rs
use gpu_solver::{explicit_step, Grid2D, StepInput};

#[test]
fn vessel_cells_reset_to_fixed_concentration() {
    let grid = Grid2D::new(3, 3, 1.0, 1.0);
    let center = grid.idx(1, 1);

    let mut vessel_mask = vec![false; grid.len()];
    vessel_mask[center] = true;

    let mut vessel_concentration = vec![0.0; grid.len()];
    vessel_concentration[center] = 5.0;

    let input = StepInput {
        grid,
        concentration: vec![1.0; grid.len()],
        diffusivity: vec![1.0; grid.len()],
        vmax: vec![0.0; grid.len()],
        km: vec![1.0; grid.len()],
        vessel_mask,
        vessel_concentration,
        dt: 0.1,
        reset_vessels: true,
    };

    let output = explicit_step(&input);

    assert!((output[center] - 5.0).abs() < 1e-6);
}
