// tests/test_flux.rs
use gpu_solver::flux::{compute_flux, compute_flux_divergence};
use gpu_solver::Grid2D;

#[test]
fn uniform_concentration_has_zero_flux_and_zero_divergence() {
    let grid = Grid2D::new(5, 5, 1.0, 1.0);
    let c = vec![1.0; grid.len()];
    let d = vec![1.0; grid.len()];

    let (jx, jy) = compute_flux(grid, &c, &d);
    let div = compute_flux_divergence(grid, &jx, &jy);

    assert!(jx.iter().all(|v| v.abs() < 1e-6));
    assert!(jy.iter().all(|v| v.abs() < 1e-6));
    assert!(div.iter().all(|v| v.abs() < 1e-6));
}
