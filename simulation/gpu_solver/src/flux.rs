use crate::types::Grid2D;

/// Compute horizontal and vertical diffusive fluxes on cell faces.
///
/// This is the Rust equivalent of the Python `compute_flux` function.
/// Arrays are flattened in row-major order.
pub fn compute_flux(
    grid: Grid2D,
    concentration: &[f32],
    diffusivity: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let width = grid.width;
    let height = grid.height;

    assert_eq!(concentration.len(), grid.len());
    assert_eq!(diffusivity.len(), grid.len());

    let mut jx = vec![0.0; height * (width - 1)];
    let mut jy = vec![0.0; (height - 1) * width];

    // Horizontal fluxes between (row, col) and (row, col + 1).
    for row in 0..height {
        for col in 0..(width - 1) {
            let left = grid.idx(row, col);
            let right = grid.idx(row, col + 1);
            let face = row * (width - 1) + col;

            let d_avg = 0.5 * (diffusivity[left] + diffusivity[right]);
            let dc_dx = (concentration[right] - concentration[left]) / grid.dx;

            jx[face] = -d_avg * dc_dx;
        }
    }

    // Vertical fluxes between (row, col) and (row + 1, col).
    for row in 0..(height - 1) {
        for col in 0..width {
            let top = grid.idx(row, col);
            let bottom = grid.idx(row + 1, col);
            let face = row * width + col;

            let d_avg = 0.5 * (diffusivity[top] + diffusivity[bottom]);
            let dc_dy = (concentration[bottom] - concentration[top]) / grid.dy;

            jy[face] = -d_avg * dc_dy;
        }
    }

    (jx, jy)
}

/// Convert face fluxes into per-cell concentration change from diffusion.
///
/// This is the Rust equivalent of the Python `compute_flux_divergence` function.
pub fn compute_flux_divergence(grid: Grid2D, jx: &[f32], jy: &[f32]) -> Vec<f32> {
    let width = grid.width;
    let height = grid.height;

    assert_eq!(jx.len(), height * (width - 1));
    assert_eq!(jy.len(), (height - 1) * width);

    let mut dcdt = vec![0.0; grid.len()];

    // Horizontal flux contribution.
    for row in 0..height {
        for col in 0..(width - 1) {
            let flux = jx[row * (width - 1) + col];
            let left = grid.idx(row, col);
            let right = grid.idx(row, col + 1);

            dcdt[left] -= flux / grid.dx;
            dcdt[right] += flux / grid.dx;
        }
    }

    // Vertical flux contribution.
    for row in 0..(height - 1) {
        for col in 0..width {
            let flux = jy[row * width + col];
            let top = grid.idx(row, col);
            let bottom = grid.idx(row + 1, col);

            dcdt[top] -= flux / grid.dy;
            dcdt[bottom] += flux / grid.dy;
        }
    }

    dcdt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_concentration_has_zero_flux() {
        let grid = Grid2D::new(5, 5, 1.0, 1.0);
        let concentration = vec![1.0; grid.len()];
        let diffusivity = vec![1.0; grid.len()];

        let (jx, jy) = compute_flux(grid, &concentration, &diffusivity);

        assert!(jx.iter().all(|value| value.abs() < 1e-6));
        assert!(jy.iter().all(|value| value.abs() < 1e-6));
    }
}
