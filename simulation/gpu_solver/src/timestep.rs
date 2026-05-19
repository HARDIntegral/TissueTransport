use crate::flux::{compute_flux, compute_flux_divergence};
use crate::types::StepInput;

/// Advance the concentration field by one explicit reaction-diffusion timestep.
///
/// This is the Rust CPU reference version of the Python `explicit_step` function.
/// The future WGPU solver should match this output before it is trusted.
pub fn explicit_step(input: &StepInput) -> Vec<f32> {
    input.validate();

    let (jx, jy) = compute_flux(input.grid, &input.concentration, &input.diffusivity);
    let diffusion = compute_flux_divergence(input.grid, &jx, &jy);

    let mut concentration_new = vec![0.0; input.grid.len()];

    for idx in 0..input.grid.len() {
        let c = input.concentration[idx];
        let vmax = input.vmax[idx];
        let km = input.km[idx];

        let consumption = if km + c > 0.0 {
            vmax * c / (km + c)
        } else {
            0.0
        };

        concentration_new[idx] = c + input.dt * (diffusion[idx] - consumption);
    }

    if input.reset_vessels {
        for idx in 0..input.grid.len() {
            if input.vessel_mask[idx] {
                concentration_new[idx] = input.vessel_concentration[idx];
            }
        }
    }

    concentration_new
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Grid2D;

    #[test]
    fn uniform_concentration_without_consumption_stays_constant() {
        let grid = Grid2D::new(5, 5, 1.0, 1.0);

        let input = StepInput {
            grid,
            concentration: vec![1.0; grid.len()],
            diffusivity: vec![1.0; grid.len()],
            vmax: vec![0.0; grid.len()],
            km: vec![1.0; grid.len()],
            vessel_mask: vec![false; grid.len()],
            vessel_concentration: vec![0.0; grid.len()],
            dt: 0.1,
            reset_vessels: true,
        };

        let output = explicit_step(&input);

        for value in output {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn michaelis_menten_consumption_decreases_concentration() {
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

        for value in output {
            assert!(value < 1.0);
        }
    }

    #[test]
    fn vessel_cells_reset_after_step() {
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
}
