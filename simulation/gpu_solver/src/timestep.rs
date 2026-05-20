use crate::types::{Grid2D, StepInput};

/// Borrowed arrays and scalar parameters needed for one fused timestep.
///
/// This struct groups the concentration field, transport properties, reaction
/// parameters, vessel source mask, and output buffer without taking ownership of
/// the data. The CPU reference path uses this to avoid allocating temporary arrays
/// inside every timestep, and the layout mirrors the same data that is passed into
/// the WGPU compute shader.
///
/// Parameters:
/// grid                  -> Two-dimensional grid metadata, including shape and spacing.
/// concentration_current -> Concentration field at the start of the timestep.
/// concentration_next    -> Output buffer that receives the updated concentration field.
/// diffusivity           -> Effective diffusivity at each grid cell.
/// vmax                  -> Michaelis-Menten maximum consumption rate at each cell.
/// km                    -> Michaelis-Menten half-saturation constant at each cell.
/// vessel_mask           -> Boolean mask marking cells that act as vessel/source cells.
/// vessel_concentration  -> Fixed concentration imposed on vessel cells when enabled.
/// dt                    -> Timestep size.
/// reset_vessels         -> Whether vessel cells are reset after the diffusion/reaction update.
///
/// Notes:
/// The timestep is explicit, meaning every cell is updated from the old
/// concentration field and written into a separate output buffer. This avoids
/// mixing old and new values during the same update.
pub struct FusedStepInput<'a> {
    pub grid: Grid2D,
    pub concentration_current: &'a [f32],
    pub concentration_next: &'a mut [f32],
    pub diffusivity: &'a [f32],
    pub vmax: &'a [f32],
    pub km: &'a [f32],
    pub vessel_mask: &'a [bool],
    pub vessel_concentration: &'a [f32],
    pub dt: f32,
    pub reset_vessels: bool,
}

impl FusedStepInput<'_> {
    fn validate(&self) {
        let n = self.grid.len();

        assert_eq!(self.concentration_current.len(), n);
        assert_eq!(self.concentration_next.len(), n);
        assert_eq!(self.diffusivity.len(), n);
        assert_eq!(self.vmax.len(), n);
        assert_eq!(self.km.len(), n);
        assert_eq!(self.vessel_mask.len(), n);
        assert_eq!(self.vessel_concentration.len(), n);
    }
}

/// Advance the concentration field by one explicit reaction-diffusion timestep.
///
/// This is the simple one-step CPU reference API. It delegates to `run_steps_cpu`
/// with `steps = 1`, so the one-step and multi-step CPU paths use the same fused
/// stencil implementation.
///
/// Parameters:
/// input -> Complete timestep input containing grid, concentration, diffusivity,
///          reaction terms, vessel source terms, and timestep size.
///
/// Returns:
/// A new concentration vector after one timestep.
pub fn explicit_step(input: &StepInput) -> Vec<f32> {
    run_steps_cpu(input, 1)
}

/// Run multiple CPU reaction-diffusion timesteps using the fused reference path.
///
/// This is the CPU fallback path when WGPU is unavailable, and it is also the
/// reference implementation used for checking GPU correctness. The function keeps
/// two concentration buffers and swaps them after each timestep, so it avoids
/// allocating a new concentration array every iteration.
///
/// Parameters:
/// input -> Complete reaction-diffusion input state.
/// steps -> Number of explicit timesteps to run.
///
/// Returns:
/// Concentration field after `steps` timesteps.
///
/// Notes:
/// The physical time simulated by this function is `steps * input.dt`. If the grid
/// spacing is reduced, the timestep may also need to be reduced because explicit
/// diffusion methods are limited by a stability condition proportional to `dx^2 / D`.
pub fn run_steps_cpu(input: &StepInput, steps: usize) -> Vec<f32> {
    input.validate();

    if steps == 0 {
        return input.concentration.clone();
    }

    let mut concentration_current = input.concentration.clone();
    let mut concentration_next = vec![0.0; input.grid.len()];
    let mut workspace = SolverWorkspace::new(input);

    for _ in 0..steps {
        workspace.explicit_step_fused(FusedStepInput {
            grid: input.grid,
            concentration_current: &concentration_current,
            concentration_next: &mut concentration_next,
            diffusivity: &input.diffusivity,
            vmax: &input.vmax,
            km: &input.km,
            vessel_mask: &input.vessel_mask,
            vessel_concentration: &input.vessel_concentration,
            dt: input.dt,
            reset_vessels: input.reset_vessels,
        });

        std::mem::swap(&mut concentration_current, &mut concentration_next);
    }

    concentration_current
}

/// Workspace object for CPU fused timestep updates.
///
/// The current CPU implementation does not need to store intermediate arrays inside
/// the workspace, because the caller provides both the current and next concentration
/// buffers. The type still exists as the CPU solver entry point so the structure of
/// the CPU path stays parallel to the GPU path, where a solver object owns reusable
/// buffers, pipeline state, and dispatch logic.
///
/// Notes:
/// Keeping this as a workspace instead of a free function makes it easier to add
/// future CPU-side reusable scratch buffers without changing the public call pattern.
pub struct SolverWorkspace;

impl SolverWorkspace {
    /// Create a CPU solver workspace for a fixed problem shape.
    ///
    /// The input is validated here so shape mismatches are caught before timestep
    /// execution begins.
    pub fn new(input: &StepInput) -> Self {
        input.validate();
        Self
    }

    /// Advance one timestep using a fused finite-difference reaction-diffusion stencil.
    ///
    /// The update combines spatial diffusion, Michaelis-Menten consumption, and optional
    /// vessel concentration reset in one pass. Interior cells are handled separately from
    /// boundary cells so the hot inner loop does not need boundary checks.
    ///
    /// Parameters:
    /// input -> Borrowed timestep buffers and scalar parameters. The current concentration
    ///          is read from `concentration_current`, and the result is written into
    ///          `concentration_next`.
    ///
    /// Notes:
    /// Diffusion is computed with a finite-difference stencil using face-averaged
    /// diffusivity:
    ///
    /// D_left  = 0.5 * (D_center + D_left_cell)
    /// D_right = 0.5 * (D_center + D_right_cell)
    /// D_up    = 0.5 * (D_center + D_up_cell)
    /// D_down  = 0.5 * (D_center + D_down_cell)
    ///
    /// The reaction term follows Michaelis-Menten consumption:
    ///
    /// consumption = vmax * C / (km + C)
    ///
    /// The final explicit update is:
    ///
    /// C_next = C_current + dt * (diffusion - consumption)
    ///
    /// If `reset_vessels` is enabled, cells marked in `vessel_mask` are overwritten
    /// after the update with their prescribed `vessel_concentration` values. This treats
    /// vessel cells as fixed concentration source cells.
    pub fn explicit_step_fused(&mut self, mut input: FusedStepInput<'_>) {
        input.validate();

        let grid = input.grid;
        let width = grid.width;
        let height = grid.height;
        let inv_dx2 = 1.0 / (grid.dx * grid.dx);
        let inv_dy2 = 1.0 / (grid.dy * grid.dy);
        let dt = input.dt;
        let concentration_current = input.concentration_current;
        let diffusivity = input.diffusivity;
        let vmax = input.vmax;
        let km = input.km;

        if width == 0 || height == 0 {
            return;
        }

        if width > 2 && height > 2 {
            for row in 1..(height - 1) {
                let row_offset = row * width;
                let up_offset = (row - 1) * width;
                let down_offset = (row + 1) * width;

                for col in 1..(width - 1) {
                    let idx = row_offset + col;
                    let left_idx = idx - 1;
                    let right_idx = idx + 1;
                    let up_idx = up_offset + col;
                    let down_idx = down_offset + col;

                    let center = concentration_current[idx];
                    let d_center = diffusivity[idx];

                    let d_left = 0.5 * (d_center + diffusivity[left_idx]);
                    let d_right = 0.5 * (d_center + diffusivity[right_idx]);
                    let d_up = 0.5 * (d_center + diffusivity[up_idx]);
                    let d_down = 0.5 * (d_center + diffusivity[down_idx]);

                    let diffusion = d_left * (concentration_current[left_idx] - center) * inv_dx2
                        + d_right * (concentration_current[right_idx] - center) * inv_dx2
                        + d_up * (concentration_current[up_idx] - center) * inv_dy2
                        + d_down * (concentration_current[down_idx] - center) * inv_dy2;

                    let consumption = michaelis_menten(center, vmax[idx], km[idx]);
                    input.concentration_next[idx] = center + dt * (diffusion - consumption);
                }
            }
        }

        // Top and bottom rows.
        for col in 0..width {
            update_boundary_cell(&mut input, 0, col, inv_dx2, inv_dy2);

            if height > 1 {
                update_boundary_cell(&mut input, height - 1, col, inv_dx2, inv_dy2);
            }
        }

        // Left and right columns, excluding corners already handled above.
        if height > 2 {
            for row in 1..(height - 1) {
                update_boundary_cell(&mut input, row, 0, inv_dx2, inv_dy2);

                if width > 1 {
                    update_boundary_cell(&mut input, row, width - 1, inv_dx2, inv_dy2);
                }
            }
        }

        if input.reset_vessels {
            for idx in 0..grid.len() {
                if input.vessel_mask[idx] {
                    input.concentration_next[idx] = input.vessel_concentration[idx];
                }
            }
        }
    }
}

/// Compute Michaelis-Menten oxygen consumption for one cell.
///
/// The zero-denominator guard prevents invalid consumption values if both `km` and
/// concentration are zero.
#[inline]
fn michaelis_menten(concentration: f32, vmax: f32, km: f32) -> f32 {
    if km + concentration > 0.0 {
        vmax * concentration / (km + concentration)
    } else {
        0.0
    }
}

/// Update a single boundary cell using only neighbors that exist inside the grid.
///
/// Boundary cells cannot use the branch-free interior stencil because one or more
/// neighbors may lie outside the domain. This helper applies the same diffusion and
/// consumption model as the interior loop, but skips missing neighbors instead of
/// assuming ghost cells.
#[inline]
fn update_boundary_cell(
    input: &mut FusedStepInput<'_>,
    row: usize,
    col: usize,
    inv_dx2: f32,
    inv_dy2: f32,
) {
    let grid = input.grid;
    let width = grid.width;
    let height = grid.height;
    let idx = grid.idx(row, col);

    let center = input.concentration_current[idx];
    let d_center = input.diffusivity[idx];
    let mut diffusion = 0.0;

    if col > 0 {
        let left_idx = idx - 1;
        let d_left = 0.5 * (d_center + input.diffusivity[left_idx]);
        diffusion += d_left * (input.concentration_current[left_idx] - center) * inv_dx2;
    }

    if col + 1 < width {
        let right_idx = idx + 1;
        let d_right = 0.5 * (d_center + input.diffusivity[right_idx]);
        diffusion += d_right * (input.concentration_current[right_idx] - center) * inv_dx2;
    }

    if row > 0 {
        let up_idx = idx - width;
        let d_up = 0.5 * (d_center + input.diffusivity[up_idx]);
        diffusion += d_up * (input.concentration_current[up_idx] - center) * inv_dy2;
    }

    if row + 1 < height {
        let down_idx = idx + width;
        let d_down = 0.5 * (d_center + input.diffusivity[down_idx]);
        diffusion += d_down * (input.concentration_current[down_idx] - center) * inv_dy2;
    }

    let consumption = michaelis_menten(center, input.vmax[idx], input.km[idx]);
    input.concentration_next[idx] = center + input.dt * (diffusion - consumption);
}

// ----------------------------------------------------------------------------
// Inline Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_input() -> StepInput {
        let grid = Grid2D::new(3, 3, 1.0, 1.0);

        StepInput {
            grid,
            concentration: vec![5.0, 5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 5.0],
            diffusivity: vec![1.0; grid.len()],
            vmax: vec![0.5; grid.len()],
            km: vec![1.0; grid.len()],
            vessel_mask: vec![false, false, false, false, true, false, false, false, false],
            vessel_concentration: vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            dt: 0.1,
            reset_vessels: true,
        }
    }

    #[test]
    fn fused_buffered_step_matches_explicit_step() {
        let input = reference_input();
        let mut workspace = SolverWorkspace::new(&input);
        let mut concentration_next = vec![0.0; input.grid.len()];

        workspace.explicit_step_fused(FusedStepInput {
            grid: input.grid,
            concentration_current: &input.concentration,
            concentration_next: &mut concentration_next,
            diffusivity: &input.diffusivity,
            vmax: &input.vmax,
            km: &input.km,
            vessel_mask: &input.vessel_mask,
            vessel_concentration: &input.vessel_concentration,
            dt: input.dt,
            reset_vessels: input.reset_vessels,
        });

        let output = explicit_step(&input);

        for (direct, fused) in output.iter().zip(concentration_next.iter()) {
            assert!((direct - fused).abs() < 1e-6);
        }
    }
}
