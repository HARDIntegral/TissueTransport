use crate::types::Grid2D;

/// Borrowed arrays and scalar parameters needed for one coupled O₂/CO₂ timestep.
///
/// This struct mirrors the coupled Python reference case used to validate gas
/// exchange behavior. Oxygen diffuses, is consumed through Michaelis-Menten
/// metabolism, and is reset high in vessel cells. Carbon dioxide diffuses, is
/// produced from the oxygen consumption rate, and is reset low in vessel cells.
///
/// Parameters:
/// grid                                 -> Two-dimensional grid metadata.
/// oxygen_current                       -> Oxygen concentration at the start of the timestep.
/// carbon_dioxide_current               -> Carbon dioxide concentration at the start of the timestep.
/// oxygen_next                          -> Output buffer for updated oxygen concentration.
/// carbon_dioxide_next                  -> Output buffer for updated carbon dioxide concentration.
/// oxygen_diffusivity                   -> Effective oxygen diffusivity at each grid cell.
/// carbon_dioxide_diffusivity           -> Effective carbon dioxide diffusivity at each grid cell.
/// vmax                                 -> Michaelis-Menten maximum oxygen consumption rate.
/// km                                   -> Michaelis-Menten half-saturation constant.
/// vessel_mask                          -> Boolean mask marking vessel/source/sink cells.
/// vessel_oxygen_concentration          -> Fixed oxygen concentration imposed on vessel cells.
/// vessel_carbon_dioxide_concentration  -> Fixed carbon dioxide concentration imposed on vessel cells.
/// dt                                   -> Timestep size.
/// co2_yield                            -> CO₂ produced per unit O₂ consumed.
/// reset_vessels                        -> Whether vessel cells are reset after the update.
pub struct GasExchangeStepInput<'a> {
    pub grid: Grid2D,
    pub oxygen_current: &'a [f32],
    pub carbon_dioxide_current: &'a [f32],
    pub oxygen_next: &'a mut [f32],
    pub carbon_dioxide_next: &'a mut [f32],
    pub oxygen_diffusivity: &'a [f32],
    pub carbon_dioxide_diffusivity: &'a [f32],
    pub vmax: &'a [f32],
    pub km: &'a [f32],
    pub vessel_mask: &'a [bool],
    pub vessel_oxygen_concentration: &'a [f32],
    pub vessel_carbon_dioxide_concentration: &'a [f32],
    pub dt: f32,
    pub co2_yield: f32,
    pub reset_vessels: bool,
}

impl GasExchangeStepInput<'_> {
    fn validate(&self) {
        let n = self.grid.len();

        assert_eq!(self.oxygen_current.len(), n);
        assert_eq!(self.carbon_dioxide_current.len(), n);
        assert_eq!(self.oxygen_next.len(), n);
        assert_eq!(self.carbon_dioxide_next.len(), n);
        assert_eq!(self.oxygen_diffusivity.len(), n);
        assert_eq!(self.carbon_dioxide_diffusivity.len(), n);
        assert_eq!(self.vmax.len(), n);
        assert_eq!(self.km.len(), n);
        assert_eq!(self.vessel_mask.len(), n);
        assert_eq!(self.vessel_oxygen_concentration.len(), n);
        assert_eq!(self.vessel_carbon_dioxide_concentration.len(), n);
    }
}
/// Advance coupled oxygen and carbon dioxide fields by one CPU reference timestep.
///
/// This helper allocates fresh output buffers, delegates to the fused O₂/CO₂
/// stencil, and returns the updated fields. It is used by tests and validation
/// code as the readable one-step reference for the coupled gas-exchange model.
///
/// Parameters:
/// input -> Complete borrowed gas-exchange timestep input.
///
/// Returns:
/// `(oxygen_next, carbon_dioxide_next)` after one timestep.
pub fn explicit_gas_exchange_step(input: GasExchangeStepInput<'_>) -> (Vec<f32>, Vec<f32>) {
    input.validate();

    let n = input.grid.len();
    let mut oxygen_next = vec![0.0; n];
    let mut carbon_dioxide_next = vec![0.0; n];
    explicit_gas_exchange_step_fused(GasExchangeStepInput {
        grid: input.grid,
        oxygen_current: input.oxygen_current,
        carbon_dioxide_current: input.carbon_dioxide_current,
        oxygen_next: &mut oxygen_next,
        carbon_dioxide_next: &mut carbon_dioxide_next,
        oxygen_diffusivity: input.oxygen_diffusivity,
        carbon_dioxide_diffusivity: input.carbon_dioxide_diffusivity,
        vmax: input.vmax,
        km: input.km,
        vessel_mask: input.vessel_mask,
        vessel_oxygen_concentration: input.vessel_oxygen_concentration,
        vessel_carbon_dioxide_concentration: input.vessel_carbon_dioxide_concentration,
        dt: input.dt,
        co2_yield: input.co2_yield,
        reset_vessels: input.reset_vessels,
    });

    (oxygen_next, carbon_dioxide_next)
}

/// Advance one timestep using a fused coupled O₂/CO₂ gas-exchange stencil.
///
/// The oxygen field controls the Michaelis-Menten metabolic consumption term.
/// That same oxygen consumption term is subtracted from oxygen and added to
/// carbon dioxide using `co2_yield`.
///
/// The implementation keeps the hot interior loop separate from boundary handling
/// so most cells avoid per-neighbor bounds checks.
pub fn explicit_gas_exchange_step_fused(mut input: GasExchangeStepInput<'_>) {
    input.validate();

    let grid = input.grid;
    let width = grid.width;
    let height = grid.height;
    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);
    let dt = input.dt;
    let co2_yield = input.co2_yield;

    let oxygen_current = input.oxygen_current;
    let carbon_dioxide_current = input.carbon_dioxide_current;
    let oxygen_diffusivity = input.oxygen_diffusivity;
    let carbon_dioxide_diffusivity = input.carbon_dioxide_diffusivity;
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

                let oxygen_center = oxygen_current[idx];
                let carbon_dioxide_center = carbon_dioxide_current[idx];

                let oxygen_diffusion = diffusion_stencil(
                    oxygen_center,
                    oxygen_current[left_idx],
                    oxygen_current[right_idx],
                    oxygen_current[up_idx],
                    oxygen_current[down_idx],
                    oxygen_diffusivity[idx],
                    oxygen_diffusivity[left_idx],
                    oxygen_diffusivity[right_idx],
                    oxygen_diffusivity[up_idx],
                    oxygen_diffusivity[down_idx],
                    inv_dx2,
                    inv_dy2,
                );

                let carbon_dioxide_diffusion = diffusion_stencil(
                    carbon_dioxide_center,
                    carbon_dioxide_current[left_idx],
                    carbon_dioxide_current[right_idx],
                    carbon_dioxide_current[up_idx],
                    carbon_dioxide_current[down_idx],
                    carbon_dioxide_diffusivity[idx],
                    carbon_dioxide_diffusivity[left_idx],
                    carbon_dioxide_diffusivity[right_idx],
                    carbon_dioxide_diffusivity[up_idx],
                    carbon_dioxide_diffusivity[down_idx],
                    inv_dx2,
                    inv_dy2,
                );

                let oxygen_consumption = michaelis_menten(oxygen_center, vmax[idx], km[idx]);

                input.oxygen_next[idx] =
                    oxygen_center + dt * (oxygen_diffusion - oxygen_consumption);
                input.carbon_dioxide_next[idx] = carbon_dioxide_center
                    + dt * (carbon_dioxide_diffusion + co2_yield * oxygen_consumption);
            }
        }
    }

    for col in 0..width {
        update_gas_exchange_boundary_cell(&mut input, 0, col, inv_dx2, inv_dy2);

        if height > 1 {
            update_gas_exchange_boundary_cell(&mut input, height - 1, col, inv_dx2, inv_dy2);
        }
    }

    if height > 2 {
        for row in 1..(height - 1) {
            update_gas_exchange_boundary_cell(&mut input, row, 0, inv_dx2, inv_dy2);

            if width > 1 {
                update_gas_exchange_boundary_cell(&mut input, row, width - 1, inv_dx2, inv_dy2);
            }
        }
    }

    if input.reset_vessels {
        for idx in 0..grid.len() {
            if input.vessel_mask[idx] {
                input.oxygen_next[idx] = input.vessel_oxygen_concentration[idx];
                input.carbon_dioxide_next[idx] = input.vessel_carbon_dioxide_concentration[idx];
            }
        }
    }
}
/// Compute one face-averaged diffusion stencil for an interior cell.
#[inline]
fn diffusion_stencil(
    center: f32,
    left: f32,
    right: f32,
    up: f32,
    down: f32,
    d_center: f32,
    d_left_cell: f32,
    d_right_cell: f32,
    d_up_cell: f32,
    d_down_cell: f32,
    inv_dx2: f32,
    inv_dy2: f32,
) -> f32 {
    let d_left = 0.5 * (d_center + d_left_cell);
    let d_right = 0.5 * (d_center + d_right_cell);
    let d_up = 0.5 * (d_center + d_up_cell);
    let d_down = 0.5 * (d_center + d_down_cell);

    d_left * (left - center) * inv_dx2
        + d_right * (right - center) * inv_dx2
        + d_up * (up - center) * inv_dy2
        + d_down * (down - center) * inv_dy2
}
/// Update one boundary cell for the coupled O₂/CO₂ gas-exchange model.
#[inline]
fn update_gas_exchange_boundary_cell(
    input: &mut GasExchangeStepInput<'_>,
    row: usize,
    col: usize,
    inv_dx2: f32,
    inv_dy2: f32,
) {
    let grid = input.grid;
    let width = grid.width;
    let height = grid.height;
    let idx = grid.idx(row, col);

    let oxygen_center = input.oxygen_current[idx];
    let carbon_dioxide_center = input.carbon_dioxide_current[idx];
    let oxygen_d_center = input.oxygen_diffusivity[idx];
    let carbon_dioxide_d_center = input.carbon_dioxide_diffusivity[idx];

    let mut oxygen_diffusion = 0.0;
    let mut carbon_dioxide_diffusion = 0.0;

    if col > 0 {
        let left_idx = idx - 1;

        let oxygen_d_left = 0.5 * (oxygen_d_center + input.oxygen_diffusivity[left_idx]);
        oxygen_diffusion +=
            oxygen_d_left * (input.oxygen_current[left_idx] - oxygen_center) * inv_dx2;

        let carbon_dioxide_d_left =
            0.5 * (carbon_dioxide_d_center + input.carbon_dioxide_diffusivity[left_idx]);
        carbon_dioxide_diffusion += carbon_dioxide_d_left
            * (input.carbon_dioxide_current[left_idx] - carbon_dioxide_center)
            * inv_dx2;
    }

    if col + 1 < width {
        let right_idx = idx + 1;

        let oxygen_d_right = 0.5 * (oxygen_d_center + input.oxygen_diffusivity[right_idx]);
        oxygen_diffusion +=
            oxygen_d_right * (input.oxygen_current[right_idx] - oxygen_center) * inv_dx2;

        let carbon_dioxide_d_right =
            0.5 * (carbon_dioxide_d_center + input.carbon_dioxide_diffusivity[right_idx]);
        carbon_dioxide_diffusion += carbon_dioxide_d_right
            * (input.carbon_dioxide_current[right_idx] - carbon_dioxide_center)
            * inv_dx2;
    }

    if row > 0 {
        let up_idx = idx - width;

        let oxygen_d_up = 0.5 * (oxygen_d_center + input.oxygen_diffusivity[up_idx]);
        oxygen_diffusion += oxygen_d_up * (input.oxygen_current[up_idx] - oxygen_center) * inv_dy2;

        let carbon_dioxide_d_up =
            0.5 * (carbon_dioxide_d_center + input.carbon_dioxide_diffusivity[up_idx]);
        carbon_dioxide_diffusion += carbon_dioxide_d_up
            * (input.carbon_dioxide_current[up_idx] - carbon_dioxide_center)
            * inv_dy2;
    }

    if row + 1 < height {
        let down_idx = idx + width;

        let oxygen_d_down = 0.5 * (oxygen_d_center + input.oxygen_diffusivity[down_idx]);
        oxygen_diffusion +=
            oxygen_d_down * (input.oxygen_current[down_idx] - oxygen_center) * inv_dy2;

        let carbon_dioxide_d_down =
            0.5 * (carbon_dioxide_d_center + input.carbon_dioxide_diffusivity[down_idx]);
        carbon_dioxide_diffusion += carbon_dioxide_d_down
            * (input.carbon_dioxide_current[down_idx] - carbon_dioxide_center)
            * inv_dy2;
    }

    let oxygen_consumption = michaelis_menten(oxygen_center, input.vmax[idx], input.km[idx]);

    input.oxygen_next[idx] = oxygen_center + input.dt * (oxygen_diffusion - oxygen_consumption);
    input.carbon_dioxide_next[idx] = carbon_dioxide_center
        + input.dt * (carbon_dioxide_diffusion + input.co2_yield * oxygen_consumption);
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

// ----------------------------------------------------------------------------
// Inline Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gas_exchange_step_consumes_oxygen_and_produces_carbon_dioxide() {
        let grid = Grid2D::new(3, 3, 1.0, 1.0);
        let oxygen = vec![5.0; grid.len()];
        let carbon_dioxide = vec![1.0; grid.len()];
        let diffusivity = vec![0.0; grid.len()];
        let vmax = vec![0.5; grid.len()];
        let km = vec![1.0; grid.len()];
        let vessel_mask = vec![false; grid.len()];
        let vessel_oxygen = vec![10.0; grid.len()];
        let vessel_carbon_dioxide = vec![0.0; grid.len()];
        let mut oxygen_next = vec![0.0; grid.len()];
        let mut carbon_dioxide_next = vec![0.0; grid.len()];

        explicit_gas_exchange_step_fused(GasExchangeStepInput {
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

        let expected_consumption = 0.5 * 5.0 / (1.0 + 5.0);
        let expected_oxygen = 5.0 - 0.1 * expected_consumption;
        let expected_carbon_dioxide = 1.0 + 0.1 * expected_consumption;

        for idx in 0..grid.len() {
            assert!((oxygen_next[idx] - expected_oxygen).abs() < 1e-6);
            assert!((carbon_dioxide_next[idx] - expected_carbon_dioxide).abs() < 1e-6);
        }
    }
}
