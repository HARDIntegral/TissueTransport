pub mod gpu;
pub mod timestep;
pub mod types;

#[cfg(feature = "python")]
mod python;

pub use timestep::{explicit_gas_exchange_step, GasExchangeStepInput};
pub use types::Grid2D;

/// Run coupled oxygen/carbon dioxide gas-exchange timesteps using the GPU when available.
///
/// If WGPU cannot find or initialize a compatible device, this falls back to the
/// fused CPU reference solver.
pub async fn run_gas_exchange_steps_auto(
	input: GasExchangeStepInput<'_>,
	steps: usize,
) -> (Vec<f32>, Vec<f32>) {
	match gpu::WgpuGasExchangeSolver::try_new(input.grid).await {
		Ok(solver) => {
			solver
				.run_gas_exchange_steps(
					input.oxygen_current,
					input.carbon_dioxide_current,
					input.oxygen_diffusivity,
					input.carbon_dioxide_diffusivity,
					input.vmax,
					input.km,
					input.vessel_mask,
					input.vessel_oxygen_concentration,
					input.vessel_carbon_dioxide_concentration,
					input.dt,
					input.co2_yield,
					steps,
				)
				.await
		}
		Err(_) => {
			let mut oxygen_current = input.oxygen_current.to_vec();
			let mut carbon_dioxide_current = input.carbon_dioxide_current.to_vec();

			for _ in 0..steps {
				let mut oxygen_next = vec![0.0; input.grid.len()];
				let mut carbon_dioxide_next = vec![0.0; input.grid.len()];

				let (next_oxygen, next_carbon_dioxide) = explicit_gas_exchange_step(
					GasExchangeStepInput {
						grid: input.grid,
						oxygen_current: &oxygen_current,
						carbon_dioxide_current: &carbon_dioxide_current,
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
					},
				);

				oxygen_current = next_oxygen;
				carbon_dioxide_current = next_carbon_dioxide;
			}

			(oxygen_current, carbon_dioxide_current)
		}
	}
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn gpu_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
	python::register_python_functions(m)
}
