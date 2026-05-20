pub mod gpu;
pub mod timestep;
pub mod types;

#[cfg(feature = "python")]
mod python;

pub use timestep::{explicit_step, run_steps_cpu};
pub use types::{Grid2D, StepInput};

/// Run reaction-diffusion timesteps using the GPU when available.
///
/// If WGPU cannot find or initialize a compatible device, this falls back to the
/// fused CPU reference solver.
pub async fn run_steps_auto(input: &StepInput, steps: usize) -> Vec<f32> {
	match gpu::WgpuSolver::try_new(input.grid).await {
		Ok(solver) => {
			solver
				.run_steps(
					&input.concentration,
					&input.diffusivity,
					&input.vmax,
					&input.km,
					&input.vessel_mask,
					&input.vessel_concentration,
					input.dt,
					input.reset_vessels,
					steps,
				)
				.await
		}
		Err(_) => run_steps_cpu(input, steps),
	}
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn gpu_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
	python::register_python_functions(m)
}
