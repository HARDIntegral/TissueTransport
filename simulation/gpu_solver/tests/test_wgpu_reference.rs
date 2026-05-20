// tests/test_wgpu_reference.rs

use gpu_solver::gpu::WgpuSolver;
use gpu_solver::{explicit_step, Grid2D, StepInput};

#[test]
fn gpu_matches_cpu_timestep() {
	pollster::block_on(async {
		let grid = Grid2D {
			width: 3,
			height: 3,
			dx: 1.0,
			dy: 1.0,
		};

		let concentration = vec![
			5.0, 5.0, 5.0,
			5.0, 1.0, 5.0,
			5.0, 5.0, 5.0,
		];
		let diffusivity = vec![1.0; 9];
		let vmax = vec![0.5; 9];
		let km = vec![1.0; 9];
		let vessel_mask = vec![false, false, false, false, true, false, false, false, false];
		let vessel_concentration = vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0];
		let dt = 0.1;

		let cpu_next = explicit_step(&StepInput {
			grid,
			concentration: concentration.clone(),
			diffusivity: diffusivity.clone(),
			vmax: vmax.clone(),
			km: km.clone(),
			vessel_mask: vessel_mask.clone(),
			vessel_concentration: vessel_concentration.clone(),
			dt,
			reset_vessels: true,
		});

		let solver = WgpuSolver::new(grid).await;
		let gpu_next = solver.run_steps(
			&concentration,
			&diffusivity,
			&vmax,
			&km,
			&vessel_mask,
			&vessel_concentration,
			dt,
			true,
			1,
		).await;

		for (cpu, gpu) in cpu_next.iter().zip(gpu_next.iter()) {
			assert!(
				(cpu - gpu).abs() < 1e-5,
				"CPU {cpu} != GPU {gpu}"
			);
		}
	});
}