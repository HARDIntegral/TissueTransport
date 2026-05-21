use gpu_solver::gpu::WgpuGasExchangeSolver;
use gpu_solver::{explicit_gas_exchange_step, GasExchangeStepInput, Grid2D};

#[test]
fn gpu_gas_exchange_solver_initializes() {
	pollster::block_on(async {
		let grid = Grid2D {
			width: 3,
			height: 3,
			dx: 1.0,
			dy: 1.0,
		};

		let solver = WgpuGasExchangeSolver::new(grid).await;

		assert_eq!(solver.grid().width, grid.width);
		assert_eq!(solver.grid().height, grid.height);
		assert_eq!(solver.grid().dx, grid.dx);
		assert_eq!(solver.grid().dy, grid.dy);
	});
}

#[test]
fn gpu_gas_exchange_matches_cpu_reference() {
	pollster::block_on(async {
		let grid = Grid2D {
			width: 3,
			height: 3,
			dx: 1.0,
			dy: 1.0,
		};

		let oxygen = vec![
			5.0, 5.0, 5.0,
			5.0, 1.0, 5.0,
			5.0, 5.0, 5.0,
		];
		let carbon_dioxide = vec![1.0; 9];
		let diffusivity_o2 = vec![1.0; 9];
		let diffusivity_co2 = vec![0.8; 9];
		let vmax = vec![0.5; 9];
		let km = vec![1.0; 9];
		let vessel_mask = vec![false, false, false, false, true, false, false, false, false];
		let vessel_o2 = vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0];
		let vessel_co2 = vec![0.0; 9];
		let dt = 0.1;
		let co2_yield = 1.0;

		let mut cpu_oxygen_next = vec![0.0; grid.len()];
		let mut cpu_carbon_dioxide_next = vec![0.0; grid.len()];
		let (cpu_oxygen, cpu_carbon_dioxide) = explicit_gas_exchange_step(GasExchangeStepInput {
			grid,
			oxygen_current: &oxygen,
			carbon_dioxide_current: &carbon_dioxide,
			oxygen_next: &mut cpu_oxygen_next,
			carbon_dioxide_next: &mut cpu_carbon_dioxide_next,
			oxygen_diffusivity: &diffusivity_o2,
			carbon_dioxide_diffusivity: &diffusivity_co2,
			vmax: &vmax,
			km: &km,
			vessel_mask: &vessel_mask,
			vessel_oxygen_concentration: &vessel_o2,
			vessel_carbon_dioxide_concentration: &vessel_co2,
			dt,
			co2_yield,
			reset_vessels: true,
		});

		let solver = WgpuGasExchangeSolver::new(grid).await;
		let (gpu_oxygen, gpu_carbon_dioxide) = solver
			.run_gas_exchange_steps(
				&oxygen,
				&carbon_dioxide,
				&diffusivity_o2,
				&diffusivity_co2,
				&vmax,
				&km,
				&vessel_mask,
				&vessel_o2,
				&vessel_co2,
				dt,
				co2_yield,
				1,
			)
			.await;

		for (cpu, gpu) in cpu_oxygen.iter().zip(gpu_oxygen.iter()) {
			assert!((cpu - gpu).abs() < 1e-5, "CPU O2 {cpu} != GPU O2 {gpu}");
		}

		for (cpu, gpu) in cpu_carbon_dioxide.iter().zip(gpu_carbon_dioxide.iter()) {
			assert!((cpu - gpu).abs() < 1e-5, "CPU CO2 {cpu} != GPU CO2 {gpu}");
		}
	});
}