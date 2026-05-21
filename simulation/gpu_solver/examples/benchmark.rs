use gpu_solver::gpu::WgpuGasExchangeSolver;
use gpu_solver::{explicit_gas_exchange_step, GasExchangeStepInput, Grid2D};
use std::time::Instant;

fn main() {
    let width = 250;
    let height = 250;
    let steps = 1000;

    let grid = Grid2D::new(width, height, 1.0, 1.0);
    let n = grid.len();

    let mut oxygen = vec![1.0_f32; n];
    let mut carbon_dioxide = vec![0.0_f32; n];
    let diffusivity_o2 = vec![1.0; n];
    let diffusivity_co2 = vec![0.8; n];
    let vmax = vec![0.01; n];
    let km = vec![1.0; n];
    let vessel_mask = vec![false; n];
    let vessel_o2 = vec![0.0; n];
    let vessel_co2 = vec![0.0; n];
    let dt = 0.1;
    let co2_yield = 1.0;

    let start = Instant::now();
    for _ in 0..steps {
        let mut oxygen_next = vec![0.0; n];
        let mut carbon_dioxide_next = vec![0.0; n];
        (oxygen, carbon_dioxide) = explicit_gas_exchange_step(GasExchangeStepInput {
            grid,
            oxygen_current: &oxygen,
            carbon_dioxide_current: &carbon_dioxide,
            oxygen_next: &mut oxygen_next,
            carbon_dioxide_next: &mut carbon_dioxide_next,
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
    }
    let cpu_elapsed = start.elapsed();

    let solver = pollster::block_on(WgpuGasExchangeSolver::new(grid));
    let gpu_start = Instant::now();
    let (gpu_o2, gpu_co2) = pollster::block_on(solver.run_gas_exchange_steps(
        &vec![1.0; n],
        &vec![0.0; n],
        &diffusivity_o2,
        &diffusivity_co2,
        &vmax,
        &km,
        &vessel_mask,
        &vessel_o2,
        &vessel_co2,
        dt,
        co2_yield,
        steps,
    ));
    let gpu_elapsed = gpu_start.elapsed();

    println!("CPU: {:?}\nGPU: {:?}", cpu_elapsed, gpu_elapsed);
    println!(
        "Center O2 diff: {:.6e}",
        (oxygen[n / 2] - gpu_o2[n / 2]).abs()
    );
    println!(
        "Center CO2 diff: {:.6e}",
        (carbon_dioxide[n / 2] - gpu_co2[n / 2]).abs()
    );
}
