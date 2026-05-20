use gpu_solver::gpu::WgpuSolver;
use gpu_solver::timestep::{FusedStepInput, SolverWorkspace};
use gpu_solver::{Grid2D, StepInput};
use std::time::Instant;

fn main() {
    let width = 250;
    let height = 250;
    let steps = 1000;

    let grid = Grid2D::new(width, height, 1.0, 1.0);
    let n = grid.len();

    let mut concentration = vec![1.0_f32; n];

    let input_template = StepInput {
        grid,
        concentration: concentration.clone(),
        diffusivity: vec![1.0; n],
        vmax: vec![0.01; n],
        km: vec![1.0; n],
        vessel_mask: vec![false; n],
        vessel_concentration: vec![0.0; n],
        dt: 0.1,
        reset_vessels: true,
    };

    let mut workspace = SolverWorkspace::new(&input_template);
    let mut concentration_next = vec![0.0_f32; n];

    let start = Instant::now();

    for _ in 0..steps {
        workspace.explicit_step_fused(FusedStepInput {
            grid,
            concentration_current: &concentration,
            concentration_next: &mut concentration_next,
            diffusivity: &input_template.diffusivity,
            vmax: &input_template.vmax,
            km: &input_template.km,
            vessel_mask: &input_template.vessel_mask,
            vessel_concentration: &input_template.vessel_concentration,
            dt: input_template.dt,
            reset_vessels: input_template.reset_vessels,
        });

        std::mem::swap(&mut concentration, &mut concentration_next);
    }

    let cpu_elapsed = start.elapsed();

    let solver = pollster::block_on(WgpuSolver::new(grid));
    let mut gpu_concentration = vec![1.0_f32; n];

    let gpu_start = Instant::now();

    gpu_concentration = pollster::block_on(solver.run_steps(
        &gpu_concentration,
        &input_template.diffusivity,
        &input_template.vmax,
        &input_template.km,
        &input_template.vessel_mask,
        &input_template.vessel_concentration,
        input_template.dt,
        input_template.reset_vessels,
        steps as usize,
    ));

    let gpu_elapsed = gpu_start.elapsed();

    println!("Grid: {}x{}", width, height);
    println!("Steps: {}", steps);

    println!("\nCPU fused:");
    println!("Elapsed: {:.3?}", cpu_elapsed);
    println!("Time per step: {:.6?}", cpu_elapsed / steps);

    println!("\nGPU persistent buffers + one readback:");
    println!("Elapsed: {:.3?}", gpu_elapsed);
    println!("Time per step: {:.6?}", gpu_elapsed / steps);

    let center_idx = n / 2;
    let center_difference = (concentration[center_idx] - gpu_concentration[center_idx]).abs();

    println!("\nCorrectness check:");
    println!("CPU center concentration: {:.6}", concentration[center_idx]);
    println!(
        "GPU center concentration: {:.6}",
        gpu_concentration[center_idx]
    );
    println!("Absolute difference: {:.6e}", center_difference);
}
