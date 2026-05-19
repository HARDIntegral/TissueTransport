use gpu_solver::{explicit_step, Grid2D, StepInput};
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

    let start = Instant::now();

    for _ in 0..steps {
        let input = StepInput {
            concentration,
            ..input_template.clone()
        };

        concentration = explicit_step(&input);
    }

    let elapsed = start.elapsed();

    println!("Grid: {}x{}", width, height);
    println!("Steps: {}", steps);
    println!("Elapsed: {:.3?}", elapsed);
    println!("Time per step: {:.6?}", elapsed / steps);
}
