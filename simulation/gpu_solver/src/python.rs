use crate::{run_steps_auto, Grid2D, StepInput};
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python/NumPy entry point for the Rust reaction-diffusion solver.
///
/// This function converts NumPy arrays into Rust-owned vectors, validates shape
/// consistency, constructs a `StepInput`, runs either the GPU or CPU fallback solver,
/// and returns the resulting concentration field back to Python.
///
/// All arrays must share the same two-dimensional shape:
///
/// (height, width)
///
/// Parameters:
/// concentration        -> Initial concentration field.
/// diffusivity          -> Effective diffusivity at each grid cell.
/// vmax                 -> Michaelis-Menten maximum consumption rate.
/// km                   -> Michaelis-Menten half-saturation constant.
/// vessel_mask          -> Boolean mask indicating vessel/source cells.
/// vessel_concentration -> Fixed concentration imposed on vessel cells.
/// dx                   -> Physical spacing in the x direction.
/// dy                   -> Physical spacing in the y direction.
/// dt                   -> Explicit timestep size.
/// steps                -> Number of timesteps to simulate.
/// reset_vessels        -> Whether vessel cells reset after each update.
///
/// Returns:
/// A NumPy array with the same shape as `concentration`, representing the
/// concentration field after `steps * dt` seconds of simulated time.
///
/// Notes:
/// Arrays are flattened into row-major vectors before entering the solver.
/// The Python API hides whether execution occurred through WGPU or the CPU
/// fallback path.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn run_steps_auto_numpy<'py>(
    py: Python<'py>,
    concentration: PyReadonlyArray2<'_, f32>,
    diffusivity: PyReadonlyArray2<'_, f32>,
    vmax: PyReadonlyArray2<'_, f32>,
    km: PyReadonlyArray2<'_, f32>,
    vessel_mask: PyReadonlyArray2<'_, bool>,
    vessel_concentration: PyReadonlyArray2<'_, f32>,
    dx: f32,
    dy: f32,
    dt: f32,
    steps: usize,
    reset_vessels: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (height, width) = concentration.as_array().dim();
    let expected_shape = (height, width);

    validate_shape("diffusivity", diffusivity.as_array().dim(), expected_shape)?;
    validate_shape("vmax", vmax.as_array().dim(), expected_shape)?;
    validate_shape("km", km.as_array().dim(), expected_shape)?;
    validate_shape("vessel_mask", vessel_mask.as_array().dim(), expected_shape)?;
    validate_shape(
        "vessel_concentration",
        vessel_concentration.as_array().dim(),
        expected_shape,
    )?;

    let input = StepInput {
        grid: Grid2D::new(width, height, dx, dy),
        concentration: flatten_f32(concentration),
        diffusivity: flatten_f32(diffusivity),
        vmax: flatten_f32(vmax),
        km: flatten_f32(km),
        vessel_mask: flatten_bool(vessel_mask),
        vessel_concentration: flatten_f32(vessel_concentration),
        dt,
        reset_vessels,
    };

    let result = pollster::block_on(run_steps_auto(&input, steps));
    let output = Array2::from_shape_vec((height, width), result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok(PyArray2::from_owned_array(py, output))
}

/// Register Python-visible functions with the module.
///
/// Keeping registration separate makes it easier to expose additional solver
/// utilities without growing `lib.rs`.
pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_steps_auto_numpy, m)?)?;
    Ok(())
}

/// Ensure a NumPy array matches the expected solver grid shape.
///
/// Shape mismatches become Python `ValueError`s instead of Rust panics.
fn validate_shape(name: &str, actual: (usize, usize), expected: (usize, usize)) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "{name} shape mismatch: expected {:?}, got {:?}",
            expected, actual
        )));
    }

    Ok(())
}

/// Convert a 2D NumPy float array into a flat row-major Rust vector.
///
/// The solver operates on contiguous one-dimensional buffers for CPU and GPU
/// compatibility.
fn flatten_f32(array: PyReadonlyArray2<'_, f32>) -> Vec<f32> {
    array.as_array().iter().copied().collect()
}

/// Convert a 2D NumPy boolean mask into a flat Rust vector.
fn flatten_bool(array: PyReadonlyArray2<'_, bool>) -> Vec<bool> {
    array.as_array().iter().copied().collect()
}
