use crate::{run_gas_exchange_steps_auto, GasExchangeStepInput, Grid2D};
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// (Old single-species Python wrapper removed)

/// Python/NumPy entry point for the coupled oxygen/carbon dioxide gas-exchange solver.
///
/// This function converts two-dimensional NumPy arrays into Rust-owned row-major
/// vectors, validates that every field shares the same grid shape, runs the
/// coupled O₂/CO₂ gas-exchange solver, and returns both updated concentration
/// fields back to Python.
///
/// All arrays must share the same two-dimensional shape:
///
/// (height, width)
///
/// Parameters:
/// oxygen_initial         -> Initial oxygen concentration field.
/// carbon_dioxide_initial -> Initial carbon dioxide concentration field.
/// diffusivity_o2         -> Effective oxygen diffusivity at each grid cell.
/// diffusivity_co2        -> Effective carbon dioxide diffusivity at each grid cell.
/// vmax                   -> Michaelis-Menten maximum oxygen consumption rate.
/// km                     -> Michaelis-Menten half-saturation constant.
/// vessel_mask            -> Boolean mask indicating vessel source/sink cells.
/// vessel_o2              -> Fixed oxygen concentration imposed on vessel cells.
/// vessel_co2             -> Fixed carbon dioxide concentration imposed on vessel cells.
/// dx                     -> Physical spacing in the x direction.
/// dy                     -> Physical spacing in the y direction.
/// dt                     -> Explicit timestep size.
/// co2_yield              -> CO₂ produced per unit O₂ consumed.
/// steps                  -> Number of timesteps to simulate.
/// reset_vessels          -> Whether vessel cells reset after each update.
///
/// Returns:
/// `(oxygen, carbon_dioxide)` NumPy arrays after `steps * dt` seconds of
/// simulated time.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn run_gas_exchange_steps_auto_numpy<'py>(
    py: Python<'py>,
    oxygen_initial: PyReadonlyArray2<'_, f32>,
    carbon_dioxide_initial: PyReadonlyArray2<'_, f32>,
    diffusivity_o2: PyReadonlyArray2<'_, f32>,
    diffusivity_co2: PyReadonlyArray2<'_, f32>,
    vmax: PyReadonlyArray2<'_, f32>,
    km: PyReadonlyArray2<'_, f32>,
    vessel_mask: PyReadonlyArray2<'_, bool>,
    vessel_o2: PyReadonlyArray2<'_, f32>,
    vessel_co2: PyReadonlyArray2<'_, f32>,
    dx: f32,
    dy: f32,
    dt: f32,
    co2_yield: f32,
    steps: usize,
    reset_vessels: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let (height, width) = oxygen_initial.as_array().dim();
    let expected_shape = (height, width);

    validate_shape(
        "carbon_dioxide_initial",
        carbon_dioxide_initial.as_array().dim(),
        expected_shape,
    )?;
    validate_shape(
        "diffusivity_o2",
        diffusivity_o2.as_array().dim(),
        expected_shape,
    )?;
    validate_shape(
        "diffusivity_co2",
        diffusivity_co2.as_array().dim(),
        expected_shape,
    )?;
    validate_shape("vmax", vmax.as_array().dim(), expected_shape)?;
    validate_shape("km", km.as_array().dim(), expected_shape)?;
    validate_shape("vessel_mask", vessel_mask.as_array().dim(), expected_shape)?;
    validate_shape("vessel_o2", vessel_o2.as_array().dim(), expected_shape)?;
    validate_shape("vessel_co2", vessel_co2.as_array().dim(), expected_shape)?;

    let oxygen_current = flatten_f32(oxygen_initial);
    let carbon_dioxide_current = flatten_f32(carbon_dioxide_initial);
    let mut oxygen_next = vec![0.0; oxygen_current.len()];
    let mut carbon_dioxide_next = vec![0.0; carbon_dioxide_current.len()];

    let input = GasExchangeStepInput {
        grid: Grid2D::new(width, height, dx, dy),
        oxygen_current: &oxygen_current,
        carbon_dioxide_current: &carbon_dioxide_current,
        oxygen_next: &mut oxygen_next,
        carbon_dioxide_next: &mut carbon_dioxide_next,
        oxygen_diffusivity: &flatten_f32(diffusivity_o2),
        carbon_dioxide_diffusivity: &flatten_f32(diffusivity_co2),
        vmax: &flatten_f32(vmax),
        km: &flatten_f32(km),
        vessel_mask: &flatten_bool(vessel_mask),
        vessel_oxygen_concentration: &flatten_f32(vessel_o2),
        vessel_carbon_dioxide_concentration: &flatten_f32(vessel_co2),
        dt,
        co2_yield,
        reset_vessels,
    };

    let (oxygen_result, carbon_dioxide_result) =
        pollster::block_on(run_gas_exchange_steps_auto(input, steps));

    let oxygen_output = Array2::from_shape_vec((height, width), oxygen_result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let carbon_dioxide_output = Array2::from_shape_vec((height, width), carbon_dioxide_result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok((
        PyArray2::from_owned_array(py, oxygen_output),
        PyArray2::from_owned_array(py, carbon_dioxide_output),
    ))
}

/// Register Python-visible functions with the module.
///
/// Keeping registration separate makes it easier to expose additional solver
/// utilities without growing `lib.rs`.
pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_gas_exchange_steps_auto_numpy, m)?)?;
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
