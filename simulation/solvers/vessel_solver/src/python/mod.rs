use pyo3::prelude::*;

pub mod bindings;

pub use bindings::{
    register_python_bindings, solve_flow_from_python, solve_vessel_transport_from_python,
    PyPressureBoundary, PyVesselNode, PyVesselPoint, PyVesselSegment,
};

#[pymodule]
fn vessel_solver(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_python_bindings(module)
}
