use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::exchange::compute_exchange_set;
use crate::flow::solve_flow;
use crate::maps::{build_boundary_source_maps, rasterize_segments};
use crate::network::{
    build_topology_from_perfusion, classify_perfusion_paths, source_nodes_from_pressure_boundaries,
    VesselNetwork,
};
use crate::transport::{default_species_parameters, propagate_species_set};
use crate::types::{
    BoundarySourceMaps, DirectedTopology, ExchangeResult, FlowSolution, NodeKind, PressureBoundary,
    SpeciesTransportResult, VesselNode, VesselPoint, VesselSegment, VesselTransportOutput,
};

// Python-facing node input.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyVesselNode {
    #[pyo3(get, set)]
    pub id: usize,
    #[pyo3(get, set)]
    pub x_um: f32,
    #[pyo3(get, set)]
    pub y_um: f32,
}

#[pymethods]
impl PyVesselNode {
    #[new]
    pub fn new(id: usize, x_um: f32, y_um: f32) -> Self {
        Self { id, x_um, y_um }
    }
}

// Python-facing centerline point input.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyVesselPoint {
    #[pyo3(get, set)]
    pub x_um: f32,
    #[pyo3(get, set)]
    pub y_um: f32,
}

#[pymethods]
impl PyVesselPoint {
    #[new]
    pub fn new(x_um: f32, y_um: f32) -> Self {
        Self { x_um, y_um }
    }
}

// Python-facing segment input.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyVesselSegment {
    #[pyo3(get, set)]
    pub id: usize,
    #[pyo3(get, set)]
    pub start_node: usize,
    #[pyo3(get, set)]
    pub end_node: usize,
    #[pyo3(get, set)]
    pub length_um: f32,
    #[pyo3(get, set)]
    pub radius_um: f32,
    #[pyo3(get, set)]
    pub centerline: Vec<PyVesselPoint>,
}

#[pymethods]
impl PyVesselSegment {
    #[new]
    pub fn new(
        id: usize,
        start_node: usize,
        end_node: usize,
        length_um: f32,
        radius_um: f32,
        centerline: Vec<PyVesselPoint>,
    ) -> Self {
        Self {
            id,
            start_node,
            end_node,
            length_um,
            radius_um,
            centerline,
        }
    }
}

// Python-facing pressure boundary input.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyPressureBoundary {
    #[pyo3(get, set)]
    pub node_id: usize,
    #[pyo3(get, set)]
    pub pressure_mmhg: f32,
}

#[pymethods]
impl PyPressureBoundary {
    #[new]
    pub fn new(node_id: usize, pressure_mmhg: f32) -> Self {
        Self {
            node_id,
            pressure_mmhg,
        }
    }
}

// Solve only the hydraulic flow stage from Python.
#[pyfunction]
pub fn solve_flow_from_python(
    nodes: Vec<PyVesselNode>,
    segments: Vec<PyVesselSegment>,
    boundaries: Vec<PyPressureBoundary>,
    viscosity_pa_s: f32,
) -> PyResult<String> {
    let network = build_network(nodes, segments);
    let boundaries = build_boundaries(boundaries);
    let flow = solve_flow(&network, &boundaries, viscosity_pa_s);

    Ok(format_flow_summary(&flow))
}

// Solve the current high-level vessel transport pipeline from Python.
//
// Python provides graph data and grid dimensions. Rust owns flow, topology,
// species propagation, exchange, and source-map construction. The returned
// Python dictionary contains real Rust-computed data, not Python-side fallback
// transport values.
#[pyfunction]
pub fn solve_vessel_transport_from_python(
    py: Python<'_>,
    nodes: Vec<PyVesselNode>,
    segments: Vec<PyVesselSegment>,
    boundaries: Vec<PyPressureBoundary>,
    viscosity_pa_s: f32,
    grid_width: usize,
    grid_height: usize,
) -> PyResult<Py<PyDict>> {
    let network = build_network(nodes, segments);
    let boundaries = build_boundaries(boundaries);

    let source_nodes = source_nodes_from_pressure_boundaries(&boundaries);
    let flow = solve_flow(&network, &boundaries, viscosity_pa_s);
    let perfusion = classify_perfusion_paths(&network, &source_nodes);
    let topology = build_topology_from_perfusion(&perfusion, &flow);
    let species = default_species_parameters();
    let species_transport = propagate_species_set(&topology, &species);
    let exchange = compute_exchange_set(&species_transport);
    let segment_cells = rasterize_segments(&network, grid_width, grid_height);
    let boundary_source_maps =
        build_boundary_source_maps(grid_width, grid_height, &exchange, &segment_cells);

    let output = VesselTransportOutput {
        flow,
        topology,
        species_transport,
        exchange,
        boundary_source_maps,
    };

    transport_output_to_python(py, &output)
}

fn transport_output_to_python(
    py: Python<'_>,
    output: &VesselTransportOutput,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    dict.set_item("summary", format_transport_summary(output))?;
    dict.set_item("flow", flow_to_python(py, &output.flow)?)?;
    dict.set_item("topology", topology_to_python(py, &output.topology)?)?;
    dict.set_item(
        "species_transport",
        species_transport_to_python(py, &output.species_transport)?,
    )?;
    dict.set_item("exchange", exchange_to_python(py, &output.exchange)?)?;
    dict.set_item(
        "boundary_source_maps",
        boundary_source_maps_to_python(py, &output.boundary_source_maps)?,
    )?;

    Ok(dict.into())
}

fn flow_to_python(py: Python<'_>, flow: &FlowSolution) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let node_pressures = PyList::empty(py);
    let segment_flows = PyList::empty(py);

    for node_pressure in &flow.node_pressures {
        let item = PyDict::new(py);
        item.set_item("node_id", node_pressure.node_id)?;
        item.set_item("pressure_mmhg", node_pressure.pressure_mmhg)?;
        node_pressures.append(item)?;
    }

    for segment_flow in &flow.segment_flows {
        let item = PyDict::new(py);
        item.set_item("segment_id", segment_flow.segment_id)?;
        item.set_item("flow_um3_per_s", segment_flow.flow_um3_per_s)?;
        segment_flows.append(item)?;
    }

    dict.set_item("node_pressures", node_pressures)?;
    dict.set_item("segment_flows", segment_flows)?;

    Ok(dict.into())
}

fn topology_to_python(py: Python<'_>, topology: &DirectedTopology) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let downstream_connections = PyList::empty(py);

    for connection in &topology.downstream_connections {
        let item = PyDict::new(py);
        item.set_item("from_node", connection.from_node)?;
        item.set_item("to_node", connection.to_node)?;
        item.set_item("segment_id", connection.segment_id)?;
        item.set_item("flow_um3_per_s", connection.flow_um3_per_s)?;
        downstream_connections.append(item)?;
    }

    dict.set_item("inlet_nodes", topology.inlet_nodes.clone())?;
    dict.set_item("outlet_nodes", topology.outlet_nodes.clone())?;
    dict.set_item("traversal_segments", topology.traversal_segments.clone())?;
    dict.set_item("downstream_connections", downstream_connections)?;

    Ok(dict.into())
}

fn species_transport_to_python(
    py: Python<'_>,
    species_transport: &[SpeciesTransportResult],
) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);

    for result in species_transport {
        let item = PyDict::new(py);
        let states = PyList::empty(py);

        item.set_item("species", format!("{:?}", result.species))?;

        for state in &result.segment_states {
            let state_item = PyDict::new(py);
            state_item.set_item("segment_id", state.segment_id)?;
            state_item.set_item("inlet_concentration", state.inlet_concentration)?;
            state_item.set_item("outlet_concentration", state.outlet_concentration)?;
            state_item.set_item("exchanged_amount_per_s", state.exchanged_amount_per_s)?;
            states.append(state_item)?;
        }

        item.set_item("segment_states", states)?;
        list.append(item)?;
    }

    Ok(list.into())
}

fn exchange_to_python(py: Python<'_>, exchange: &ExchangeResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let segment_exchanges = PyList::empty(py);

    for segment_exchange in &exchange.segment_exchanges {
        let item = PyDict::new(py);
        item.set_item("segment_id", segment_exchange.segment_id)?;
        item.set_item("species", format!("{:?}", segment_exchange.species))?;
        item.set_item("source_amount_per_s", segment_exchange.source_amount_per_s)?;
        segment_exchanges.append(item)?;
    }

    dict.set_item("segment_exchanges", segment_exchanges)?;

    Ok(dict.into())
}

fn boundary_source_maps_to_python(
    py: Python<'_>,
    boundary_source_maps: &BoundarySourceMaps,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let maps = PyList::empty(py);

    for source_map in &boundary_source_maps.maps {
        let item = PyDict::new(py);
        item.set_item("species", format!("{:?}", source_map.species))?;
        item.set_item("width", source_map.width)?;
        item.set_item("height", source_map.height)?;
        item.set_item("values", source_map.values.clone())?;
        maps.append(item)?;
    }

    dict.set_item("maps", maps)?;

    Ok(dict.into())
}

// Register Python bindings for the vessel solver module.
pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyVesselNode>()?;
    module.add_class::<PyVesselPoint>()?;
    module.add_class::<PyVesselSegment>()?;
    module.add_class::<PyPressureBoundary>()?;
    module.add_function(wrap_pyfunction!(solve_flow_from_python, module)?)?;
    module.add_function(wrap_pyfunction!(
        solve_vessel_transport_from_python,
        module
    )?)?;

    Ok(())
}

fn build_network(nodes: Vec<PyVesselNode>, segments: Vec<PyVesselSegment>) -> VesselNetwork {
    let nodes = nodes
        .into_iter()
        .map(|node| VesselNode {
            id: node.id,
            x_um: node.x_um,
            y_um: node.y_um,
            kind: NodeKind::Unknown,
        })
        .collect();

    let segments = segments
        .into_iter()
        .map(|segment| VesselSegment {
            id: segment.id,
            start_node: segment.start_node,
            end_node: segment.end_node,
            length_um: segment.length_um,
            radius_um: segment.radius_um,
            centerline: segment
                .centerline
                .into_iter()
                .map(|point| VesselPoint {
                    x_um: point.x_um,
                    y_um: point.y_um,
                })
                .collect(),
        })
        .collect();

    VesselNetwork::from_parts(nodes, segments)
}

fn build_boundaries(boundaries: Vec<PyPressureBoundary>) -> Vec<PressureBoundary> {
    boundaries
        .into_iter()
        .map(|boundary| PressureBoundary {
            node_id: boundary.node_id,
            pressure_mmhg: boundary.pressure_mmhg,
        })
        .collect()
}

fn format_flow_summary(flow: &FlowSolution) -> String {
    format!(
        "solved node pressures: {}\nsolved segment flows: {}",
        flow.node_pressures.len(),
        flow.segment_flows.len(),
    )
}

fn format_transport_summary(output: &VesselTransportOutput) -> String {
    format!(
		"solved node pressures: {}\nsolved segment flows: {}\nperfusion inlet nodes: {}\nperfusion outlet nodes: {}\nperfusion traversal segments: {}\nspecies transported: {}\nexchange entries: {}\nsource maps: {}",
		output.flow.node_pressures.len(),
		output.flow.segment_flows.len(),
		output.topology.inlet_nodes.len(),
		output.topology.outlet_nodes.len(),
		output.topology.traversal_segments.len(),
		output.species_transport.len(),
		output.exchange.segment_exchanges.len(),
		map_count(&output.boundary_source_maps),
	)
}

fn map_count(boundary_source_maps: &BoundarySourceMaps) -> usize {
    boundary_source_maps.maps.len()
}
