pub mod pressure;
pub mod resistance;

pub use pressure::solve_pressure_flow;
pub use resistance::{poiseuille_resistance, M3_TO_UM3, UM3_TO_M3, UM_TO_M};

use crate::network::model::VesselNetwork;
use crate::types::{FlowSolution, PressureBoundary};

// Solve the hydraulic pressure-flow stage.
//
// This produces node pressures and signed segment flow magnitudes only. Segment
// direction for vessel transport now comes from the perfusion-front topology,
// not from pressure-derived direction reconstruction.
pub fn solve_flow(
    network: &VesselNetwork,
    boundaries: &[PressureBoundary],
    viscosity_pa_s: f32,
) -> FlowSolution {
    solve_pressure_flow(network, boundaries, viscosity_pa_s)
}
