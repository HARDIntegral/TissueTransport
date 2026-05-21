// -----------------------------------------------------------------------------
// Uniform parameters shared between Rust and the coupled O₂/CO₂ GPU shader.
//
// width, height -> Grid dimensions.
// dx, dy        -> Physical spacing between cells.
// dt            -> Explicit timestep size.
// reset_vessels -> Whether vessel cells are overwritten after each step.
// co2_yield     -> CO₂ produced per unit O₂ consumed.
//
// NOTE:
// Layout must exactly match `GpuGasExchangeParams` in Rust.
// -----------------------------------------------------------------------------
struct StepParams {
	width: u32,
	height: u32,
	dx: f32,
	dy: f32,
	dt: f32,
	reset_vessels: u32,
	co2_yield: f32,
	_pad0: u32,
};

@group(0) @binding(0) var<storage, read> gas_current: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> gas_next: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> diffusivity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> metabolism: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> vessel_mask: array<u32>;
@group(0) @binding(5) var<storage, read> vessel_concentration: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> params: StepParams;

// Convert (row, col) coordinates into a flattened 1D array index.
fn index(row: u32, col: u32) -> u32 {
	return row * params.width + col;
}

// Compute Michaelis-Menten oxygen consumption:
// consumption = vmax * [O2] / (Km + [O2])
fn michaelis_menten(concentration: f32, local_vmax: f32, local_km: f32) -> f32 {
	return local_vmax * concentration / (local_km + concentration);
}

// -----------------------------------------------------------------------------
// One coupled gas-exchange timestep.
//
// Each invocation updates a single grid cell:
//
// O₂_next  = O₂ + dt * (diffusion - consumption)
// CO₂_next = CO₂ + dt * (diffusion + co2_yield * consumption)
//
// Vessel cells optionally reset to fixed concentrations so they act as
// persistent O₂ sources and CO₂ sinks.
// -----------------------------------------------------------------------------
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let col = global_id.x;
	let row = global_id.y;

	if (col >= params.width || row >= params.height) {
		return;
	}

	let idx = index(row, col);

	if (params.reset_vessels == 1u && vessel_mask[idx] != 0u) {
		gas_next[idx] = vessel_concentration[idx];
		return;
	}

	let inv_dx2 = 1.0 / (params.dx * params.dx);
	let inv_dy2 = 1.0 / (params.dy * params.dy);

	let center = gas_current[idx];
	let d_center = diffusivity[idx];

	var gas_diffusion = vec2<f32>(0.0, 0.0);

	// Accumulate diffusion flux from neighboring cells.
	if (col > 0u) {
		let neighbor_idx = idx - 1u;
		let d_face = 0.5 * (d_center + diffusivity[neighbor_idx]);
		gas_diffusion += d_face * (gas_current[neighbor_idx] - center) * inv_dx2;
	}

	if (col + 1u < params.width) {
		let neighbor_idx = idx + 1u;
		let d_face = 0.5 * (d_center + diffusivity[neighbor_idx]);
		gas_diffusion += d_face * (gas_current[neighbor_idx] - center) * inv_dx2;
	}

	if (row > 0u) {
		let neighbor_idx = idx - params.width;
		let d_face = 0.5 * (d_center + diffusivity[neighbor_idx]);
		gas_diffusion += d_face * (gas_current[neighbor_idx] - center) * inv_dy2;
	}

	if (row + 1u < params.height) {
		let neighbor_idx = idx + params.width;
		let d_face = 0.5 * (d_center + diffusivity[neighbor_idx]);
		gas_diffusion += d_face * (gas_current[neighbor_idx] - center) * inv_dy2;
	}

	// Compute metabolic oxygen consumption and resulting carbon dioxide production.
	let local_metabolism = metabolism[idx];
	let oxygen_consumption = michaelis_menten(center.x, local_metabolism.x, local_metabolism.y);
	let reaction = vec2<f32>(-oxygen_consumption, params.co2_yield * oxygen_consumption);

	// Explicit Euler update.
	gas_next[idx] = center + params.dt * (gas_diffusion + reaction);
}