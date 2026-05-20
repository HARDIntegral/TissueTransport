// -----------------------------------------------------------------------------
// Scalar parameters shared between Rust and the WGSL compute shader.
//
// These values remain constant during a simulation chunk and describe:
// - grid dimensions
// - physical spacing (dx, dy)
// - timestep size (dt)
// - whether vessel/source cells reset after each update
//
// Rust and WGSL layouts must remain identical.
// -----------------------------------------------------------------------------
struct StepParams {
    width: u32,
    height: u32,
    dx: f32,
    dy: f32,
    dt: f32,
    reset_vessels: u32,
    _pad0_: u32,
    _pad1_: u32,
}

@group(0) @binding(0) 
var<storage> concentration_current: array<f32>;
@group(0) @binding(1) 
var<storage, read_write> concentration_next: array<f32>;
@group(0) @binding(2) 
var<storage> diffusivity: array<f32>;
@group(0) @binding(3) 
var<storage> vmax: array<f32>;
@group(0) @binding(4) 
var<storage> km: array<f32>;
@group(0) @binding(5) 
var<storage> vessel_mask: array<u32>;
@group(0) @binding(6) 
var<storage> vessel_concentration: array<f32>;
@group(0) @binding(7) 
var<uniform> params: StepParams;

// -----------------------------------------------------------------------------
// Compute Michaelis-Menten oxygen consumption.
//
// consumption = vmax * C / (km + C)
//
// Returns zero if the denominator would be invalid.
// -----------------------------------------------------------------------------
fn michaelis_menten(concentration: f32, vmax_value: f32, km_value: f32) -> f32 {
    if ((km_value + concentration) > 0f) {
        return ((vmax_value * concentration) / (km_value + concentration));
    }
    return 0f;
}

// -----------------------------------------------------------------------------
// Main fused reaction-diffusion compute shader.
//
// Each invocation updates one grid cell:
//
// C_next = C_current + dt * (diffusion - consumption)
//
// Diffusion:
// - uses face-averaged diffusivity
// - applies finite differences in x and y directions
//
// Reaction:
// - follows Michaelis-Menten consumption kinetics
//
// Source terms:
// - optional vessel reset overwrites concentration after update
// -----------------------------------------------------------------------------
@compute @workgroup_size(16, 16, 1) 
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var local: bool;
    var diffusion: f32 = 0f;
    var next_value: f32;
    var local_1_: bool;

    // Convert global invocation ID into grid coordinates.
    let col = id.x;
    let row = id.y;
    let _e5_ = params.width;
    if !((col >= _e5_)) {
        let _e12_ = params.height;
        local = (row >= _e12_);
    } else {
        local = true;
    }
    let _e15_ = local;
    if _e15_ {
        return;
    }
    let width = params.width;
    let height = params.height;
    // Flat row-major index used by all storage buffers.
    let idx = ((row * width) + col);
    let center = concentration_current[idx];
    let d_center = diffusivity[idx];
    let _e32_ = params.dx;
    let _e35_ = params.dx;
    let inv_dx2_ = (1f / (_e32_ * _e35_));
    let _e41_ = params.dy;
    let _e44_ = params.dy;
    let inv_dy2_ = (1f / (_e41_ * _e44_));
    // Accumulate diffusion contributions from neighboring cells.
    if (col > 0u) {
        let left_idx = (idx - 1u);
        let _e56_ = diffusivity[left_idx];
        let d_left = (0.5f * (d_center + _e56_));
        let _e62_ = concentration_current[left_idx];
        let _e66_ = diffusion;
        diffusion = (_e66_ + ((d_left * (_e62_ - center)) * inv_dx2_));
    }
    if ((col + 1u) < width) {
        let right_idx = (idx + 1u);
        let _e75_ = diffusivity[right_idx];
        let d_right = (0.5f * (d_center + _e75_));
        let _e81_ = concentration_current[right_idx];
        let _e85_ = diffusion;
        diffusion = (_e85_ + ((d_right * (_e81_ - center)) * inv_dx2_));
    }
    if (row > 0u) {
        let up_idx = (idx - width);
        let _e92_ = diffusivity[up_idx];
        let d_up = (0.5f * (d_center + _e92_));
        let _e98_ = concentration_current[up_idx];
        let _e102_ = diffusion;
        diffusion = (_e102_ + ((d_up * (_e98_ - center)) * inv_dy2_));
    }
    if ((row + 1u) < height) {
        let down_idx = (idx + width);
        let _e110_ = diffusivity[down_idx];
        let d_down = (0.5f * (d_center + _e110_));
        let _e116_ = concentration_current[down_idx];
        let _e120_ = diffusion;
        diffusion = (_e120_ + ((d_down * (_e116_ - center)) * inv_dy2_));
    }
    // Apply Michaelis-Menten oxygen consumption.
    let _e124_ = vmax[idx];
    let _e127_ = km[idx];
    let _e129 = michaelis_menten(center, _e124_, _e127_);
    let _e131_ = params.dt;
    let _e132_ = diffusion;
    next_value = (center + (_e131_ * (_e132_ - _e129)));
    // Optionally overwrite vessel cells with fixed concentrations.
    let _e139_ = params.reset_vessels;
    if (_e139_ != 0u) {
        let _e146_ = vessel_mask[idx];
        local_1_ = (_e146_ != 0u);
    } else {
        local_1_ = false;
    }
    let _e150_ = local_1_;
    if _e150_ {
        let _e153_ = vessel_concentration[idx];
        next_value = _e153_;
    }
    // Write updated concentration into the output buffer.
    let _e156_ = next_value;
    concentration_next[idx] = _e156_;
    return;
}
