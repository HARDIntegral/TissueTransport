use crate::types::Grid2D;
use wgpu::util::DeviceExt;

const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;

/// Scalar timestep parameters shared between Rust and the compute shader.
///
/// This struct is copied into a uniform buffer and read by the WGSL shader during
/// each dispatch. It contains grid shape, physical spacing, timestep size, and the
/// vessel reset flag. Per-cell arrays such as concentration and diffusivity are
/// passed separately as storage buffers.
///
/// Parameters:
/// width         -> Number of grid columns.
/// height        -> Number of grid rows.
/// dx            -> Physical spacing in the x direction.
/// dy            -> Physical spacing in the y direction.
/// dt            -> Explicit timestep size.
/// reset_vessels -> Whether vessel/source cells should be reset after each update.
///
/// Notes:
/// The padding fields keep the Rust memory layout aligned with the WGSL `StepParams`
/// struct used by `reaction_diffusion_step.wgsl`. Rust and WGSL must agree on this
/// layout exactly or the shader will read incorrect parameter values.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuStepParams {
	pub width: u32,
	pub height: u32,
	pub dx: f32,
	pub dy: f32,
	pub dt: f32,
	pub reset_vessels: u32,
	pub _pad0: u32,
	pub _pad1: u32,
}

impl GpuStepParams {
	/// Build shader parameters from grid metadata and timestep options.
	pub fn from_grid(grid: Grid2D, dt: f32, reset_vessels: bool) -> Self {
		Self {
			width: grid.width as u32,
			height: grid.height as u32,
			dx: grid.dx,
			dy: grid.dy,
			dt,
			reset_vessels: reset_vessels as u32,
			_pad0: 0,
			_pad1: 0,
		}
	}
}

/// GPU buffers whose contents stay fixed across a multi-step simulation chunk.
///
/// Diffusivity, reaction parameters, vessel masks, vessel concentrations, and scalar
/// timestep parameters do not change while `run_steps` is executing. Grouping them
/// together keeps bind group creation cleaner and makes the double-buffered
/// concentration update easier to follow.
struct StaticStepBuffers {
	diffusivity: wgpu::Buffer,
	vmax: wgpu::Buffer,
	km: wgpu::Buffer,
	vessel_mask: wgpu::Buffer,
	vessel_concentration: wgpu::Buffer,
	params: wgpu::Buffer,
}

/// GPU-backed reaction-diffusion solver.
///
/// This type owns the WGPU device, queue, compute pipeline, and bind group layout
/// needed to run fused reaction-diffusion timesteps on the GPU. The solver itself
/// stores reusable GPU state, while each `run_steps` call uploads the current input
/// arrays, dispatches the compute shader for a requested number of timesteps, and
/// reads back only the final concentration field.
///
/// Notes:
/// The solver uses two concentration buffers as a double buffer:
///
/// concentration_a -> concentration_b
/// concentration_b -> concentration_a
///
/// Each shader dispatch advances the field by one explicit timestep. Alternating
/// bind groups lets the GPU run many timesteps before a single final readback.
pub struct WgpuSolver {
	grid: Grid2D,
	device: wgpu::Device,
	queue: wgpu::Queue,
	pipeline: wgpu::ComputePipeline,
	bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuSolver {
	/// Try to create a WGPU solver and compile the compute shader.
	///
	/// This performs all one-time GPU setup: instance creation, adapter selection,
	/// device/queue creation, shader compilation, bind group layout creation, and
	/// compute pipeline creation.
	///
	/// Parameters:
	/// grid -> Grid metadata used by the solver and dispatch sizing logic.
	///
	/// Returns:
	/// A ready-to-use `WgpuSolver` if WGPU initialization succeeds.
	///
	/// Notes:
	/// This returns an error instead of panicking so higher-level code can fall back
	/// to the CPU reference solver when no compatible GPU is available.
	pub async fn try_new(grid: Grid2D) -> Result<Self, String> {
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::all(),
			flags: Default::default(),
			memory_budget_thresholds: Default::default(),
			backend_options: Default::default(),
			display: Default::default(),
		});

		let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::HighPerformance,
				force_fallback_adapter: false,
				compatible_surface: None,
			})
			.await
			.map_err(|error| format!("No compatible GPU adapter found: {error:?}"))?;

		let (device, queue) = adapter
			.request_device(&wgpu::DeviceDescriptor {
				label: Some("reaction-diffusion device"),
				required_features: wgpu::Features::empty(),
				required_limits: wgpu::Limits::default(),
				experimental_features: Default::default(),
				memory_hints: Default::default(),
				trace: Default::default(),
			})
			.await
			.map_err(|error| format!("Failed to create WGPU device: {error:?}"))?;

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("reaction_diffusion_step shader"),
			source: wgpu::ShaderSource::Wgsl(
				include_str!("shaders/reaction_diffusion_step.wgsl").into(),
			),
		});

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("reaction-diffusion bind group layout"),
			entries: &[
				storage_entry(0, true),
				storage_entry(1, false),
				storage_entry(2, true),
				storage_entry(3, true),
				storage_entry(4, true),
				storage_entry(5, true),
				storage_entry(6, true),
				uniform_entry(7),
			],
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("reaction-diffusion pipeline layout"),
			bind_group_layouts: &[Some(&bind_group_layout)],
			immediate_size: 0,
		});

		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("reaction-diffusion compute pipeline"),
			layout: Some(&pipeline_layout),
			module: &shader,
			entry_point: Some("main"),
			compilation_options: Default::default(),
			cache: None,
		});

		Ok(Self {
			grid,
			device,
			queue,
			pipeline,
			bind_group_layout,
		})
	}

	/// Create a WGPU solver and panic if no compatible GPU is available.
	///
	/// This is a convenience constructor for tests, examples, and situations where GPU
	/// initialization failure should be treated as fatal. Use `try_new` when a CPU
	/// fallback is desired.
	pub async fn new(grid: Grid2D) -> Self {
		Self::try_new(grid)
			.await
			.expect("Failed to initialize WGPU solver")
	}

	/// Return the grid associated with this solver.
	pub fn grid(&self) -> Grid2D {
		self.grid
	}

	/// Run multiple reaction-diffusion timesteps while keeping data on the GPU.
	///
	/// This is the optimized GPU path. Fixed arrays are uploaded once, two concentration
	/// buffers are reused as a double buffer, all shader dispatches are encoded into one
	/// command buffer, and only the final concentration field is copied back to CPU memory.
	///
	/// Parameters:
	/// concentration_initial -> Concentration field at the start of the chunk.
	/// diffusivity           -> Effective diffusivity at each grid cell.
	/// vmax                  -> Michaelis-Menten maximum consumption rate.
	/// km                    -> Michaelis-Menten half-saturation constant.
	/// vessel_mask           -> Boolean mask marking vessel/source cells.
	/// vessel_concentration  -> Fixed concentration imposed on vessel/source cells.
	/// dt                    -> Explicit timestep size.
	/// reset_vessels         -> Whether vessel cells reset after each shader step.
	/// steps                 -> Number of shader timesteps to run.
	///
	/// Returns:
	/// Concentration field after `steps` GPU timesteps.
	///
	/// Notes:
	/// Calling this with a large `steps` value is much faster than calling it repeatedly
	/// with `steps = 1`, because it reduces Python/Rust calls, GPU buffer uploads, and
	/// readbacks. The physical time advanced is `steps * dt`.
	pub async fn run_steps(
		&self,
		concentration_initial: &[f32],
		diffusivity: &[f32],
		vmax: &[f32],
		km: &[f32],
		vessel_mask: &[bool],
		vessel_concentration: &[f32],
		dt: f32,
		reset_vessels: bool,
		steps: usize,
	) -> Vec<f32> {
		self.validate_step_inputs(
			concentration_initial,
			diffusivity,
			vmax,
			km,
			vessel_mask,
			vessel_concentration,
		);

		if steps == 0 {
			return concentration_initial.to_vec();
		}

		let byte_size = std::mem::size_of_val(concentration_initial) as u64;
		let static_buffers = self.create_static_buffers(
			diffusivity,
			vmax,
			km,
			vessel_mask,
			vessel_concentration,
			dt,
			reset_vessels,
		);

		let concentration_a = create_storage_buffer(
			&self.device,
			"concentration A buffer",
			concentration_initial,
			wgpu::BufferUsages::COPY_SRC,
		);
		let concentration_b = self.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("concentration B buffer"),
			size: byte_size,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
			mapped_at_creation: false,
		});
		let readback = create_readback_buffer(&self.device, byte_size);

		let bind_group_a_to_b = self.create_step_bind_group(
			"reaction-diffusion bind group A to B",
			&concentration_a,
			&concentration_b,
			&static_buffers,
		);
		let bind_group_b_to_a = self.create_step_bind_group(
			"reaction-diffusion bind group B to A",
			&concentration_b,
			&concentration_a,
			&static_buffers,
		);

		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("reaction-diffusion run_steps command encoder"),
			});

		self.encode_timestep_dispatches(
			&mut encoder,
			&bind_group_a_to_b,
			&bind_group_b_to_a,
			steps,
		);

		let final_buffer = if steps % 2 == 0 {
			&concentration_a
		} else {
			&concentration_b
		};

		encoder.copy_buffer_to_buffer(final_buffer, 0, &readback, 0, byte_size);
		self.queue.submit(Some(encoder.finish()));

		self.read_buffer(&readback)
	}

	/// Validate that every input slice matches the solver grid size.
	///
	/// This catches shape mismatches before any GPU buffers are created.
	fn validate_step_inputs(
		&self,
		concentration: &[f32],
		diffusivity: &[f32],
		vmax: &[f32],
		km: &[f32],
		vessel_mask: &[bool],
		vessel_concentration: &[f32],
	) {
		let n = self.grid.len();

		assert_eq!(concentration.len(), n);
		assert_eq!(diffusivity.len(), n);
		assert_eq!(vmax.len(), n);
		assert_eq!(km.len(), n);
		assert_eq!(vessel_mask.len(), n);
		assert_eq!(vessel_concentration.len(), n);
	}

	/// Upload fixed per-cell arrays and scalar parameters into GPU buffers.
	///
	/// These buffers are reused for every dispatch inside one `run_steps` call. The
	/// vessel mask is converted from Rust `bool` values to `u32` values because WGSL
	/// storage buffers should use shader-friendly scalar types.
	fn create_static_buffers(
		&self,
		diffusivity: &[f32],
		vmax: &[f32],
		km: &[f32],
		vessel_mask: &[bool],
		vessel_concentration: &[f32],
		dt: f32,
		reset_vessels: bool,
	) -> StaticStepBuffers {
		let vessel_mask_gpu = vessel_mask
			.iter()
			.map(|value| u32::from(*value))
			.collect::<Vec<_>>();
		let params = GpuStepParams::from_grid(self.grid, dt, reset_vessels);

		StaticStepBuffers {
			diffusivity: create_storage_buffer(
				&self.device,
				"diffusivity buffer",
				diffusivity,
				wgpu::BufferUsages::empty(),
			),
			vmax: create_storage_buffer(
				&self.device,
				"vmax buffer",
				vmax,
				wgpu::BufferUsages::empty(),
			),
			km: create_storage_buffer(
				&self.device,
				"km buffer",
				km,
				wgpu::BufferUsages::empty(),
			),
			vessel_mask: create_storage_buffer(
				&self.device,
				"vessel mask buffer",
				&vessel_mask_gpu,
				wgpu::BufferUsages::empty(),
			),
			vessel_concentration: create_storage_buffer(
				&self.device,
				"vessel concentration buffer",
				vessel_concentration,
				wgpu::BufferUsages::empty(),
			),
			params: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some("step params buffer"),
				contents: bytemuck::bytes_of(&params),
				usage: wgpu::BufferUsages::UNIFORM,
			}),
		}
	}

	/// Create a bind group for one direction of the concentration double buffer.
	///
	/// The same shader is used for both directions. Only bindings 0 and 1 change:
	/// one bind group reads from A and writes to B, while the other reads from B and
	/// writes to A.
	fn create_step_bind_group(
		&self,
		label: &str,
		concentration_current: &wgpu::Buffer,
		concentration_next: &wgpu::Buffer,
		static_buffers: &StaticStepBuffers,
	) -> wgpu::BindGroup {
		self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some(label),
			layout: &self.bind_group_layout,
			entries: &[
				bind_entry(0, concentration_current),
				bind_entry(1, concentration_next),
				bind_entry(2, &static_buffers.diffusivity),
				bind_entry(3, &static_buffers.vmax),
				bind_entry(4, &static_buffers.km),
				bind_entry(5, &static_buffers.vessel_mask),
				bind_entry(6, &static_buffers.vessel_concentration),
				bind_entry(7, &static_buffers.params),
			],
		})
	}

	/// Encode all timestep dispatches for a simulation chunk.
	///
	/// Each dispatch advances the concentration field by one timestep. The bind group
	/// alternates every dispatch so the output from one step becomes the input to the
	/// next step without copying buffers between dispatches.
	fn encode_timestep_dispatches(
		&self,
		encoder: &mut wgpu::CommandEncoder,
		bind_group_a_to_b: &wgpu::BindGroup,
		bind_group_b_to_a: &wgpu::BindGroup,
		steps: usize,
	) {
		let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
			label: Some("reaction-diffusion run_steps compute pass"),
			timestamp_writes: None,
		});
		compute_pass.set_pipeline(&self.pipeline);

		let groups_x = (self.grid.width as u32).div_ceil(WORKGROUP_X);
		let groups_y = (self.grid.height as u32).div_ceil(WORKGROUP_Y);

		for step_idx in 0..steps {
			let bind_group = if step_idx % 2 == 0 {
				bind_group_a_to_b
			} else {
				bind_group_b_to_a
			};

			compute_pass.set_bind_group(0, bind_group, &[]);
			compute_pass.dispatch_workgroups(groups_x, groups_y, 1);
		}
	}

	/// Read a GPU buffer back into a Rust vector.
	///
	/// This blocks until the GPU work is complete, maps the readback buffer, copies the
	/// mapped bytes into `Vec<f32>`, and then unmaps the buffer.
	fn read_buffer(&self, buffer: &wgpu::Buffer) -> Vec<f32> {
		let buffer_slice = buffer.slice(..);
		let (sender, receiver) = std::sync::mpsc::channel();

		buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
			sender.send(result).expect("Failed to send map result");
		});

		self.device
			.poll(wgpu::PollType::Wait {
				submission_index: None,
				timeout: None,
			})
			.expect("Failed to poll WGPU device");

		receiver
			.recv()
			.expect("Failed to receive map result")
			.expect("Failed to map readback buffer");

		let mapped = buffer_slice.get_mapped_range();
		let result = bytemuck::cast_slice(&mapped).to_vec();
		drop(mapped);
		buffer.unmap();

		result
	}
}

/// Create a storage-buffer bind group layout entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: wgpu::ShaderStages::COMPUTE,
		ty: wgpu::BindingType::Buffer {
			ty: wgpu::BufferBindingType::Storage { read_only },
			has_dynamic_offset: false,
			min_binding_size: None,
		},
		count: None,
	}
}

/// Create a uniform-buffer bind group layout entry.
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: wgpu::ShaderStages::COMPUTE,
		ty: wgpu::BindingType::Buffer {
			ty: wgpu::BufferBindingType::Uniform,
			has_dynamic_offset: false,
			min_binding_size: None,
		},
		count: None,
	}
}

/// Bind an entire GPU buffer at a specific binding slot.
fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
	wgpu::BindGroupEntry {
		binding,
		resource: buffer.as_entire_binding(),
	}
}

/// Create and initialize a GPU storage buffer from a Rust slice.
fn create_storage_buffer<T: bytemuck::Pod>(
	device: &wgpu::Device,
	label: &str,
	data: &[T],
	extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
	device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some(label),
		contents: bytemuck::cast_slice(data),
		usage: wgpu::BufferUsages::STORAGE | extra_usage,
	})
}

/// Create a CPU-readable buffer used for final GPU result readback.
fn create_readback_buffer(device: &wgpu::Device, byte_size: u64) -> wgpu::Buffer {
	device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("readback buffer"),
		size: byte_size,
		usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
		mapped_at_creation: false,
	})
}
