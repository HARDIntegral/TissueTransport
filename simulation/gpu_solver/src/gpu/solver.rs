use crate::types::Grid2D;
use wgpu::util::DeviceExt;

const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;

/// GPU-backed coupled oxygen/carbon dioxide gas-exchange solver.
pub struct WgpuGasExchangeSolver {
	grid: Grid2D,
	device: wgpu::Device,
	queue: wgpu::Queue,
	pipeline: wgpu::ComputePipeline,
	bind_group_layout: wgpu::BindGroupLayout,
}
impl WgpuGasExchangeSolver {
	/// Try to create a WGPU solver for coupled O₂/CO₂ gas exchange.
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
				label: Some("gas-exchange device"),
				required_features: wgpu::Features::empty(),
				required_limits: wgpu::Limits::default(),
				experimental_features: Default::default(),
				memory_hints: Default::default(),
				trace: Default::default(),
			})
			.await
			.map_err(|error| format!("Failed to create WGPU device: {error:?}"))?;

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("gas_exchange_step shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gas_exchange_step.wgsl").into()),
		});

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("gas-exchange bind group layout"),
			entries: &[
				storage_entry(0, true),
				storage_entry(1, false),
				storage_entry(2, true),
				storage_entry(3, true),
				storage_entry(4, true),
				storage_entry(5, true),
				uniform_entry(6),
			],
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("gas-exchange pipeline layout"),
			bind_group_layouts: &[Some(&bind_group_layout)],
			immediate_size: 0,
		});

		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("gas-exchange compute pipeline"),
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

	/// Create a coupled gas-exchange solver and panic if GPU setup fails.
	pub async fn new(grid: Grid2D) -> Self {
		Self::try_new(grid)
			.await
			.expect("Failed to initialize WGPU gas-exchange solver")
	}

	/// Return the grid associated with this solver.
	pub fn grid(&self) -> Grid2D {
		self.grid
	}


	/// Run multiple coupled O₂/CO₂ gas-exchange timesteps on the GPU.
	///
	/// Input arrays are packed into `vec2<f32>` buffers before upload:
	/// O₂/CO₂ concentrations, O₂/CO₂ diffusivities, `vmax`/`km`, and vessel
	/// O₂/CO₂ concentrations. The shader then advances both species together so
	/// oxygen consumption and carbon dioxide production stay numerically coupled.
	///
	/// The method uses two packed gas buffers as a double buffer and reads back only
	/// the final packed field after all dispatches complete.
	///
	/// Notes:
	/// Vessel concentrations are currently reset every timestep on the GPU path
	/// (`reset_vessels = true`) to model persistent O₂ sources and CO₂ sinks.
	pub async fn run_gas_exchange_steps(
		&self,
		oxygen_initial: &[f32],
		carbon_dioxide_initial: &[f32],
		diffusivity_o2: &[f32],
		diffusivity_co2: &[f32],
		vmax: &[f32],
		km: &[f32],
		vessel_mask: &[bool],
		vessel_o2: &[f32],
		vessel_co2: &[f32],
		dt: f32,
		co2_yield: f32,
		steps: usize,
	) -> (Vec<f32>, Vec<f32>) {
		let n = self.grid.len();

		assert_eq!(oxygen_initial.len(), n);
		assert_eq!(carbon_dioxide_initial.len(), n);
		assert_eq!(diffusivity_o2.len(), n);
		assert_eq!(diffusivity_co2.len(), n);
		assert_eq!(vmax.len(), n);
		assert_eq!(km.len(), n);
		assert_eq!(vessel_mask.len(), n);
		assert_eq!(vessel_o2.len(), n);
		assert_eq!(vessel_co2.len(), n);

		if steps == 0 {
			return (oxygen_initial.to_vec(), carbon_dioxide_initial.to_vec());
		}

		let packed_gas = oxygen_initial
			.iter()
			.zip(carbon_dioxide_initial.iter())
			.map(|(&o2, &co2)| [o2, co2])
			.collect::<Vec<[f32; 2]>>();

		let packed_diffusivity = diffusivity_o2
			.iter()
			.zip(diffusivity_co2.iter())
			.map(|(&o2, &co2)| [o2, co2])
			.collect::<Vec<[f32; 2]>>();

		let packed_metabolism = vmax
			.iter()
			.zip(km.iter())
			.map(|(&v, &k)| [v, k])
			.collect::<Vec<[f32; 2]>>();

		let packed_vessels = vessel_o2
			.iter()
			.zip(vessel_co2.iter())
			.map(|(&o2, &co2)| [o2, co2])
			.collect::<Vec<[f32; 2]>>();

		let vessel_mask_gpu = vessel_mask
			.iter()
			.map(|value| u32::from(*value))
			.collect::<Vec<u32>>();

		let params = GpuGasExchangeParams::from_grid(self.grid, dt, true, co2_yield);
		let byte_size = std::mem::size_of_val(packed_gas.as_slice()) as u64;

		let gas_a = create_storage_buffer(
			&self.device,
			"gas exchange gas A buffer",
			&packed_gas,
			wgpu::BufferUsages::COPY_SRC,
		);
		let gas_b = self.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("gas exchange gas B buffer"),
			size: byte_size,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
			mapped_at_creation: false,
		});

		let diffusivity_buffer = create_storage_buffer(
			&self.device,
			"gas exchange diffusivity buffer",
			&packed_diffusivity,
			wgpu::BufferUsages::empty(),
		);
		let metabolism_buffer = create_storage_buffer(
			&self.device,
			"gas exchange metabolism buffer",
			&packed_metabolism,
			wgpu::BufferUsages::empty(),
		);
		let vessel_mask_buffer = create_storage_buffer(
			&self.device,
			"gas exchange vessel mask buffer",
			&vessel_mask_gpu,
			wgpu::BufferUsages::empty(),
		);
		let vessel_concentration_buffer = create_storage_buffer(
			&self.device,
			"gas exchange vessel concentration buffer",
			&packed_vessels,
			wgpu::BufferUsages::empty(),
		);
		let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("gas exchange params buffer"),
			contents: bytemuck::bytes_of(&params),
			usage: wgpu::BufferUsages::UNIFORM,
		});

		let readback = create_readback_buffer(&self.device, byte_size);

		let bind_group_a_to_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("gas exchange bind group A to B"),
			layout: &self.bind_group_layout,
			entries: &[
				bind_entry(0, &gas_a),
				bind_entry(1, &gas_b),
				bind_entry(2, &diffusivity_buffer),
				bind_entry(3, &metabolism_buffer),
				bind_entry(4, &vessel_mask_buffer),
				bind_entry(5, &vessel_concentration_buffer),
				bind_entry(6, &params_buffer),
			],
		});
		let bind_group_b_to_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("gas exchange bind group B to A"),
			layout: &self.bind_group_layout,
			entries: &[
				bind_entry(0, &gas_b),
				bind_entry(1, &gas_a),
				bind_entry(2, &diffusivity_buffer),
				bind_entry(3, &metabolism_buffer),
				bind_entry(4, &vessel_mask_buffer),
				bind_entry(5, &vessel_concentration_buffer),
				bind_entry(6, &params_buffer),
			],
		});

		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("gas exchange run steps command encoder"),
			});

		{
			let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
				label: Some("gas exchange compute pass"),
				timestamp_writes: None,
			});
			compute_pass.set_pipeline(&self.pipeline);

			let groups_x = (self.grid.width as u32).div_ceil(WORKGROUP_X);
			let groups_y = (self.grid.height as u32).div_ceil(WORKGROUP_Y);

			for step_idx in 0..steps {
				let bind_group = if step_idx % 2 == 0 {
					&bind_group_a_to_b
				} else {
					&bind_group_b_to_a
				};

				compute_pass.set_bind_group(0, bind_group, &[]);
				compute_pass.dispatch_workgroups(groups_x, groups_y, 1);
			}
		}

		let final_buffer = if steps % 2 == 0 { &gas_a } else { &gas_b };
		encoder.copy_buffer_to_buffer(final_buffer, 0, &readback, 0, byte_size);
		self.queue.submit(Some(encoder.finish()));

		let buffer_slice = readback.slice(..);
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
		let packed_result = bytemuck::cast_slice::<u8, [f32; 2]>(&mapped).to_vec();
		drop(mapped);
		readback.unmap();

		let mut oxygen = Vec::with_capacity(n);
		let mut carbon_dioxide = Vec::with_capacity(n);

		for [o2, co2] in packed_result {
			oxygen.push(o2);
			carbon_dioxide.push(co2);
		}

		(oxygen, carbon_dioxide)
	}
}

/// Scalar timestep parameters shared between Rust and the coupled gas-exchange shader.
///
/// This struct is copied into a uniform buffer and must match the WGSL
/// `StepParams` layout in `gas_exchange_step.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGasExchangeParams {
	pub width: u32,
	pub height: u32,
	pub dx: f32,
	pub dy: f32,
	pub dt: f32,
	pub reset_vessels: u32,
	pub co2_yield: f32,
	pub _pad0: u32,
}

impl GpuGasExchangeParams {
	/// Construct packed shader parameters from grid metadata and timestep settings.
	///
	/// Parameters:
	/// grid           -> Grid dimensions and physical spacing.
	/// dt             -> Explicit timestep size.
	/// reset_vessels  -> Whether vessel cells are overwritten after each step.
	/// co2_yield      -> Carbon dioxide produced per unit oxygen consumed.
	///
	/// Returns:
	/// A `GpuGasExchangeParams` matching the WGSL `StepParams` memory layout.
	pub fn from_grid(grid: Grid2D, dt: f32, reset_vessels: bool, co2_yield: f32) -> Self {
		Self {
			width: grid.width as u32,
			height: grid.height as u32,
			dx: grid.dx,
			dy: grid.dy,
			dt,
			reset_vessels: reset_vessels as u32,
			co2_yield,
			_pad0: 0,
		}
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
