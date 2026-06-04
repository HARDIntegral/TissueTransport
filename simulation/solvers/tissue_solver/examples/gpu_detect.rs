use wgpu::{Backends, Instance, InstanceDescriptor};

fn main() {
	pollster::block_on(run());
}

async fn run() {
	let instance = Instance::new(InstanceDescriptor {
		backends: Backends::all(),
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
		.expect("No GPU adapter found");

	let info = adapter.get_info();

	println!("GPU Name: {}", info.name);
	println!("Backend: {:?}", info.backend);
	println!("Vendor: {}", info.vendor);
	println!("Device: {}", info.device);
	println!("Driver: {}", info.driver);

	let (_device, _queue) = adapter
		.request_device(&wgpu::DeviceDescriptor {
			label: None,
			required_features: wgpu::Features::empty(),
			required_limits: wgpu::Limits::default(),
			experimental_features: Default::default(),
			memory_hints: Default::default(),
			trace: Default::default(),
		})
		.await
		.expect("Failed to create device");

	println!("Device + queue created successfully");
}