import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from domain import TissueDomain
from solver import explicit_step
from species import Oxygen
from image_processing import load_image_rgb, threshold_vessels, remove_stray_pixels, smooth_image

 # Simulation domain setup
shape = (250, 250)
scale = (1e-3, 1e-3)
test_domain = TissueDomain(shape, scale)
test_domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
test_domain.set_initial_concentration(0.0)

 # Load and preprocess vessel image into a binary mask
image = load_image_rgb(
	"blood_vessel_network_images/structure3.png",
	shape,
	5.0
)
image = smooth_image(image, 1)
mask = threshold_vessels(image)
mask = remove_stray_pixels(mask, 2)

 # Assign vessels as fixed oxygen sources and set tissue metabolism
test_domain.set_vessel_mask(mask, 1.0)
test_domain.set_uniform_consumption(vmax=0.05, km=0.05)

oxygen = Oxygen()

plt.ion()

 # Live concentration visualization during simulation
fig, ax = plt.subplots()

concentration_plot = ax.imshow(
	test_domain.concentration,
	vmin=0,
	vmax=1.0
)

ax.imshow(mask, cmap="Reds", alpha=0.4)

plt.colorbar(concentration_plot, ax=ax, label="O2 concentration")
ax.set_title("Oxygen diffusion | step 0")

frames = []
frame_steps = []

 # Main reaction-diffusion loop
for step in tqdm(range(1001), desc="Main simulation"):
	C_new = explicit_step(test_domain, oxygen, dt=0.01)
	test_domain.concentration = C_new

	if step % 10 == 0:
		concentration_plot.set_data(test_domain.concentration)
		ax.set_title(f"Oxygen diffusion | step {step}")
		plt.pause(0.001)

		frames.append(test_domain.concentration.copy())
		frame_steps.append(step)

plt.ioff()

 # Save recorded frames as a GIF
fig_gif, ax_gif = plt.subplots()
gif_plot = ax_gif.imshow(
	frames[0],
	vmin=0,
	vmax=1.0
)
ax_gif.imshow(mask, cmap="Reds", alpha=0.4)
plt.colorbar(gif_plot, ax=ax_gif, label="O2 concentration")


def update_gif(frame_index):
	gif_plot.set_data(frames[frame_index])
	ax_gif.set_title(f"Oxygen diffusion | step {frame_steps[frame_index]}")
	return [gif_plot]


animation = FuncAnimation(
	fig_gif,
	update_gif,
	frames=len(frames),
	interval=50,
	blit=False
)

animation.save(
	"oxygen_diffusion.gif",
	writer=PillowWriter(fps=20)
)
plt.close(fig)
plt.close(fig_gif)

# Compare steady-state oxygen distributions across
# different Michaelis-Menten consumption parameters

vmax_values = [0.01, 0.05, 0.1]
km_values = [0.01, 0.05, 0.1]

fig_maps, axes = plt.subplots(
	len(km_values),
	len(vmax_values),
	figsize=(12,12)
)

 # Run independent simulations for each parameter combination
for row, km in enumerate(km_values):
	for col, vmax in enumerate(vmax_values):
		domain = TissueDomain(shape, scale)
		domain.set_uniform_properties(epsilon=0.3, tau=2.0, mu=0.001)
		domain.set_initial_concentration(0.0)
		domain.set_vessel_mask(mask, 1.0)
		domain.set_uniform_consumption(vmax=vmax, km=km)

		for _ in tqdm(
			range(1000),
			desc=f"Sweep vmax={vmax}, km={km}",
			leave=False
		):
			C_new = explicit_step(domain, oxygen, dt=0.01)
			domain.concentration = C_new

		ax_map = axes[row, col]
		im = ax_map.imshow(
			domain.concentration,
			vmin=0,
			vmax=1.0
		)
		ax_map.imshow(mask, cmap="Reds", alpha=0.25)
		ax_map.set_title(f"vmax={vmax}\nkm={km}")
		ax_map.axis("off")

fig_maps.suptitle("Michaelis-Menten parameter sweep")
plt.tight_layout()
 # Save sensitivity analysis figure
fig_maps.savefig("parameter_sweep.png", dpi=300)

plt.show(block=False)
plt.pause(0.1)
plt.close('all')