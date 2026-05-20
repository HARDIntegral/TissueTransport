import time

import matplotlib.pyplot as plt
import numpy as np

import gpu_solver
from solver_reference.flux import (
	compute_flux,
	compute_flux_divergence
)


RUNS = 5
WARMUP_RUNS = 1
TOTAL_RUNS = RUNS + WARMUP_RUNS
WIDTH = 1000
HEIGHT = 1000
STEPS = 1000

DX = 1.0
DY = 1.0
DT = 0.1

OUTPUT_PATH = "benchmark_performance.png"


# NumPy-optimized Python timestep used as the baseline for comparison.
def explicit_step(
	concentration,
	diffusivity,
	vmax,
	km,
	vessel_mask,
	vessel_concentration,
	dt,
	dx,
	dy
):
	Jx, Jy = compute_flux(
		concentration,
		diffusivity,
		dx,
		dy
	)

	diffusion = compute_flux_divergence(
		Jx,
		Jy,
		dx,
		dy,
		concentration.shape
	)

	consumption = (
		vmax *
		concentration /
		(km + concentration)
	)

	concentration_new = (
		concentration +
		dt * (diffusion - consumption)
	)

	concentration_new[vessel_mask] = (
		vessel_concentration[vessel_mask]
	)

	return concentration_new


# Build the same starting arrays for every run so each solver gets identical input.
def make_benchmark_arrays():
	concentration = np.ones(
		(HEIGHT, WIDTH),
		dtype=np.float32
	)

	diffusivity = np.ones_like(
		concentration
	)

	vmax = np.full_like(
		concentration,
		0.01
	)

	km = np.ones_like(
		concentration
	)

	vessel_mask = np.zeros_like(
		concentration,
		dtype=bool
	)

	vessel_concentration = np.zeros_like(
		concentration
	)

	return (
		concentration,
		diffusivity,
		vmax,
		km,
		vessel_mask,
		vessel_concentration
	)


# Run the NumPy-optimized Python solver for the requested number of steps.
def benchmark_python_reference():
	(
		concentration,
		diffusivity,
		vmax,
		km,
		vessel_mask,
		vessel_concentration
	) = make_benchmark_arrays()

	start = time.perf_counter()

	for _ in range(STEPS):
		concentration = explicit_step(
			concentration,
			diffusivity,
			vmax,
			km,
			vessel_mask,
			vessel_concentration,
			DT,
			DX,
			DY
		)

	return time.perf_counter() - start


# Run the Rust/WGPU solver through the Python NumPy interface.
def benchmark_rust_gpu():
	(
		concentration,
		diffusivity,
		vmax,
		km,
		vessel_mask,
		vessel_concentration
	) = make_benchmark_arrays()

	start = time.perf_counter()

	gpu_solver.run_steps_auto_numpy(
		concentration,
		diffusivity,
		vmax,
		km,
		vessel_mask,
		vessel_concentration,
		DX,
		DY,
		DT,
		STEPS,
		True
	)

	return time.perf_counter() - start


# Collect repeated timings for a benchmark function.
def collect_runs(label, benchmark_function):
	times = []

	for run_idx in range(TOTAL_RUNS):
		elapsed = benchmark_function()

		if run_idx < WARMUP_RUNS:
			print(f"{label} warmup run {run_idx + 1}: {elapsed:.6f} s (ignored)")
			continue

		times.append(elapsed)
		measured_run = run_idx - WARMUP_RUNS + 1
		print(f"{label} run {measured_run}: {elapsed:.6f} s")

	return np.array(times, dtype=np.float64)


# Draw a horizontal bar chart showing every run plus mean and standard deviation.
def plot_benchmark_results(python_times, rust_times):
	labels = [f"Run {idx + 1}" for idx in range(RUNS)]
	y_positions = np.arange(RUNS)
	bar_height = 0.35

	python_mean = python_times.mean()
	rust_mean = rust_times.mean()
	python_std = python_times.std(ddof=1)
	rust_std = rust_times.std(ddof=1)
	speedups = python_times / rust_times
	mean_speedup = python_mean / rust_mean

	fig, ax = plt.subplots(figsize=(12, 7))

	ax.barh(
		y_positions - bar_height / 2,
		python_times,
		height=bar_height,
		label="NumPy-optimized Python"
	)

	ax.barh(
		y_positions + bar_height / 2,
		rust_times,
		height=bar_height,
		label="Rust/WGPU"
	)

	ax.axvline(
		python_mean,
		linestyle=":",
		linewidth=2,
		label=f"NumPy-optimized Python average: {python_mean:.6f} s"
	)

	ax.axvline(
		rust_mean,
		linestyle=":",
		linewidth=2,
		label=f"Rust/WGPU average: {rust_mean:.6f} s"
	)

	ax.axvspan(
		python_mean - python_std,
		python_mean + python_std,
		alpha=0.10
	)

	ax.axvspan(
		rust_mean - rust_std,
		rust_mean + rust_std,
		alpha=0.10
	)

	for idx, (python_time, rust_time) in enumerate(zip(python_times, rust_times)):
		ax.text(
			python_time,
			idx - bar_height / 2,
			f" {python_time:.5f}s",
			va="center"
		)
		ax.text(
			rust_time,
			idx + bar_height / 2,
			f" {rust_time:.5f}s",
			va="center"
		)

	stats_text = (
		f"Grid: {WIDTH}x{HEIGHT} | Steps per run: {STEPS} | Runs: {RUNS}\n"
		f"NumPy-optimized Python: mean = {python_mean:.6f} s, std = {python_std:.6f} s, "
		f"min = {python_times.min():.6f} s, max = {python_times.max():.6f} s\n"
		f"Rust/WGPU: mean = {rust_mean:.6f} s, std = {rust_std:.6f} s, "
		f"min = {rust_times.min():.6f} s, max = {rust_times.max():.6f} s\n"
		f"Average speedup: {mean_speedup:.2f}x | "
		f"Run speedups: {', '.join(f'{speedup:.2f}x' for speedup in speedups)}"
	)

	fig.text(
		0.98,
		0.03,
		stats_text,
		ha="right",
		va="bottom",
		fontsize=9,
		bbox={
			"boxstyle": "round",
			"alpha": 0.12,
			"pad": 0.6
		}
	)

	ax.set_yticks(y_positions)
	ax.set_yticklabels(labels)
	ax.invert_yaxis()
	ax.set_xlabel("Elapsed time per full simulation run (seconds)")
	ax.set_title("NumPy-optimized Python vs Rust/WGPU Solver Performance")
	handles, legend_labels = ax.get_legend_handles_labels()
	fig.legend(
		handles,
		legend_labels,
		loc="lower left",
		bbox_to_anchor=(0.08, 0.03),
		frameon=True
	)
	ax.grid(axis="x", linestyle=":", alpha=0.35)

	fig.tight_layout(rect=(0, 0.22, 1, 1))
	fig.savefig(OUTPUT_PATH, dpi=300)
	plt.show()


print(f"Grid: {WIDTH}x{HEIGHT}")
print(f"Steps: {STEPS}")
print(f"Measured runs per solver: {RUNS}")
print(f"Warmup runs ignored per solver: {WARMUP_RUNS}")
print()

python_times = collect_runs(
	"NumPy-optimized Python",
	benchmark_python_reference
)

print()

rust_times = collect_runs(
	"Rust/WGPU",
	benchmark_rust_gpu
)

print()
print(f"NumPy-optimized Python average: {python_times.mean():.6f} s")
print(f"Rust/WGPU average: {rust_times.mean():.6f} s")
print(f"Average speedup: {python_times.mean() / rust_times.mean():.2f}x")
print(f"Saved benchmark graphic to {OUTPUT_PATH}")

plot_benchmark_results(
	python_times,
	rust_times
)