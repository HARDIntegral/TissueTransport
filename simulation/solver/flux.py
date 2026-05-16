import numpy as np


def compute_flux(C, D, dx, dy):
	"""
		Computes the diffusive flux between neighboring grid cells using Fick's Law and a 
		face-centered finite-volume approach

		Parameters:
			C	-> a 2D array of solute concentration values
			D	-> a 2D array of effective diffusivity values for the solute at given grid cell
			dx	-> grid spacing in the x direction
			dy	-> grid spacing in the y direction

		Returns:
			Jx	-> flux between left/right neighboring cells
			Jy	-> flux between top/bottom neighboring cells

		Notes:
			Concentration values are stored at cell centers while flux values are computed at the
			faces between neighboring cells. Flux direction and magnitude are determined by the
			concentration gradient and local effective diffusivity.

			For neighboring cells C[i,j] and C[i,j+1]:

				dC/dx = (C[i,j+1] - C[i,j]) / dx
				Jx = -D * dC/dx

			Positive flux corresponds to transport in the positive coordinate direction, while 
			negative flux indicates transport in the opposite direction.
	"""
	rows, cols = C.shape

	Jx = np.zeros((rows, cols - 1))
	Jy = np.zeros((rows - 1, cols))

	# Horizontal fluxes: between cell (i, j) and cell (i, j + 1)
	for i in range(rows):
		for j in range(cols - 1):
			C_diff = C[i, j + 1] - C[i, j]
			D_avg = 0.5 * (D[i, j] + D[i, j + 1])

			Jx[i, j] = -D_avg * C_diff / dx

	# Vertical fluxes: between cell (i, j) and cell (i + 1, j)
	for i in range(rows - 1):
		for j in range(cols):
			C_diff = C[i + 1, j] - C[i, j]
			D_avg = 0.5 * (D[i, j] + D[i + 1, j])

			Jy[i, j] = -D_avg * C_diff / dy

	return Jx, Jy



def compute_flux_divergence(Jx, Jy, dx, dy, shape):
	"""
		Computes dC/dt for each grid cell from the fluxes between neighboring grid cells

		Parameters:
			Jx		-> flux between neighboring grid cells along the x axis
			Jy		-> flux between neighboring grid cells along the y axis
			dx		-> grid spacing in the x direction
			dy		-> grid spacing in the y direction
			shape	-> shape of the concetration gridß
	
		Returns:
			dCdt	-> an array representing the change in concentration of the solute in each
					grid cell
	
		Notes:
			Flux values are stored at the faces between neighboring grid cells while 
			concentration changes (dC/dt) are accumulated at grid cell centers.

			This function computes the net concentration change within each grid cell by
			enforcing conservation of mass: solute leaving one cell through a shared boundary
			enters the neighboring cell through the same boundary.

			Rather than explicitly computing:

				dC/dt = -(dJx/dx + dJy/dy)

			concentration changes are obtained by directly distributing flux across shared
			grid cell faces. This approach is mathematically equivalent to computing the
			divergence of flux while ensuring local conservation of mass.

			Positive outgoing flux decreases concentration in the source grid cell, while the
			same flux increases concentration in the neighboring receiving grid cell.
	"""
	dCdt = np.zeros(shape)

	# Horizontal flux contribution
	for i in range(shape[0]):
		for j in range(shape[1] - 1):
			flux = Jx[i, j]

			dCdt[i, j] -= flux / dx
			dCdt[i, j + 1] += flux / dx

	# Vertical flux contribution
	for i in range(shape[0] - 1):
		for j in range(shape[1]):
			flux = Jy[i, j]

			dCdt[i, j] -= flux / dy
			dCdt[i + 1, j] += flux / dy

	return dCdt