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
	# Face-centered diffusivity values between neighboring cells.
	D_x = 0.5 * (D[:, :-1] + D[:, 1:])
	D_y = 0.5 * (D[:-1, :] + D[1:, :])

	# Concentration gradients across cell faces.
	dCdx = (C[:, 1:] - C[:, :-1]) / dx
	dCdy = (C[1:, :] - C[:-1, :]) / dy

	# Fick's law: J = -D grad(C)
	Jx = -D_x * dCdx
	Jy = -D_y * dCdy

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
	dCdt = np.zeros(shape, dtype=Jx.dtype)

	# Horizontal flux contribution.
	# Jx[:, j] is the flux from cell j to cell j + 1.
	dCdt[:, :-1] -= Jx / dx
	dCdt[:, 1:] += Jx / dx

	# Vertical flux contribution.
	# Jy[i, :] is the flux from row i to row i + 1.
	dCdt[:-1, :] -= Jy / dy
	dCdt[1:, :] += Jy / dy

	return dCdt