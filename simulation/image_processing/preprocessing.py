import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def load_image_rgb(image_path, shape, contrast=1.0):
	"""
	Load an image, convert it to grayscale, resize it, and normalize pixel values.

	Parameters:
		image_path	-> path to the input vessel image
		shape		-> target simulation grid shape as (rows, columns)
		contrast	-> contrast enhancement factor applied before conversion to array

	Returns:
		image_array	-> grayscale image array normalized from 0 to 1

	Notes:
		The image is resized to match the simulation domain so that each image pixel
		corresponds directly to one tissue grid cell.

		The image is converted to grayscale because the current segmentation pipeline
		uses pixel intensity rather than color channels. Darker pixels are later
		interpreted as vessel-like regions after thresholding.

		Normalization maps raw 8-bit image values into:

			0 -> black
			1 -> white

		This makes the thresholding functions independent of the original image's
		integer pixel range.
	"""
	image = Image.open(image_path)
	image = image.convert("L")
	image = image.resize((shape[1], shape[0]))

	enhancer = ImageEnhance.Contrast(image)
	image = enhancer.enhance(contrast)

	image_array = np.asarray(
		image,
		dtype=float
	) / 255.0
	return image_array

def threshold_vessels(image):
	"""
	Segment vessel pixels using an automatically computed Otsu threshold.

	Parameters:
		image	-> normalized grayscale image with values from 0 to 1

	Returns:
		mask	-> boolean vessel mask where True values represent vessel pixels

	Notes:
		Otsu thresholding separates the image into two intensity classes by choosing
		the threshold that maximizes between-class variance.

		Because vessels appear darker than the surrounding tissue/background in the
		current images, vessel pixels are selected using:

			mask = image < threshold

		This means pixels below the threshold are treated as vessel structures.
	"""
	threshold = otsu_threshold(image)
	mask = image < threshold
	return mask

def smooth_image(image, radius=1):
	"""
	Smooth an image using a square mean filter.

	Parameters:
		image	-> input image array
		radius	-> number of pixels included around each center pixel

	Returns:
		smoothed	-> image after local averaging

	Notes:
		Each output pixel is replaced by the mean of a square neighborhood:

			window size = 2 * radius + 1

		This can reduce high-frequency noise before thresholding, but it can also
		blur vessel boundaries if the radius is too large.
	"""
	if radius <= 0:
		return image

	padded = np.pad(image, radius, mode="edge")
	smoothed = np.zeros_like(image)

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			window = padded[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
			smoothed[i, j] = np.mean(window)

	return smoothed

def binary_erode(mask, radius=1):
	if radius <= 0:
		return mask

	padded = np.pad(mask, radius, mode="constant", constant_values=False)
	eroded = np.zeros_like(mask, dtype=bool)

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			window = padded[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
			eroded[i, j] = np.all(window)

	return eroded

def binary_dilate(mask, radius=1):
	if radius <= 0:
		return mask

	padded = np.pad(mask, radius, mode="constant", constant_values=False)
	dilated = np.zeros_like(mask, dtype=bool)

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			window = padded[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
			dilated[i, j] = np.any(window)

	return dilated

def binary_open(mask, radius=1):
	return binary_dilate(binary_erode(mask, radius), radius)

def binary_close(mask, radius=1):
	return binary_erode(binary_dilate(mask, radius), radius)

def remove_stray_pixels(mask, min_size=20):
	"""
	Remove small disconnected components from a binary vessel mask.

	Parameters:
		mask		-> boolean vessel mask
		min_size	-> minimum connected-component size allowed to remain

	Returns:
		cleaned	-> boolean mask with small isolated components removed

	Notes:
		The function performs connected-component filtering. It walks through every
		True pixel in the mask and groups neighboring True pixels into components.

		Eight-neighbor connectivity is used, meaning diagonal contact counts as
		connected:

			(-1, -1), (-1, 0), (-1, 1)
			( 0, -1),          ( 0, 1)
			( 1, -1), ( 1, 0), ( 1, 1)

		Components smaller than min_size are treated as segmentation noise and are
		removed from the output mask.

		This is useful for eliminating small dark specks produced by thresholding
		without erasing larger connected vessel structures.
	"""
	cleaned = np.zeros_like(mask, dtype=bool)
	visited = np.zeros_like(mask, dtype=bool)

	rows, cols = mask.shape

	for i in range(rows):
		for j in range(cols):
			if visited[i, j] or not mask[i, j]:
				continue

			component = []
			stack = [(i, j)]
			visited[i, j] = True

			while stack:
				x, y = stack.pop()
				component.append((x, y))

				for dx in (-1, 0, 1):
					for dy in (-1, 0, 1):
						if dx == 0 and dy == 0:
							continue

						nx = x + dx
						ny = y + dy

						if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
							continue

						if visited[nx, ny] or not mask[nx, ny]:
							continue

						visited[nx, ny] = True
						stack.append((nx, ny))

			if len(component) >= min_size:
				for x, y in component:
					cleaned[x, y] = True

	return cleaned

def otsu_threshold(image):
	"""
	Compute an Otsu intensity threshold for a normalized grayscale image.

	Parameters:
		image	-> normalized grayscale image with values from 0 to 1

	Returns:
		threshold	-> intensity threshold normalized from 0 to 1

	Notes:
		Otsu's method searches for the threshold that best separates the image into
		two intensity classes: background and foreground.

		For each possible threshold, the image histogram is split into:

			background -> pixels less than or equal to the threshold bin
			foreground -> pixels greater than the threshold bin

		The selected threshold maximizes the between-class variance:

			variance = weight_background
			         * weight_foreground
			         * (mean_background - mean_foreground)^2

		A larger between-class variance means the two groups are more distinct.
		The final threshold is divided by 255 so it matches the normalized image
		intensity range.
	"""
	hist, _bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))

	total = image.size
	sum_total = np.sum(hist * np.arange(256))

	sum_background = 0
	weight_background = 0
	max_variance = 0
	threshold = 0

	for i in range(256):
		weight_background += hist[i]

		if weight_background == 0:
			continue

		weight_foreground = total - weight_background

		if weight_foreground == 0:
			break

		sum_background += i * hist[i]

		mean_background = sum_background / weight_background
		mean_foreground = (sum_total - sum_background) / weight_foreground

		variance_between = (
			weight_background
			* weight_foreground
			* (mean_background - mean_foreground) ** 2
		)

		if variance_between > max_variance:
			max_variance = variance_between
			threshold = i

	return threshold / 255