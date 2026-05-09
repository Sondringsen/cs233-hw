import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams


def codensity(x, dist, k, m):
	"""
	Filters out points with the lowest codensity
	Input:
		x: point cloud (n, dim)
		dist: distance matrix (n, n)
		k: k-th nearest neighbor to use in codensity computation
		m: number of points to return
	Output:
		x_good: points with the lowest codensity_k, (m, dim)
		x_bad: other points, (n - m, dim)
	"""
	sorted_dist = np.sort(dist, axis=-1)  # sort each row
	kth_dist = sorted_dist[:, k]
	sorted_x = x[np.argsort(kth_dist)]
	x_good, x_bad = sorted_x[:m], sorted_x[m:]
	return x_good, x_bad


def plot2d(data, title=None):
	"""
	Visualizes 2D points
	Input:
		data: (N, 2) 2D point coordinates
	"""
	plt.scatter(data[:, 0], data[:, 1])
	plt.axis('equal')
	plt.title(title)
	plt.show()


def main():
	pass
	# -----------------------------------------------------------------------
	# 1. Sample from a unit circle.
	# -----------------------------------------------------------------------

	# -----------------------------------------------------------------------
	# 2. Sample from a pair of circles with Gaussian noise.
	#    Consider overlapping and non-overlapping cases.
	# -----------------------------------------------------------------------

	# -----------------------------------------------------------------------
	# 3. Sample from a noisy circle and add background noise from a square film.
	# -----------------------------------------------------------------------

	# -----------------------------------------------------------------------
	# 4. Denoise the points in 3 based on codensity.
	# -----------------------------------------------------------------------

	# -----------------------------------------------------------------------
	# 5. Sample n points from k-spheres and try to recover Betti profiles.
	# -----------------------------------------------------------------------

	# -----------------------------------------------------------------------
	# 6. Add uniform Gaussian noise with mean=-2, stdev=4 to points in 5.
	#    Denoise based on codensity and recover Betti profiles.
	# -----------------------------------------------------------------------


if __name__ == '__main__':
	main()
