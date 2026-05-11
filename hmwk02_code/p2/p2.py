import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from pathlib import Path


BASE_PLOT_DIR = "hmwk02_code/plots"

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


def plot2d(data, title=None, filename: str | None = None):
	"""
	Visualizes 2D points
	Input:
		data: (N, 2) 2D point coordinates
	"""
	plt.scatter(data[:, 0], data[:, 1])
	plt.axis('equal')
	plt.title(title)
	if filename: 
		plt.savefig(BASE_PLOT_DIR / Path(filename))
		plt.close()
	else:
		plt.show()

def problem2_1():
	n = 100
	u = np.random.uniform(0, 2*np.pi, size=n)
	data = np.array([np.cos(u), np.sin(u)]).T

	plot2d(data, title='Sampling of unit circle in the plane', filename='sampling_circle.png')
	dgms = ripser(data)["dgms"]
	plot_diagrams(dgms)
	plt.savefig(BASE_PLOT_DIR / Path('sample_circle_bd.png'))
	plt.close()

def problem2_2():
	n = 100
	u = np.random.uniform(0, 2*np.pi, size=2*n)

	# Fully-overlapping:
	circle1 = np.array([np.cos(u[:n]), np.sin(u[:n])]).T
	circle2 = np.array([np.cos(u[n:]), np.sin(u[n:])]).T
	circles = np.concatenate((circle1, circle2))
	circles_noise = circles + 0.2*np.random.normal(size=(2*n, 2))

	plot2d(circles, title='Fully overlapping circles', filename='fully_overlapping.png')
	dgms = ripser(circles)["dgms"]
	plot_diagrams(dgms)
	plt.savefig(BASE_PLOT_DIR / Path('fully_overlapping_bd.png'))
	plt.close()

	plot2d(circles_noise, title='Fully overlapping circles with noise', filename='fully_overlapping_noise.png')
	dgms = ripser(circles_noise)["dgms"]
	plot_diagrams(dgms)
	plt.savefig(BASE_PLOT_DIR / Path('fully_overlapping_noise_bd.png'))
	plt.close()


	# Overlapping:
	circle1 = np.array([np.cos(u[:n]), np.sin(u[:n])]).T
	circle2 = np.array([np.cos(u[n:]), np.sin(u[n:])]).T + [1, 1]
	circles = np.concatenate((circle1, circle2))
	circles_noise = circles + 0.2*np.random.normal(size=(2*n, 2))

	plot2d(circles, title='Overlapping circles', filename='overlapping.png')
	dgms = ripser(circles)["dgms"]
	plot_diagrams(dgms)
	plt.savefig(BASE_PLOT_DIR / Path('overlapping_bd.png'))
	plt.close()

	plot2d(circles_noise, title='Overlapping circles with noise', filename='overlapping_noise.png')
	dgms = ripser(circles_noise)["dgms"]
	plot_diagrams(dgms, )
	plt.savefig(BASE_PLOT_DIR / Path('overlapping_noise_bd.png'))
	plt.close()


	# Non-overlapping:
	circle1 = np.array([np.cos(u[:n]), np.sin(u[:n])]).T
	circle2 = np.array([np.cos(u[n:]), np.sin(u[n:])]).T + [2, 2]
	circles = np.concatenate((circle1, circle2))
	circles_noise = circles + 0.2*np.random.normal(size=(2*n, 2))

	plot2d(circles, title='Non-overlapping circles', filename='non_overlapping.png')
	dgms = ripser(circles)["dgms"]
	plot_diagrams(dgms)
	plt.savefig(BASE_PLOT_DIR / Path('non_overlapping_bd.png'))
	plt.close()

	plot2d(circles_noise, title='Non-overlapping circles with noise', filename='non_overlapping_noise.png')
	dgms = ripser(circles_noise)["dgms"]
	plot_diagrams(dgms)
	plt.savefig(BASE_PLOT_DIR / Path('non_overlapping_noise_bd.png'))
	plt.close()

def problem2_3():
	n = 100
	u = np.random.uniform(0, 2*np.pi, size=n)
	background = np.random.uniform(-1, 1, size=(n, 2))
	data = np.array([np.cos(u), np.sin(u)]).T
	data = np.concatenate((data, background))

	plot2d(data, title='Circle with background noise', filename='circle_background.png')

def problem2_4():
	k = 5
	n = 100
	u = np.random.uniform(0, 2*np.pi, size=n)
	background = np.random.uniform(-1, 1, size=(n, 2))
	data = np.array([np.cos(u), np.sin(u)]).T
	data = np.concatenate((data, background))

	distance = np.linalg.norm(data[:, None] - data[None, :], axis=-1)
	
	xgood, xbad = codensity(data, distance, k, n)

	_, axes = plt.subplots(1, 3, figsize=(15, 5))
	for ax, pts, title in zip(axes, [data, xgood, xbad], ['All points', 'Good points', 'Bad points']):
		ax.scatter(pts[:, 0], pts[:, 1])
		ax.set_aspect('equal')
		ax.set_title(title)
	plt.tight_layout()
	plt.savefig(BASE_PLOT_DIR / Path('codensity_filter.png'))
	plt.close()

def problem2_5():
	ks = [1, 2, 3, 4]
	n = 100
	
	_, axs = plt.subplots(1, len(ks), figsize=(8 * len(ks), 6))

	for i, k in enumerate(ks):
		data = np.random.normal(size=(n, k+1))
		norm = np.linalg.norm(data, axis=1, keepdims=True)
		data = data/norm

		dgms = ripser(data, maxdim=ks[-1])["dgms"]
		plot_diagrams(dgms, ax=axs[i])
		axs[i].set_title(f'{k}-sphere')

	plt.tight_layout()
	plt.savefig(BASE_PLOT_DIR / Path('betti_profiles.png'))
	plt.close()

def problem2_6():
	ks = [1, 2, 3]
	n = 100

	fig, axs = plt.subplots(2, len(ks), figsize=(8 * len(ks), 12))

	for s, k in enumerate(ks):
		data = np.random.normal(size=(n, k+1))
		norm = np.linalg.norm(data, axis=1, keepdims=True)
		data = data/norm

		noise = -2 + 4*np.random.randn(n, k + 1)
		data = np.concatenate((data, noise))

		distance = np.linalg.norm(data[:, None] - data[None, :], axis=-1)

		dgms_before = ripser(data, maxdim=ks[-1])["dgms"]
		plot_diagrams(dgms_before, ax=axs[0, s])
		axs[0, s].set_title(f'{k}-sphere (before cleaning)')

		xgood, _ = codensity(data, distance, k, n)

		dgms = ripser(xgood, maxdim=ks[-1])["dgms"]
		plot_diagrams(dgms, ax=axs[1, s])
		axs[1, s].set_title(f'{k}-sphere (after cleaning)')

	fig.tight_layout()
	fig.savefig(BASE_PLOT_DIR / Path('betti_profiles_noise.png'))
	plt.close(fig)
	

def main():
	pass
	# -----------------------------------------------------------------------
	# 1. Sample from a unit circle.
	# -----------------------------------------------------------------------
	# problem2_1()

	# -----------------------------------------------------------------------
	# 2. Sample from a pair of circles with Gaussian noise.
	#    Consider overlapping and non-overlapping cases.
	# -----------------------------------------------------------------------
	# problem2_2()

	# -----------------------------------------------------------------------
	# 3. Sample from a noisy circle and add background noise from a square film.
	# -----------------------------------------------------------------------
	# problem2_3()

	# -----------------------------------------------------------------------
	# 4. Denoise the points in 3 based on codensity.
	# -----------------------------------------------------------------------
	# problem2_4()
	
	# -----------------------------------------------------------------------
	# 5. Sample n points from k-spheres and try to recover Betti profiles.
	# -----------------------------------------------------------------------
	# problem2_5()

	# -----------------------------------------------------------------------
	# 6. Add uniform Gaussian noise with mean=-2, stdev=4 to points in 5.
	#    Denoise based on codensity and recover Betti profiles.
	# -----------------------------------------------------------------------
	problem2_6()


if __name__ == '__main__':
	main()
