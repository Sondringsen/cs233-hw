#
# CS233 HW 3, Problem 1 Starter Code

import numpy as np
import skimage
from skimage import exposure
import skimage.io
import skimage.transform
import skimage.feature
import os
import matplotlib.pyplot as plt
from pathlib import Path

from mrf import mrf

PLOT_SAVE_PATH = Path("hmwk03_code_data/plots/")


# ---Visualization Utilities--- #
#
# ----------------------------- #

def mat_show(matrix):
	"""
	Visualize matrix
	"""
	plt.matshow(matrix)
	plt.colorbar()
	# plt.show()


def save_hog_image(hog_image, title, output_path):
	"""
	save the visualization of the hog feature
	"""
	fig, ax = plt.subplots(1, 1, figsize=(4, 4))

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	ax.axis('off')
	ax.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax.set_title(title)
	plt.savefig(output_path)


# ----------------------------- #

# directory of rendered views of each shape
rendering_dir = 'hmwk03_code_data/data_p1/100chairs_rendering'
# number of chairs in our dataset
num_shapes = 100
# number of rendered views for each shape
num_views = 16

output_path = '../outputs_p1'
os.makedirs(output_path, exist_ok=True)

##
# P1(a): extract HoG features of each input image
#        to represent each shape by V x H dim feature.

def hog_extraction(image):
	"""
		Input: 
			image: H x W 
		Output:
			feature: HoG feature
			hog_image: HoG feature visualization

	"""
	# ------ Your code here ------- #
	# TODO resize the image to 120x120,
	#  extract HoG feature for the image,
	#  return the feature vector and a visualization of the feature
	# You may use skimage.transform.resize and skimage.feature.hog
	# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html
	# ----------------------------- #
	resized_image = skimage.transform.resize(image, (120, 120), anti_aliasing=True)
	fd, hog_image = skimage.feature.hog(
		resized_image,
		visualize=True,
	)
	return fd, hog_image

def get_object_str(obj: int) -> str:
	if obj < 10:
		obj_str = f"00{obj}"
	elif obj < 100:
		obj_str = f"0{obj}"
	elif obj == 100:
		obj_str = f"100"

	return obj_str



# ------ Your code here ------- #
# TODO: implement function hog_extraction
# TODO: compute the representation of each shape 
#       by computing the HoG feature of each view
# You may use skimage.io.imread for loading images
# ----------------------------- #
def concat_hog(obj: int = 1) -> np.ndarray:
	assert obj >= 1 and obj <= 100, "Object must be between 1 and 100"
	obj_str = get_object_str(obj)
	hog_features = []

	for view in range(16):
		image = skimage.io.imread(rendering_dir + f"/{obj_str}_{view}.png", as_gray=True)
		fd, _ = hog_extraction(image)
		hog_features.append(fd)

	hog_features = np.array(hog_features)
	return hog_features

# Visualization
def visualize_p1a():
	for obj in range(1, 4):
		for view in range(3):
			image = skimage.io.imread(rendering_dir + f"/00{obj}_{view}.png", as_gray=True)
			_, hog_image = hog_extraction(image)
			save_hog_image(hog_image, f"Hog image object = {obj}, view = {view}", PLOT_SAVE_PATH / f"hog_image_00{obj}_{view}.png")



# ----------------------------- #

##
# P1(b): compute pairwise dissimilarity matrix between shapes

# ------ Your code here ------- #
# TODO: implement function pairwise_dissimilarity
# TODO: compute the dissimilarity matrix between each pair of shapes
# ----------------------------- #


def pairwise_dissimilarity(feat1, feat2):
	"""
	Compute pairwise dissimilarity matrix between two shapes.
	Input:
	 - feat1, feat2: VxH shape feature. Each row corresponds 
	to the HoG feature of an image view.    
	Output:                                                   
	  - D : VxV dissimilarity matrix                          
	"""
	# ------ Your code here ------- #
	# ----------------------------- #
	V = 16
	norm1_sq = np.sum(feat1 ** 2)
	norm2_sq = np.sum(feat2 ** 2)

	cross = np.array([np.sum(feat1 * np.roll(feat2, delta, axis=0)) for delta in range(V)])

	deltas = (np.arange(V)[np.newaxis, :] - np.arange(V)[:, np.newaxis]) % V
	D = np.sqrt(np.maximum(norm1_sq + norm2_sq - 2 * cross[deltas], 0))

	return D

# Visualization
def visualize_p1b():
	feat1 = concat_hog(1)
	feat2 = concat_hog(2)
	D = pairwise_dissimilarity(feat1, feat2)
	mat_show(D)
	plt.savefig(PLOT_SAVE_PATH / "d_matrix.png")
	
	# Finding best and worst alignment
	print(np.argmax(D))
	best_flat_idx = np.argmin(D)
	b_i, b_j = np.unravel_index(best_flat_idx, D.shape)
	print(b_i, b_j)

	worst_flat_idx = np.argmax(D)
	w_i, w_j = np.unravel_index(worst_flat_idx, D.shape)
	print(w_i, w_j)

	fig, axs = plt.subplots(2, 2)

	image = skimage.io.imread(rendering_dir + f"/00{1}_{b_i}.png")
	axs[0][0].imshow(image)
	image = skimage.io.imread(rendering_dir + f"/00{2}_{b_j}.png")
	axs[0][1].imshow(image)

	image = skimage.io.imread(rendering_dir + f"/00{1}_{w_i}.png")
	axs[1][0].imshow(image)
	image = skimage.io.imread(rendering_dir + f"/00{2}_{w_j}.png")
	axs[1][1].imshow(image)
	fig.suptitle("Best alignment top, worst alignment bottom")
	plt.tight_layout()
	plt.savefig(PLOT_SAVE_PATH / "worst_best_alignment.png")



# ----------------------------- #


##
# P1(c): joint shape alignment by MRF

# ------ Your code here ------- #
# TODO: build W_ij matrix that holds affinities for all pairs of shapes
# TODO: build unary vector U
# TODO: call MRF solver from mrf.py to jointly align shapes
# ----------------------------- #
def joint_shape_alignment():
	sigma = 1.0
	num_shapes = 100
	num_views = 16

	# Data loading to save time
	F = []
	for i in range(1, num_shapes + 1):
		feat = concat_hog(i)
		F.append(feat)
	print("Data loading done")

	W = np.zeros((num_shapes*num_views, num_shapes*num_views), float)
	for i in range(num_shapes):
		for j in range(i, num_shapes):
			D_ij = pairwise_dissimilarity(F[i], F[j])
			W_ij = np.exp(-D_ij / sigma)
			W[i*num_views:(i+1)*num_views, j*num_views:(j+1)*num_views] = W_ij
			if i != j:
				W[j*num_views:(j+1)*num_views, i*num_views:(i+1)*num_views] = W_ij.T
	print("Pairwise affinities done")
	
	U = np.random.uniform(0, 1, size=num_shapes*num_views)


	return mrf(W, U, np.repeat(np.arange(num_shapes), num_views), np.tile(np.arange(num_views), num_shapes), 30, 200)



# Visualize
def visualize_p1c():
	import zipfile
	num_shapes = 100

	sol, score, labels = joint_shape_alignment()

	fig, axs = plt.subplots(3, 3, figsize=(9, 9))
	zip_path = PLOT_SAVE_PATH / "optimal_alignment.zip"

	with zipfile.ZipFile(zip_path, 'w') as zf:
		for i in range(num_shapes):
			obj_str = get_object_str(i + 1)
			view = labels[i]
			img_path = rendering_dir + f"/{obj_str}_{view}.png"

			if i < 9:
				image = skimage.io.imread(img_path)
				ax = axs[i // 3][i % 3]
				ax.imshow(image)
				ax.set_title(f"Obj {i+1}, view {view}")
				ax.axis('off')

			zf.write(img_path, f"obj_{obj_str}_view_{view}.png")

	plt.suptitle("Optimal alignment per shape")
	plt.tight_layout()
	plt.savefig(PLOT_SAVE_PATH / "optimal_alignment.png")
	print(f"Zip saved to {zip_path}")




def main():
	pass
	# 1a
	# hog_features = concat_hog()
	# print(hog_features)
	# visualize_p1a()

	# 1b
	# visualization_p1b()

	# 1c
	# visualize_p1c()

if __name__ == "__main__":
	main()
