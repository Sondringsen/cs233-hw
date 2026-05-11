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

from mrf import mrf



# ---Visualization Utilities--- #
#
# ----------------------------- #

def mat_show(matrix):
	"""
	Visualize matrix
	"""
	plt.matshow(matrix)
	plt.colorbar()
	plt.show()


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
rendering_dir = '../data_p1/100chairs_rendering'
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


# ------ Your code here ------- #
# TODO: implement function hog_extraction
# TODO: compute the representation of each shape 
#       by computing the HoG feature of each view
# You may use skimage.io.imread for loading images
# ----------------------------- #


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



# ----------------------------- #


##
# P1(c): joint shape alignment by MRF

# ------ Your code here ------- #
# TODO: build W_ij matrix that holds affinities for all pairs of shapes
# TODO: build unary vector U
# TODO: call MRF solver from mrf.py to jointly align shapes
# ----------------------------- #
