import numpy as np
from ripser import ripser
from gudhi import bottleneck_distance


def main():

	# --------------------------------------------------------
	# shape_dict: 3D point coordinates and pairwise point dijkstra distances of M shapes
	# 	'coord': a list of M (N, 3) point coordinates
	# 	'dist': a list of M (N, N) pairwise point dijkstra distances
	# --------------------------------------------------------

	shape_dict = np.load('shapes.npz', allow_pickle=True)['data'].item()
	coord_list, dist_list = shape_dict['coord'], shape_dict['dist']
	num_shapes = len(dist_list)

	# ********************************************************
	# TODO:
	# 	Compute the persistence diagram of shapes and their bottleneck distances.
	#   Classify shapes in shapes.npz into two classes of isometric shapes
	#   based on the bottleneck distances between shapes
	# ********************************************************


if __name__ == '__main__':
	main()
