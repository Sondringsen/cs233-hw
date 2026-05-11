import numpy as np
from ripser import ripser
from gudhi import bottleneck_distance


def main():

	# --------------------------------------------------------
	# shape_dict: 3D point coordinates and pairwise point dijkstra distances of M shapes
	# 	'coord': a list of M (N, 3) point coordinates
	# 	'dist': a list of M (N, N) pairwise point dijkstra distances
	# --------------------------------------------------------

	shape_dict = np.load('hmwk02_code/p3/shapes.npz', allow_pickle=True)['data'].item()
	coord_list, dist_list = shape_dict['coord'], shape_dict['dist']
	num_shapes = len(dist_list)

	# ********************************************************
	# TODO:
	# 	Compute the persistence diagram of shapes and their bottleneck distances.
	#   Classify shapes in shapes.npz into two classes of isometric shapes
	#   based on the bottleneck distances between shapes
	# ********************************************************
	tol = 1e-3

	# Precompute all persistence diagrams once
	dgms_list = [ripser(d, maxdim=0, distance_matrix=True)["dgms"] for d in dist_list]

	# Pairwise bottleneck distances
	same_class = {}
	for i in range(num_shapes):
		for j in range(num_shapes):
			if i <= j:
				continue
			dist = bottleneck_distance(dgms_list[i][0], dgms_list[j][0])
			same_class[(i, j)] = dist <= tol
			print(f'i = {i}, j = {j}, dist = {dist}')

	# Assign class labels (label shape 0 as class 0)
	labels = [-1] * num_shapes
	labels[0] = 0
	next_label = 1
	for i in range(num_shapes):
		for j in range(num_shapes):
			if i <= j:
				continue
			if same_class[(i, j)]:
				if labels[i] == -1 and labels[j] == -1:
					labels[i] = labels[j] = next_label
					next_label += 1 # not needed when we only have two classes...
				elif labels[i] == -1:
					labels[i] = labels[j]
				elif labels[j] == -1:
					labels[j] = labels[i]

	print(f'\nClass assignments: {labels}')
	for cls in set(labels):
		members = [i for i, l in enumerate(labels) if l == cls]
		print(f'Class {cls}: shapes {members}')


if __name__ == '__main__':
	main()
