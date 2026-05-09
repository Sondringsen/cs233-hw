from argparse import ArgumentParser
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csgraph
import networkx as nx

# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------


def compute_components(mesh, mask):
	'''
	Input:
		mesh: a Trimesh mesh with vn vertices
		mask: (vn) mask over vertices
	Output:
		labels: (vn), the label of the component that each vertex belongs to;
				set to -1 for standalone vertices (including vertices not in the mask)

	Computes the connected components of the mesh.
		graph vertices: vertices in the mask
		graph edges: edges belonging to faces whose vertices are all in the mask

	Standalone points are ignored
	'''

	triangles = mesh.faces
	tri_mask = np.all(mask[triangles], axis=-1)
	masked_triangles = triangles[np.where(tri_mask)]
	edges = np.stack([masked_triangles.reshape(-1),
					  masked_triangles[:, [1, 2, 0]].reshape(-1)], axis=-1)

	vn = len(mesh.vertices)
	matrix = coo_matrix((np.ones(len(edges), dtype=bool), edges.T), dtype=bool, shape=(vn, vn))
	body_count, labels = csgraph.connected_components(matrix, directed=False)

	_, comp_sizes = np.unique(labels, return_counts=True)
	standalone = np.zeros_like(comp_sizes)
	standalone[comp_sizes == 1] = 1

	labels[standalone[labels] == 1] = -1
	labels[mask == 0] = -1

	_, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)

	return labels


def visualize_mesh_function(mesh, f):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='gray', alpha=0.5, edgecolor='k')

	scatter = ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], c=f, cmap='viridis', s=30)
	fig.colorbar(scatter, label='function value')

	ax.set_aspect('equal', 'box')
	plt.show()


def main(file_path, ints, isize):

	# ------------------------------------------------------
	# Load shape and compute feature function
	# ------------------------------------------------------

	S = trimesh.load(file_path)
	num_vertices = len(S.vertices)
	print(f'Read shape {file_path} with {num_vertices} vertices')

	# As an example, we use the Z coordinate as the feature function
	f = S.vertices[:, 2]

	# Visualize the function on mesh
	visualize_mesh_function(S, f)

	# Normalize the function to be between 0 and 1
	f = (f - np.min(f)) / (np.max(f) - np.min(f))

	print(f'Computing the Mapper graph with {ints} intervals...')

	# ------------------------------------------------------
	# Compute components associated with each interval
	# ------------------------------------------------------

	# total number of components
	num_comp = 0

	# itv_comp_labels[i] stores the per-point component label associated with the i^th interval
	# initialized with -1 to indicate not belonging to any component
	itv_comp_labels = np.ones((ints, num_vertices), dtype=int) * -1

	# comp_itv[j]: the index of the interval that j^th component is associated with
	comp_itv = []

	for i in range(ints):
		Fmin = i / ints
		Fmax = Fmin + isize

		# The indicator function of the interval.
		F = np.zeros(len(f))

		# ********************************************************
		#  TODO: ENTER YOUR CODE HERE -
		#        set all values of F in the pre-image of f
		#        within Fmin and Fmax to be 1.
		# ********************************************************


		# Compute the connected components of the indicator function of the
		# interval

		# ********************************************************
		# TODO: ENTER YOUR CODE HERE -
		#       to obtain connected components of F in shape S.
		#       For this you can use compute_components
		#       the result should be stored in the variable comp_labels
		#       that is used later
		# ********************************************************

		comp_labels = ...

		# renumber the components
		num_new_comp = np.max(comp_labels) + 1
		comp_labels[comp_labels >= 0] = comp_labels[comp_labels >= 0] + num_comp
		num_comp += num_new_comp

		# mark new components to be associated with interval i
		comp_itv += [i for _ in range(num_new_comp)]

		# update the component labels to itv_comp_labels
		itv_comp_labels[i] = comp_labels

	# ------------------------------------------------------
	# Build the graph of components
	# ------------------------------------------------------

	# Adjacency matrix between all components
	comp_graph = np.zeros((num_comp, num_comp))

	# ********************************************************
	# TODO: ENTER YOUR CODE HERE -
	#       Fill in the adjacency matrix comp_graph between all components.
	#       Note that every component associated with i^th interval
	#       can only be adjacent to a component associated with either
	#       (i-1)^th or (i+1)^th interval.
	#       Given point j associated with i^th component itv_comp_labels[i, j]
	#       and (i + 1)^th component itv_comp_labels[i + 1, j], we have
	#       itv_comp_labels[i, j] and itv_comp_labels[i + 1, j]
	#       adjacent to each other.
	# ********************************************************



	# ------------------------------------------------------
	# Visualization
	# ------------------------------------------------------

	# label components by their associated interval
	node_labels = {i: comp_itv[i] + 1 for i in range(len(comp_itv))}

	G = nx.from_numpy_array(comp_graph)
	pos = nx.kamada_kawai_layout(G)

	nx.draw_networkx(G, pos, labels=node_labels, with_labels=True)

	plt.show()



def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--file_path', type=str, default='shapes/victoria0.off',
						help='path to the OFF shape file')
	parser.add_argument('--ints', type=int, default=10,
						help='number of intervals to use')
	return parser.parse_args()


if __name__ == '__main__':

	args = parse_args()

	# By default we take the size of each interval to be double the gap
	# between intervals, to make sure that intervals overlap.
	isize = 2. / args.ints - 1e-6

	main(args.file_path, args.ints, isize)