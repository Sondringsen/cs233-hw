import os

import numpy as np
import scipy.optimize
from utils import load_landmark_file, load_mesh_info, visualize_mat, \
    compute_all_pair_normalized_geodesics
from mrf import mrf

# Set 'true' for comparing with mobius matching.

mesh_dir = 'hmwk03_code_data/data_p2/meshes/'

output_dir = 'hmwk03_code_data/outputs_p2/pairwise_matching'
landmark_filepath = 'hmwk03_code_data/data_p2/landmark_vids.txt'

os.makedirs(output_dir, exist_ok=True)

# # Load landmark points
landmark_vids, landmark_names = load_landmark_file(landmark_filepath)
num_landmarks = len(landmark_vids)

# # Load all meshes
template_mesh_filepath = os.path.join(mesh_dir, 'template', 'mesh000.obj')
template_mesh_info = load_mesh_info(template_mesh_filepath, compute_feature=False)

test_mesh_filepaths = [os.path.join(mesh_dir, 'test', fname) for fname in os.listdir(os.path.join(mesh_dir, 'test')) if fname.endswith('.obj')]
num_test_meshes = len(test_mesh_filepaths)
test_mesh_infos = [load_mesh_info(test_mesh_filepath, compute_feature=False)
                   for test_mesh_filepath in test_mesh_filepaths]

C_acc = np.zeros((num_landmarks, num_landmarks))

template_geod_dists = compute_all_pair_normalized_geodesics(template_mesh_info, landmark_vids)

for k in range(num_test_meshes):
    test_mesh_info = test_mesh_infos[k]
    test_name = test_mesh_info['name']
    print(test_name)

    # # Find point correspondences

    # ---- To Do ---- #
    # Solve a MRF problem comparing geodesic distances of point pairs,
    # and compute binary correspondence matrix C.
    #
    # Let 'i' and 'j' indicate i - th(ia, ib) and j - th(ja, jb) landmark
    # point pairs between template and test meshes.
    # 'ia' and 'ja' are points in the template mesh, and
    # 'ib' and 'jb' are points in the test mesh.Then,
    # M(i, j): exp(- | geod(ia, ja) - geod(ib, jb) | _2 ^ 2 / (2 * sigma)),
    # where 'geod' is geodesic distance normalized by the longest distance
    # on the surface.
    # Use sigma = 0.05, and use pairwise normalized geodesic distances
    # stored in 'template_geod_dists' and 'test_geod_dists' below.

    param_sigma = 0.05
    test_geod_dists = compute_all_pair_normalized_geodesics(test_mesh_info, landmark_vids)

    # ---- Fill Here ---- #
    L = num_landmarks

    diff = template_geod_dists[:, np.newaxis, :, np.newaxis] - test_geod_dists[np.newaxis, :, np.newaxis, :]
    M = np.exp(-diff ** 2 / (2 * param_sigma)).reshape(L * L, L * L)

    self_node_mask = np.repeat(np.eye(L, dtype=bool), L, axis=0)
    self_node_mask = np.repeat(self_node_mask, L, axis=1)
    M[self_node_mask] = 0.0

    U = np.random.uniform(0, 1, size=L * L) * 1e-3
    node_indices = np.repeat(np.arange(L), L)
    label_indices = np.tile(np.arange(L), L)

    sol, score, matched_labels = mrf(M, U, node_indices, label_indices, 30, 200)

    C = np.zeros((L, L), int)
    for i, a in enumerate(matched_labels):
        C[i, a] = 1
    per_mesh_acc = float(np.diagonal(C).sum() / num_landmarks)
    bijective = len(set(matched_labels)) == num_landmarks
    print(f'  {test_name}: per-mesh accuracy = {per_mesh_acc:.2f}  bijective={bijective}')
    # -------- $

    C_acc += C

    visualize_mat(C, test_name, os.path.join(output_dir, f'{test_name}.png'))


# Compute overall accuracy.
accuracy = np.sum(np.diagonal(C_acc)) / np.sum(C_acc)
print(f'Overall accuracy = {accuracy:.4f}')

# Save overall frequency
visualize_mat(C_acc, 'All frequency', os.path.join(output_dir, 'all_frequency.png'))
