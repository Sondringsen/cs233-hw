import os

import numpy as np
import scipy.optimize
from utils import load_landmark_file, load_mesh_info, visualize_mat


mesh_dir = 'hmwk03_code_data/data_p2/meshes/'

output_dir = 'hmwk03_code_data/outputs_p2/desc_matching'
landmark_filepath = 'hmwk03_code_data/data_p2/landmark_vids.txt'

os.makedirs(output_dir, exist_ok=True)

# # Load landmark points
landmark_vids, landmark_names = load_landmark_file(landmark_filepath)
num_landmarks = len(landmark_vids)

# # Load all meshes
template_mesh_filepath = os.path.join(mesh_dir, 'template', 'mesh000.obj')
template_mesh_info = load_mesh_info(template_mesh_filepath)

test_mesh_filepaths = [os.path.join(mesh_dir, 'test', fname) for fname in os.listdir(os.path.join(mesh_dir, 'test')) if fname.endswith('.obj')]
num_test_meshes = len(test_mesh_filepaths)
test_mesh_infos = [load_mesh_info(test_mesh_filepath) for test_mesh_filepath in test_mesh_filepaths]

C_acc = np.zeros((num_landmarks, num_landmarks))

for k in range(num_test_meshes):
    test_mesh_info = test_mesh_infos[k]
    test_name = test_mesh_info['name']
    print(test_name)

    # # Find point correspondences

    # ---- To Do ---- #
    # Compute binary correspondence matrix C by comparing point
    # descriptors.
    #
    # Use descriptor vectors stored in mesh_info['feature'].
    # Each row is a feature vector for each vertex
    # The vertex IDs of landmark points are stored in landmark_vids
    #
    # For finding best correspondences, we recommend using 'Hungarian algorithm'
    # (https: // en.wikipedia.org / wiki / Hungarian_algorithm).
    # But any reasonable heuristic algorithm is also accepted.

    # ---- FILL HERE - --- $
    test_relevant_vertices = test_mesh_info['feature'][landmark_vids]
    template_relevant_vertices = template_mesh_info['feature'][landmark_vids]
    diff = test_relevant_vertices[:, np.newaxis, :] - template_relevant_vertices[np.newaxis, :, :]
    cost = np.linalg.norm(diff, axis=2)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    C = np.zeros(cost.shape, int)
    C[row_ind, col_ind] = 1

    # -------- $

    C_acc += C

    visualize_mat(C, test_name, os.path.join(output_dir, f'{test_name}.png'))


# Compute overall accuracy.
accuracy = np.sum(np.diagonal(C_acc)) / np.sum(C_acc)
print(f'Overall accuracy = {accuracy:.4f}')

# Save overall frequency
visualize_mat(C_acc, 'All frequency', os.path.join(output_dir, 'all_frequency.png'))
