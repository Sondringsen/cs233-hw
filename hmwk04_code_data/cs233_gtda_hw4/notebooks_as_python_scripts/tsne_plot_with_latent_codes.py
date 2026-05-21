# %%
import numpy as np
import os.path as osp 
from sklearn.manifold import TSNE   # Students: you can use this to extract the TSNE
from cs233_gtda_hw4.in_out.plotting import plot_2d_embedding_in_grid_greedy_way


# %%
random_seed = 42 # Students: use THIS seed 
                 # IF you use sklearn's TSNE with default parameters.

# %%
## LOAD DATA

# %%
## Load latent codes
vanilla_ae_emb_file = '../data/out/pc_ae_latent_codes.npz'
data = np.load(vanilla_ae_emb_file) # Students: we assume you used np.savez in the above directory
latent_codes = data['latent_codes'] # to save the embeddings
test_names = data['test_names']

# %%
## Load images of test models (Students FIRST unzip the corresponding images.zip)
im_files = []
top_im_dir = '../data/images'
for name in test_names:
    im_file = osp.join(top_im_dir, name + '.png')
    assert osp.exists(im_file)
    im_files.append(im_file)

# %%
# TODO: Students get the TSNE embedding
# tsne_lcodes = None


# %%
# Students feel free to play with the big_dim, small_dim to get different plots.
plot_2d_embedding_in_grid_greedy_way(tsne_lcodes, im_files, big_dim=2000, small_dim=40, 
                                     save_file='../data/out/vanilla_ae_test_pc_tsne.png', 
                                     transparent_pngs=True);


