
# ------------------------
# CS233 HW1
# Problem 1  Starter Code
# ------------------------

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat

extensions = [
    'centerlight', 'glasses', 'happy', 'leftlight',
    'noglasses', 'normal', 'rightlight', 'sad',
    'sleepy', 'surprised', 'wink'
]

# Uncomment this for 'clean' dataset
# extensions = [
#     'centerlight', 'happy',
#     'noglasses', 'normal', 'sad',
#     'sleepy', 'surprised', 'wink'
# ]

# make sure train_inds.txt is located in this directory
data_path = 'hmwk01_code_data/data_p1'

# Load the face data set
Faces = loadmat(os.path.join(data_path, 'YaleFaces'), simplify_cells=True)['Faces']

imH = 159
imW = 159

# example visualization of face
print('Number of unique people:', len(Faces))
sample = Faces[0]['sleepy']
print(sample.shape)
plt.imshow(sample, cmap='gray')
plt.show()

# -----------------------------------------
# TODO: (a) visualize mean faces from train set
# use matplotlib for plotting as above

mean_face = np.zeros((imH, imW))
happy_mean_face = np.zeros((imH, imW))
sad_mean_face = np.zeros((imH, imW))

# Extract the faces into training and test faces
train = []
test = []

train_inds = set()

# load in training indices explicitly to make consistent with MATLAB version of assignment
with open(os.path.join(data_path, 'train_inds.txt')) as f:
    for line in f:
        # MATLAB uses 1-indexing, so we subtract 1 to make it compatible with Python's 0-indexing
        train_inds.add(tuple(map(lambda x: int(x) - 1, line.strip().split(' '))))

mean_counter = 0
happy_mean_counter = 0
sad_mean_counter = 0
for i in range(len(Faces)):
    for j, ext in enumerate(extensions):
        X = Faces[i][ext].astype('float64')
        if (i, j) in train_inds:
            # make 'label' consistent with MATLAB which uses 1-indexing
            train.append({'data': X, 'label': i+1, 'ext': ext})

            # TODO: add code here to compute mean face
            # filter extensitions for happy and sad faces
            #
            if ext == 'happy':
                happy_mean_face += X
                happy_mean_counter += 1
            elif ext == 'sad':
                sad_mean_face += X
                sad_mean_counter += 1
            mean_face += X
            mean_counter += 1


        else:
            test.append({'data': X, 'label': i+1, 'ext': ext})
            
assert len(train_inds) == len(train)
            
mean_face = mean_face/mean_counter
happy_mean_face = happy_mean_face/happy_mean_counter
sad_mean_face = sad_mean_face/sad_mean_counter

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in zip(axes, [mean_face, happy_mean_face, sad_mean_face],
                           ['Mean Face', 'Happy Mean Face', 'Sad Mean Face']):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig('hmwk01_code_data/plots/mean_faces.png')
plt.show()


# Zero-center and put all train set images into a big matrix
M = np.stack(list(map(lambda x: x['data'], train))) - mean_face
M = M.reshape(-1, imW*imH)
print(M.shape)

# -----------------------------------------
# TODO: (b) do PCA (by numpy function np.linalg.svd, i.e. do not use PCA from sklearn or other external libraries) on face
# images and plot the energy (sum of top-k variances/total sum) curve
#

U, S, Vt = np.linalg.svd(M, full_matrices=False)

cumulative_energy_for_50 = 0
squared_sum_s = np.sum(S**2)

counter = 0
while cumulative_energy_for_50 < 50:
    cumulative_energy_for_50 += S[counter]**2 / squared_sum_s * 100
    counter += 1

print(f'# Components to explain more than 50% of the variance {counter}')

cumulative_energy_top25 = np.sum(S[:25]**2) / squared_sum_s * 100
print(f'Percent of variance explained by top 25 components {cumulative_energy_top25}')

cumulative_energy = np.cumsum(S**2) / squared_sum_s * 100

plt.figure()
plt.plot(cumulative_energy)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.title('PCA Energy Curve')
plt.savefig('hmwk01_code_data/plots/pca_energy.png')
plt.show()


# -----------------------------------------
# TODO: (c) show top 25 eigen faces
# Hint: you can create a large matrix (image) and set each
# eigen face image to a portion of it.

big_image = np.zeros((imH*5, imW*5))
for i in range(5):
    for j in range(5):
        # TODO: fill in eigen face below
        big_image[i*imH:(i+1)*imH, j*imW:(j+1)*imW] = Vt[i*5 + j].reshape(imH, imW)


plt.imshow(big_image, cmap='gray')
plt.savefig('hmwk01_code_data/plots/top_25_eigenfaces.png')
plt.show()


# -----------------------------------------
# TODO: (d) face reconstrunction
# reconstruct the first face train[0]['data'] with different
# number of principal components
#

train_idx = 0
k_values = [1,10,20,30,40,50]

fig = plt.figure(figsize=(20,5))

ax = plt.subplot(1, len(k_values) + 1, 1)
ax.set_title('Original')
ax.imshow(train[train_idx]['data'], cmap='gray')

for i, k in enumerate(k_values):
    # TODO: finish the reconstruction using top k eigen faces
    #
    face = M[train_idx]
    weights = face @ Vt[:k].T
    reconstruction = (weights @ Vt[:k] + mean_face.flatten()).reshape(imH, imW)

    ax = plt.subplot(1, len(k_values) + 1, i + 2)
    ax.set_title('Reconstruction, k={}' .format(k))
    ax.imshow(reconstruction, cmap='gray')
plt.tight_layout()
plt.savefig('hmwk01_code_data/plots/reconstruction_clean.png')
plt.show()


# -----------------------------------------
# TODO: (e) face recognition
# For each test example, find best match in training data
#

# Zero-center and put all test set images into a big matrix
M_test = np.stack(list(map(lambda x: x['data'], test))) - mean_face
M_test = M_test.reshape(-1, imW*imH)
k = 25
results = []

# TODO: project all train set images
# using top 25 eigen vectors
#
trainweights = M @ Vt[:k].T

results = []
nearest_train_idxs = []

for test_idx in range(M_test.shape[0]):

    # TODO: project test_idx test set image
    # using top 25 eigen vectors
    #
    testweights = M_test[test_idx] @ Vt[:k].T

    # TODO: nearest neighbor search:
    # find closest train image to this test image
    nearest_train_idx = np.argmin(np.sum((trainweights - testweights)**2, axis=1))
    nearest_train_idxs.append(nearest_train_idx)

    # See if recognition is correct
    if train[nearest_train_idx]['label'] == test[test_idx]['label']:
        result = 1.0
    else:
        result = 0.0

    results.append(result)

# Calculate accuracies
accuracy = np.mean(results)
print('Accuracy: {:.4f}'.format(accuracy))

rightlight_results = [r for r, t in zip(results, test) if t['ext'] == 'rightlight']
leftlight_results = [r for r, t in zip(results, test) if t['ext'] == 'leftlight']
print('Rightlight accuracy: {:.4f}'.format(np.mean(rightlight_results)))
print('Leftlight accuracy:  {:.4f}'.format(np.mean(leftlight_results)))

# Save 4 separate plots so they can be included as subfigures in a writeup
n_test = M_test.shape[0]
n_plots = 4
chunk = int(np.ceil(n_test / n_plots))

for plot_idx in range(n_plots):
    start = plot_idx * chunk
    end = min(start + chunk, n_test)
    n = end - start

    fig, axes = plt.subplots(n, 2, figsize=(4, n * 2))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, test_idx in enumerate(range(start, end)):
        nearest_train_idx = nearest_train_idxs[test_idx]

        axes[i, 0].set_title('Query (ID={})'.format(test[test_idx]['label']), fontsize=8)
        axes[i, 0].axis('off')
        axes[i, 0].imshow(test[test_idx]['data'], cmap='gray')

        axes[i, 1].set_title('NN (ID={})'.format(train[nearest_train_idx]['label']), fontsize=8)
        axes[i, 1].axis('off')
        axes[i, 1].imshow(train[nearest_train_idx]['data'], cmap='gray')

    plt.tight_layout()
    plt.savefig('hmwk01_code_data/plots/classification_{}.png'.format(plot_idx + 1))
    plt.show()

# -----------------------------------------
# TODO: (f) recognition for non-face image
#

# Now load three images and project them to both the eigenfaces,
# and to the orthogonal complement
k = 50
test_imgs = ['face1.jpg', 'face2.jpg', 'nonface1.jpg']


fig, axes = plt.subplots(len(test_imgs), 2, figsize=(5, len(test_imgs) * 2.5))

for j, test_img in enumerate(test_imgs):
    im = plt.imread(os.path.join(data_path, test_img))

    sampleImH, sampleImW = im.shape

    # TODO: use top-k eigen faces to reconstruct the image
    #
    im_flat = im.flatten().astype('float64') - mean_face.flatten()
    weights = im_flat @ Vt[:k].T
    recon_im = (weights @ Vt[:k] + mean_face.flatten()).reshape(sampleImH, sampleImW)

    axes[j, 0].set_title('Original ({})'.format(test_img))
    axes[j, 0].axis('off')
    axes[j, 0].imshow(im, cmap='gray')

    axes[j, 1].set_title('Reconstruction')
    axes[j, 1].axis('off')
    axes[j, 1].imshow(recon_im, cmap='gray')

    # TODO: show the original and reconstructed image's
    # relative norm difference
    #
    norm_diff = np.linalg.norm(im.flatten() - recon_im.flatten()) / np.linalg.norm(im.flatten())

    print('%s = %f' % (test_img, norm_diff))

plt.tight_layout()
plt.savefig('hmwk01_code_data/plots/nonface_reconstruction.png')
plt.show()

