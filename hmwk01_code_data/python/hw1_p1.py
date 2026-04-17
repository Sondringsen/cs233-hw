
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

# make sure train_inds.txt is located in this directory
data_path = '../data_p1'

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

for i in range(len(Faces)):
    for j, ext in enumerate(extensions):
        X = Faces[i][ext].astype('float64')
        if (i, j) in train_inds:
            # make 'label' consistent with MATLAB which uses 1-indexing
            train.append({'data': X, 'label': i+1, 'ext': ext})

            # TODO: add code here to compute mean face
            # filter extensitions for happy and sad faces
            #

        else:
            test.append({'data': X, 'label': i+1, 'ext': ext})
            
assert len(train_inds) == len(train)
            
mean_face = ... 
happy_mean_face = ...
sad_mean_face = ...

            
# Zero-center and put all train set images into a big matrix
M = np.stack(list(map(lambda x: x['data'], train))) - mean_face
M = M.reshape(-1, imW*imH)
print(M.shape)

# -----------------------------------------
# TODO: (b) do PCA (by numpy function np.linalg.svd, i.e. do not use PCA from sklearn or other external libraries) on face
# images and plot the energy (sum of top-k variances/total sum) curve
#


# -----------------------------------------
# TODO: (c) show top 25 eigen faces
# Hint: you can create a large matrix (image) and set each
# eigen face image to a portion of it.

big_image = np.zeros((imH*5, imW*5))
for i in range(5):
    for j in range(5):
        # TODO: fill in eigen face below
        big_image[i*imH:(i+1)*imH, j*imW:(j+1)*imW] = ...


plt.imshow(big_image, cmap='gray')
plt.show()


# -----------------------------------------
# TODO: (d) face reconstrunction
# reconstruct the first face train[0]['data'] with different
# number of principal components
#

train_idx = 0
k_values = [1,10,20,30,40,50]

fig = plt.figure(figsize=(20,20))

ax = plt.subplot(1, len(k_values) + 1, 1)
ax.set_title('Original')
ax.imshow(train[train_idx]['data'], cmap='gray')

for i, k in enumerate(k_values):
    # TODO: finish the reconstruction using top k eigen faces
    #
    reconstruction = ...
    
    ax = plt.subplot(1, len(k_values) + 1, i + 2)
    ax.set_title('Reconstruction, k={}' .format(k))
    ax.imshow(reconstruction, cmap='gray')
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
trainweights = ...

fig = plt.figure(figsize=(10,70))

for test_idx in range(M_test.shape[0]):
    
    # TODO: project test_idx test set image
    # using top 25 eigen vectors
    #
    testweights = ...

    # TODO: nearest neighbor search: 
    # find closest train image to this test image
    nearest_train_idx = ...
    
    # See if recognition is correct
    if train[nearest_train_idx]['label'] == test[test_idx]['label']:
        result = 1.0
    else:
        result = 0.0
        
    results.append(result)
    
    ax = plt.subplot(M_test.shape[0], 2, 2*test_idx+1)
    ax.set_title('Query (ID={})'.format(test[test_idx]['label']))
    ax.axis('off')
    ax.imshow(test[test_idx]['data'], cmap='gray')
    
    ax = plt.subplot(M_test.shape[0], 2, 2*test_idx+2)
    ax.set_title('Nearest Neighbor (ID={})'.format(train[nearest_train_idx]['label']))
    ax.axis('off')
    ax.imshow(train[nearest_train_idx]['data'], cmap='gray')

# Calculate accuracies
accuracy = np.mean(results)
print('Accuracy: {:.4f}'.format(accuracy))
    
plt.show()

# -----------------------------------------
# TODO: (f) recognition for non-face image
#

# Now load three images and project them to both the eigenfaces,
# and to the orthogonal complement
k = 50
test_imgs = ['face1.jpg', 'face2.jpg', 'nonface1.jpg']

fig = plt.figure(figsize=(10,10))

for j, test_img in enumerate(test_imgs):
    im = plt.imread(os.path.join(data_path, test_img))
    
    sampleImH, sampleImW = im.shape
    
    # TODO: use top-k eigen faces to reconstruct the image
    #
    recon_im = ... 
    
    ax = plt.subplot(len(test_imgs), 2, 2*j+1)
    ax.set_title('Original ({})'.format(test_img))
    ax.axis('off')
    ax.imshow(im, cmap='gray')
    
    ax = plt.subplot(len(test_imgs), 2, 2*j+2)
    ax.set_title('Reconstruction')
    ax.axis('off')
    ax.imshow(recon_im, cmap='gray')

    plt.show()
    
    # TODO: show the original and reconstructed image's
    # relative norm difference
    #
    norm_diff = ...

    print('%s = %f' % (test_img, norm_diff))

