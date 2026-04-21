# ------------------------
# CS233 HW1
# Problem 2  Starter Code
# ------------------------

import numpy as np
import os
from scipy.io import loadmat
import scipy.linalg


def multiview_CCA_train(XtX, CCAdim):
    '''
    Computing CCA transformation matrices and eigenvalues, as a generalized EV problem
    Input: 
     - XtX is the nViews*nViews list which stores X.T * X
     - CCAdim is the desired CCA dimension
    Output:
     - A contains a list of the trained CCA projection matrices
     - L is a vector storing the eigenvalues
    '''
    nViews = len(XtX)
    dims = np.array([XtX[i][i].shape[0] for i in range(nViews)])

    # ------------------------------------------
    # TODO: Assemble covariance matrices and solve the 
    # generalized eigenvalue problem (use scipy.linalg.eig).
    # Hint: eigenvalues/eigenvectors returned by eig are not in sorted order, so
    #       you must make sure the eigenvalues are in descending order such that the projections
    #       correspond to the ordering from largest to smallest eigenvalues

    # Full block covariance matrix, then separate into cross- and auto-covariance blocks
    X_mat_full = np.block([[XtX[i][j] if j >= i else XtX[j][i].T for j in range(nViews)] for i in range(nViews)])
    X_mat_diagonal = scipy.linalg.block_diag(*[XtX[i][i] for i in range(nViews)])
    X_mat_off_diagonal = X_mat_full - X_mat_diagonal

    eigvals, v = scipy.linalg.eig(X_mat_off_diagonal, X_mat_diagonal)

    # Sort descending by real part, discard imaginary noise
    eigvals = eigvals.real
    v = v.real
    idx = np.argsort(eigvals)[::-1]
    eigvals, v = eigvals[idx], v[:, idx]

    L = eigvals[:CCAdim]
    v_top = v[:, :CCAdim]

    # Split stacked eigenvectors back into per-view projection matrices
    A = []
    offset = 0
    for i in range(nViews):
        A.append(v_top[offset:offset + dims[i], :])
        offset += dims[i]

    return A, L

#
# Train multiview CCA
#

# configuration
featdim = 200 # feature dimensions (of input PCA features)
CCAdim = 500 # desired CCA dimension, note that CCAdim < featdim*nViews

# load database features, which are already PCA transformed into 200 dimensions
data_path = 'hmwk01_code_data/data_p2'
X = loadmat(os.path.join(data_path, 'DatabaseFeature_small.mat'), simplify_cells=True)['X']
nViews = len(X) # number of views
X = [X[i] for i in range(nViews)]
# compute XtX (2D list of size nViews*nViews, only upper triangle and diagonal are non-null since symmetric)
XtX = [[None for j in range(i)] + [X[i].T @ X[j] for j in range(i,nViews)]
       for i in range(nViews)]

# -----------------------------------------
# TODO: implement multiview_CCA_train function above
#       to compute CCA matrices and eigenvalues
A, L = multiview_CCA_train(XtX, CCAdim)

# -----------------------------------------
# TODO: compute shared representation Y
#
Y = np.mean(np.stack([X[i] @ A[i] for i in range(nViews)], axis=0), axis=0)

# "magic" weights on the p-th dimension: 1/sqrt(nViews-1-Lp)
# more weights on the top dimensions
CCA_weights = 1.0 / np.sqrt(nViews - 1 - L)
Y = Y * CCA_weights

# discard irrelavant data, only keep Y and CCA_weights
del X, XtX

#
# Nearest-neighbor word query
#
QueryFeature_small_data = loadmat(os.path.join(data_path, 'QueryFeature_small.mat'),
                                  simplify_cells=True)
GT_labels = QueryFeature_small_data['GT_labels'] - 1
Q = [QueryFeature_small_data['Q'][i] for i in range(nViews)]
topRetrievedLabels = []
numQueryToEvaluate = Q[0].shape[0]

k = 5 # top 5 retrieval results
for viewID in range(nViews):
    numQueries = Q[viewID].shape[0]
    query = Q[viewID]

    # query feature CCA projection
    CCACoef_q = query @ A[viewID]

    # perform exact nearest neighbor search
    queryImgFeat = CCACoef_q * CCA_weights
    # distance between query and database items
    distance = np.linalg.norm(queryImgFeat.reshape(numQueries,1,-1) - Y, axis=2)
    sortedIdx = distance.argsort()[:,:k]
    topRetrievedLabels.append(sortedIdx.T)

#
# Evaluate accuracy
#
accuracy = np.stack([(topRetrievedLabels[viewID][0] == GT_labels[:numQueryToEvaluate])
                    .mean() for viewID in range(nViews)], 0)
print(f'Mean Accuracy: {accuracy.mean() * 100:.1f}%')

# Naive baseline: shared representation is the mean of raw view features
# (requires re-loading X since it was deleted above)
X_naive = loadmat(os.path.join(data_path, 'DatabaseFeature_small.mat'), simplify_cells=True)['X']
X_naive = [X_naive[i] for i in range(nViews)]
Y_naive = np.mean(np.stack(X_naive, axis=0), axis=0)

topRetrievedLabels_naive = []
for viewID in range(nViews):
    numQueries = Q[viewID].shape[0]
    queryImgFeat = Q[viewID]
    distance = np.linalg.norm(queryImgFeat.reshape(numQueries, 1, -1) - Y_naive, axis=2)
    sortedIdx = distance.argsort()[:, :k]
    topRetrievedLabels_naive.append(sortedIdx.T)

accuracy_naive = np.stack([(topRetrievedLabels_naive[viewID][0] == GT_labels[:numQueryToEvaluate])
                           .mean() for viewID in range(nViews)], 0)
print(f'Naive Mean Accuracy: {accuracy_naive.mean() * 100:.1f}%')
