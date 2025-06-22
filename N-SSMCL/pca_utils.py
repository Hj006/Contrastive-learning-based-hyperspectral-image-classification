# pca_utils.py
import numpy as np
from sklearn.decomposition import PCA

def apply_pca_on_candidate(hsi_data, candidate_truth, num_components=40, use_pca=True):
    c, h, w = hsi_data.shape
    rows, cols = candidate_truth.row, candidate_truth.col
    spectra = hsi_data[:, rows, cols].T  # (N, C)
    coords = list(zip(rows, cols))

    if use_pca:
        pca = PCA(n_components=num_components)
        reduced = pca.fit_transform(spectra)
        result_c = num_components
        final_data = reduced
        var_ratio = pca.explained_variance_ratio_
    else:
        reduced = spectra
        result_c = c
        final_data = reduced
        var_ratio = None

    candidate_data = np.zeros((result_c, h, w), dtype=np.float32)
    for i, (r, c_) in enumerate(coords):
        candidate_data[:, r, c_] = final_data[i]

    return candidate_data, reduced, coords, var_ratio

def apply_pca_train_only(hsi_data, train_truth, num_components=20):
    c, h, w = hsi_data.shape
    rows, cols = train_truth.row, train_truth.col
    train_spectra = hsi_data[:, rows, cols].T

    pca = PCA(n_components=num_components)
    pca.fit(train_spectra)

    reshaped_data = hsi_data.reshape(c, -1).T
    reduced_data = pca.transform(reshaped_data)
    pca_data = reduced_data.T.reshape(num_components, h, w)
    return pca_data, pca.explained_variance_ratio_
