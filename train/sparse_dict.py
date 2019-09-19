import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from skimage.transform import rescale
from sklearn import datasets
from sklearn.decomposition import MiniBatchDictionaryLearning, PCA
from sklearn.feature_extraction.image import (
    extract_patches_2d,
    reconstruct_from_patches_2d,
)


def extract_features(img):
    # extract first and second order gradient features
    # in both dimensions
    horiz_f1 = np.array([[-1, 0, 1]])
    vert_f1 = horiz_f1.T
    horiz_f2 = np.array([[1, 0, -2, 0, 1]])
    vert_f2 = horiz_f2.T
    a = convolve2d(img, horiz_f1, "same")
    b = convolve2d(img, vert_f1, "same")
    c = convolve2d(img, horiz_f2, "same")
    d = convolve2d(img, vert_f2, "same")
    return a, b, c, d


def extract_feat_patches(img, patch_size):
    feats = extract_features(img)
    feat_patches = np.array(
        [extract_patches_2d(feat, (patch_size, patch_size)) for feat in feats]
    )
    # flatten patches to vectors
    n_patches = feat_patches.shape[1]  # (4, n, patch_size, patch_size)
    feat_patches = feat_patches.transpose(1, 0, 2, 3).reshape(n_patches, -1)
    return feat_patches


def extract_lr_hr_patches(hr_img, patch_size, upscale):
    lr_img = rescale(
        hr_img, (1 / upscale, 1 / upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    lr_img = rescale(
        lr_img, (upscale, upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    # this is so we extact patches from the same place in both hr and lr
    lr_feat_patches = extract_feat_patches(lr_img, patch_size)
    # subtract high freqs from hr_patches? don't know why???
    hr_img -= lr_img
    hr_patches = extract_patches_2d(hr_img, (patch_size, patch_size))
    # subtract mean on a patch by patch basis - i don't know why?
    hr_patches -= np.mean(hr_patches, axis=(1, 2), keepdims=True)
    # flatten patches to vectors
    n_patches = hr_patches.shape[0]
    hr_patches = hr_patches.reshape(n_patches, -1)
    return lr_feat_patches, hr_patches


def norm_patches(lr_patches, hr_patches, threshold=0):
    above_thresh_idxs = np.var(hr_patches, axis=1) > threshold
    hr_patches = hr_patches[above_thresh_idxs]
    lr_patches = lr_patches[above_thresh_idxs]

    l_norm = np.var(lr_patches, axis=1, keepdims=True)
    h_norm = np.var(hr_patches, axis=1, keepdims=True)

    valid_idxs = np.logical_and(l_norm.squeeze() > 0, h_norm.squeeze() > 0)

    hr_patches = hr_patches[valid_idxs] / h_norm[valid_idxs]
    lr_patches = lr_patches[valid_idxs] / l_norm[valid_idxs]

    return lr_patches, hr_patches


def scaleup_anr(img, upscale, feat_basis, lr_dict, local_projections, patch_size=9):
    mr_img = rescale(
        img, (upscale, upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    feat_patches = extract_feat_patches(mr_img, patch_size)
    features = feat_patches @ feat_basis
    nearest_atoms = (features @ lr_dict.T).argmax(axis=1)
    patches = None
    for feat, nearest_atom in zip(features, nearest_atoms):
        patch = (local_projections[nearest_atom] @ feat).reshape(
            (patch_size, patch_size)
        )[np.newaxis]
        if patches is None:
            patches = patch
        else:
            patches = np.concatenate((patches, patch))

    low_freq_patches = extract_patches_2d(mr_img, (patch_size, patch_size))
    return (
        reconstruct_from_patches_2d(patches + low_freq_patches, mr_img.shape) + mr_img
    )


def _show_patches(patches, patch_size):
    n = int(np.sqrt(len(patches)))
    plt.figure(figsize=(4.2, 4))
    for i, patch in enumerate(patches[: n ** 2]):
        plt.subplot(n, n, i + 1)
        if len(patch.shape) == 1:
            patch = patch.reshape((patch_size, patch_size))
        plt.imshow(patch)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def train_anr_dict(imgs, patch_size, upscale, dict_size, dict_alpha):
    lr_feat_patches = None
    hr_patches = None

    for i, img in enumerate(imgs):
        print(f"feature patches for img {i}")
        lr_img_feat_patches, hr_img_patches = extract_lr_hr_patches(
            img, patch_size, upscale
        )
        if lr_feat_patches is None:
            lr_feat_patches = lr_img_feat_patches
            hr_patches = hr_img_patches
        else:
            lr_feat_patches = np.append(lr_feat_patches, lr_img_feat_patches, axis=0)
            hr_patches = np.append(hr_patches, hr_img_patches, axis=0)

    print(f"pca on {lr_feat_patches.shape})")

    p = PCA(whiten=True).fit(lr_feat_patches)
    feat_basis = p.components_[p.explained_variance_ratio_ > 1e-3].T
    lr_pca_feat_patches = lr_feat_patches @ feat_basis

    print(f"dict on {lr_pca_feat_patches.shape}")
    dico = MiniBatchDictionaryLearning(
        n_components=dict_size, alpha=dict_alpha, n_jobs=1, verbose=5, n_iter=1
    ).fit(lr_pca_feat_patches)
    # dico = ApproximateKSVD(n_components=dict_size)
    # dico.fit(lr_pca_feat_patches)

    lr_coefficients = dico.transform(lr_pca_feat_patches)
    lr_dict = dico.components_
    hr_dict = np.linalg.pinv(lr_coefficients) @ hr_patches
    return feat_basis, lr_dict, hr_dict


def train_anr(
    training_imgs,
    dict_size=512,
    regularization_lambda=0.15,
    patch_size=9,
    neighborhood_size=40,
    upscale=2,
):
    feat_basis, lr_dict, hr_dict = train_anr_dict(
        training_imgs, patch_size, upscale, dict_size, regularization_lambda
    )

    # find neigbhors
    # what's faster? kd tree or this?
    nearest = (lr_dict @ lr_dict.T).argsort(axis=0)[::-1][:neighborhood_size, :].T

    # compute local projections using neighborhood
    local_projections = {}
    for i, _atom in enumerate(lr_dict):
        # columns of neighbors to match the paper
        lr_neighbors = lr_dict[nearest[i]].T
        hr_neighbors = hr_dict[nearest[i]].T
        local_projections[i] = (
            hr_neighbors
            @ np.linalg.inv(
                lr_neighbors.T @ lr_neighbors
                + regularization_lambda * np.eye(neighborhood_size)
            )
            @ lr_neighbors.T
        )

    return local_projections, lr_dict, hr_dict, feat_basis


def plot_gallery(title, images, n_col, n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2.0 * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        # vmax = max(comp.max(), -comp.min())
        plt.imshow(comp, cmap=cmap)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.0)


if __name__ == "__main__":
    upscale = 2
    face_images = datasets.fetch_olivetti_faces().images
    hr_img = face_images[0]
    local_projections, lr_dict, hr_dict, feat_basis = train_anr(
        [face_images[0]], upscale=upscale
    )

    np.save("local_projections", local_projections)
    np.save("lr_dict", lr_dict)
    np.save("hr_dict", hr_dict)
    np.save("feat_basis", feat_basis)
    local_projections = np.load("local_projections.npy", allow_pickle=True).item()
    lr_dict = np.load("lr_dict.npy", allow_pickle=True)
    hr_dict = np.load("hr_dict.npy", allow_pickle=True)
    feat_basis = np.load("feat_basis.npy", allow_pickle=True)

    lr_img = rescale(
        hr_img, (1 / upscale, 1 / upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    print(lr_img.shape)
    sr_img = scaleup_anr(
        lr_img, upscale, feat_basis, lr_dict, local_projections, patch_size=9
    )
    plot_gallery("anr", [lr_img, hr_img, sr_img], 3, 1)
    plt.show()
