from functools import partial

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
from multiprocessing import Pool

from sklearn.preprocessing import normalize
from scipy.spatial import distance


from util.util import grouper


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


def make_lr(hr_img, upscale):
    lr_img = rescale(
        hr_img, (1 / upscale, 1 / upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    lr_img = rescale(
        lr_img, (upscale, upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    h, w = min(lr_img.shape[0], hr_img.shape[0]), min(lr_img.shape[1], hr_img.shape[1])
    lr_img = lr_img[:h, :w]
    hr_img = hr_img[:h, :w]
    return lr_img, hr_img


def extract_lr_hr_patches(hr_img, patch_size, upscale):
    if hr_img is None:
        return
    lr_img, hr_img = make_lr(hr_img, upscale)
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


def scaleup_anr(img, upscale, feat_basis, lr_dict, local_projections, patch_size=9):
    mr_img = rescale(img, (upscale, upscale), order=3, multichannel=len(img.shape) == 3)
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
    n_processes = 16
    with Pool(n_processes) as p:
        for i, img_group in enumerate(grouper(imgs, n_processes)):
            print(f"feature patches for img group {i}/{len(imgs)//n_processes}")
            # lr_img_feat_patches, hr_img_patches = extract_lr_hr_patches(
            #     img, patch_size, upscale
            # )
            res = list(
                filter(
                    None,
                    p.map(
                        partial(
                            extract_lr_hr_patches,
                            patch_size=patch_size,
                            upscale=upscale,
                        ),
                        img_group,
                    ),
                )
            )
            lr_img_feat_patches = np.concatenate([x for x, _ in res], axis=0)
            hr_img_patches = np.concatenate([y for _, y in res], axis=0)
            if lr_feat_patches is None:
                lr_feat_patches = lr_img_feat_patches
                hr_patches = hr_img_patches
            else:
                lr_feat_patches = np.append(
                    lr_feat_patches, lr_img_feat_patches, axis=0
                )
                hr_patches = np.append(hr_patches, hr_img_patches, axis=0)

    print(f"pca on {lr_feat_patches.shape})")

    p = PCA(whiten=True).fit(lr_feat_patches)
    feat_basis = p.components_[p.explained_variance_ratio_ > 1e-3].T
    lr_pca_feat_patches = lr_feat_patches @ feat_basis

    print(f"dict on {lr_pca_feat_patches.shape}")
    dico = MiniBatchDictionaryLearning(
        n_components=dict_size,
        alpha=dict_alpha,
        n_jobs=1 if DEBUG else 1,
        verbose=5,
        n_iter=1 if DEBUG else 1000,
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


def train_anr_plus(
    training_imgs,
    dict_size=512,
    regularization_lambda=0.15,
    patch_size=9,
    atom_neighborhood_size=40,
    atom_sample_neighborhood_size=2048,
    num_scales=12,
    scale_factor=0.98,
    upscale=2,
):
    feat_basis, lr_dict, hr_dict = train_anr_dict(
        training_imgs, patch_size, upscale, dict_size, regularization_lambda
    )
    lr_feat_patches = None
    hr_patches = None
    for i in range(num_scales):
        scale = scale_factor ** i
        for img in training_imgs:
            hr_img = rescale(
                img, (scale, scale), order=3, multichannel=len(img.shape) == 3
            )
            lr_img_feat_patches, hr_img_patches = extract_lr_hr_patches(
                hr_img, patch_size, upscale
            )
            if lr_feat_patches is None:
                lr_feat_patches = lr_img_feat_patches
                hr_patches = hr_img_patches
            else:
                lr_feat_patches = np.append(
                    lr_feat_patches, lr_img_feat_patches, axis=0
                )
                hr_patches = np.append(hr_patches, hr_img_patches, axis=0)

    # L2 normalize patches
    lr_pca_feat_patches, norms = normalize(
        lr_feat_patches @ feat_basis, axis=1, norm="l2", return_norm=True
    )
    hr_patches /= norms[:, np.newaxis]

    local_projections = {}
    for i, atom in enumerate(lr_dict):
        distances = distance.cdist(lr_pca_feat_patches, atom[np.newaxis]).squeeze()
        nearest = distances.argsort()[:atom_sample_neighborhood_size]
        # columns of neighbors to match the paper
        lr_neighbors = lr_pca_feat_patches[nearest].T
        hr_neighbors = hr_patches[nearest].T
        local_projections[i] = (
            hr_neighbors
            @ np.linalg.inv(
                lr_neighbors.T @ lr_neighbors
                + regularization_lambda * np.eye(atom_sample_neighborhood_size)
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


def test_anr():
    upscale = 2
    # face_images = datasets.fetch_olivetti_faces().images
    face_images = datasets.fetch_lfw_people().images
    hr_img = face_images[50]
    local_projections, lr_dict, hr_dict, feat_basis = train_anr(
        face_images[: 1 if DEBUG else 500], upscale=upscale
    )

    np.save("local_projections_lfw", local_projections)
    np.save("lr_dict_lfw", lr_dict)
    np.save("hr_dict_lfw", hr_dict)
    np.save("feat_basis_lfw", feat_basis)
    # local_projections = np.load("local_projections.npy", allow_pickle=True).item()
    # lr_dict = np.load("lr_dict.npy", allow_pickle=True)
    # hr_dict = np.load("hr_dict.npy", allow_pickle=True)
    # feat_basis = np.load("feat_basis.npy", allow_pickle=True)

    lr_img = rescale(
        hr_img, (1 / upscale, 1 / upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    print(lr_img.shape)
    sr_img = scaleup_anr(
        lr_img, upscale, feat_basis, lr_dict, local_projections, patch_size=9
    )
    plot_gallery("anr", [lr_img, hr_img, sr_img], 3, 1)
    plt.show()


def test_anr_plus():
    upscale = 2
    face_images = datasets.fetch_olivetti_faces().images
    # face_images = datasets.fetch_lfw_people().images
    hr_img = face_images[50]
    local_projections, lr_dict, hr_dict, feat_basis = train_anr_plus(
        face_images[: 1 if DEBUG else 100], upscale=upscale
    )

    np.save("local_projections_lfw_aplus", local_projections)
    np.save("lr_dict_lfw_aplus", lr_dict)
    np.save("hr_dict_lfw_aplus", hr_dict)
    np.save("feat_basis_lfw_aplus", feat_basis)
    # local_projections = np.load("local_projections.npy", allow_pickle=True).item()
    # lr_dict = np.load("lr_dict.npy", allow_pickle=True)
    # hr_dict = np.load("hr_dict.npy", allow_pickle=True)
    # feat_basis = np.load("feat_basis.npy", allow_pickle=True)

    lr_img = rescale(
        hr_img, (1 / upscale, 1 / upscale), order=3, multichannel=len(hr_img.shape) == 3
    )
    print(lr_img.shape)
    sr_img = scaleup_anr(
        lr_img, upscale, feat_basis, lr_dict, local_projections, patch_size=9
    )
    plot_gallery("anr", [lr_img, hr_img, sr_img], 3, 1)
    plt.show()


if __name__ == "__main__":
    DEBUG = False
    test_anr_plus()
