import glob
import sys

import numpy as np
from PIL.Image import BICUBIC

from util.util import show_im

np.set_printoptions(threshold=sys.maxsize)
from PIL import Image
from more_itertools import windowed
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from skimage.feature import register_translation


class EstimWindow:
    def __init__(self, obs_window, lr_pixel_shape):
        self.obs_window = obs_window
        self.dy, self.dx = lr_pixel_shape

    @property
    def wy(self):
        return self.obs_window * self.dy

    @property
    def wx(self):
        return self.obs_window * self.dx

    @property
    def estim_window_shift(self):
        return ((self.obs_window - 1) // 2) * np.array([self.dy, self.dx])

    @property
    def estim_window_coords(self):
        return (
            np.indices((self.dy, self.dx)).reshape(2, self.dy * self.dx).T
            + self.estim_window_shift
        )

    def get_estim_window_shift(self, y_estim_steps=0, x_estim_steps=0):
        return np.array([y_estim_steps, x_estim_steps]) * self.estim_window_shift


for i in range(1, 10):
    for j in range(1, 10):
        a = np.indices((i, j))
        b = np.vstack([a[0].flatten(), a[1].flatten()])
        assert (b == a.reshape(2, i * j)).all()


def compute_shifts(images):
    shifts, *_ = register_translation(images[0], images[0])
    for image1, image2 in windowed(images, 2):
        shift, *_ = register_translation(image1, image2)
        shifts = np.vstack((shifts, shift))
    return np.cumsum(shifts, axis=0).astype(int)


def wiener(images, shifts):
    # how many unique shifts? if less than number of images then we have overlap. what to do? average?
    print("number images", len(images))
    print("number shifts", len({*map(tuple, shifts)}))

    # compute expansion padding - ie create space so that all images can be shifted relative to the first one
    # should probably trim edges of zeros here
    top_pad = abs(int(shifts[shifts[:, 0] <= 0, 0].min()))
    bottom_pad = int(shifts[shifts[:, 0] >= 0, 0].max())
    left_pad = abs(int(shifts[shifts[:, 1] <= 0, 1].min()))
    right_pad = int(shifts[shifts[:, 1] >= 0, 1].max())

    # an lr pixel includes all of the room needed to shift 1 pixel from each image into
    # the +1 is room for the first image, relative to which all of the other images are shifted
    lr_pixel = np.zeros((top_pad + 1 + bottom_pad, left_pad + 1 + right_pad))
    hr = np.tile(lr_pixel, images[0].shape)

    for i, (row_shift, column_shift) in enumerate(shifts):
        # think about this wrt to the first image - which has a zero shift
        # ie it will place that image's pixels at the center of each
        # lr pixel.
        # then the rest of the images will be shifted relative to that center
        hr[
            top_pad + row_shift :: (top_pad + 1 + bottom_pad),
            left_pad + column_shift :: (left_pad + 1 + right_pad),
        ] = images[i]
    # compute cross correlation and auto-correlation
    print(hr.shape)
    estim_window = EstimWindow(3, lr_pixel.shape)

    # compute distances between sample pixels on hr grid
    observation = hr[: estim_window.wy, : estim_window.wx]
    ys, xs = np.nonzero(observation)
    sample_pixel_coords = np.array([ys, xs]).T
    lr_pixel_dists = cdist(sample_pixel_coords, sample_pixel_coords)

    # sample parametric model for cross correlation
    sigma_n = sigma_d = 1
    rho = 0.75
    max_dist = np.max(lr_pixel_dists)
    rs = np.linspace(0, max_dist, 10000)
    r_dd = sigma_d ** 2 * np.power(rho, rs)
    r_df = interp1d(rs, gaussian_filter1d(r_dd, 1.5), axis=0)
    r_ff = interp1d(rs, gaussian_filter1d(gaussian_filter1d(r_dd, 1.5), 1.5), axis=0)

    R = r_ff(lr_pixel_dists) + sigma_n ** 2 * np.eye(*lr_pixel_dists.shape)

    y_step, x_step = estim_window.estim_window_shift
    # account for start one shift over and not going all the way to the edge
    n_y_steps = (hr.shape[0] - 2 * y_step) // y_step
    n_x_steps = (hr.shape[1] - 2 * x_step) // x_step

    sr = np.array(hr)
    i = 0
    for y_step in range(n_y_steps):
        for x_step in range(n_x_steps):
            shift = estim_window.get_estim_window_shift(y_step, x_step)
            estim_window_coords = estim_window.estim_window_coords + shift
            obs_sample_pixel_coords = sample_pixel_coords + shift

            estim_window_lr_pixel_dists = cdist(
                estim_window_coords, obs_sample_pixel_coords
            )
            P = r_df(estim_window_lr_pixel_dists).T
            W = np.linalg.pinv(R) @ P
            W /= W.sum(axis=0)

            lr_pixels = sr[obs_sample_pixel_coords[:, 0], obs_sample_pixel_coords[:, 1]]
            sr[estim_window_coords[:, 0], estim_window_coords[:, 1]] = lr_pixels @ W
            i += 1
            if i % 1000 == 0:
                print(f"{i}/{n_x_steps * n_y_steps}")

    return sr


def main():
    # image_fps = [
    #     f"/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/{30 * i}.tiff"
    #     for i in range(12, 14)
    # ]
    shifts = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [-1, 0], [0, -1], [-1, -1], [-1, 1]]
    )
    upscale = 3
    image_fps = sorted(
        glob.glob("/home/maksim/data/yuma/mad_tiffs/avco10425_1034/*.tiff")
    )
    images = [
        Image.open(image_fp).convert(mode="L")
        for image_fp in image_fps[17 : 17 + len(shifts)]
    ]
    w, h = images[0].size
    sr = wiener(
        [np.array(im.resize((w // upscale, h // upscale), BICUBIC)) for im in images],
        shifts,
    )
    show_im(sr, 30, "sr")

    for i, im in enumerate(images):
        hr = np.array(im)
        # show_im(hr, 30, f"hr {i}")

        mr = np.array(
            im.resize((w // upscale, h // upscale), BICUBIC).resize((w, h), BICUBIC)
        )
        show_im(mr, 30, f"mr {i}")
        # print("bicubic", psnr(hr, mr))
        # print("sr", psnr(hr, sr))


if __name__ == "__main__":
    main()
