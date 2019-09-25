from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from more_itertools import grouper, windowed, intersperse
from scipy.interpolate import NearestNDInterpolator
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

# /home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/720.tiff
# /home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/870.tiff


def make_padding(row_shift, column_shift):
    if row_shift >= 0:
        row_pad = (0, int(row_shift))
    else:
        row_pad = (-int(row_shift), 0)

    if column_shift >= 0:
        column_pad = (0, int(column_shift))
    else:
        column_pad = (-int(column_shift), 0)
    return row_pad, column_pad


def pad_array(a, top_pad, bottom_pad, left_pad, right_pad):
    rs, cs = a.shape
    rows = np.split(a, rs)
    top_padding = np.zeros((top_pad, cs))
    bottom_padding = np.zeros((bottom_pad, cs))
    if len(top_padding):
        rows = [top_padding] + list(intersperse(top_padding, rows))
    if len(bottom_padding):
        rows = list(intersperse(bottom_padding, rows, n=2)) + [bottom_padding]
    m = np.vstack(rows)

    rs, cs = m.shape
    columns = np.split(m, cs, axis=1)
    left_padding = np.zeros((rs, left_pad))
    right_padding = np.zeros((rs, right_pad))
    if len(left_padding):
        columns = [left_padding] + list(intersperse(left_padding, columns))
    if len(right_padding):
        columns = list(intersperse(right_padding, columns, n=2)) + [right_padding]

    m = np.hstack(columns)
    return m


image_fps = [
    f"/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/{30*i}.tiff"
    for i in range(24, 28)
]
images = [np.array(Image.open(image_fp).convert(mode="L")) for image_fp in image_fps]

shifts, *_ = register_translation(images[0], images[0])

for image1, image2 in windowed(images, 2):
    shift, *_ = register_translation(image1, image2)
    shifts = np.vstack((shifts, shift))
_shifts = shifts
shifts = np.cumsum(shifts, axis=0)

k = len({*map(tuple, shifts.astype(int))})

print(k, len(image_fps))
top_pad = abs(int(shifts[shifts[:, 0] < 0, 0].min()))
bottom_pad = int(shifts[shifts[:, 0] >= 0, 0].max())
left_pad = abs(int(shifts[shifts[:, 1] < 0, 1].min()))
right_pad = int(shifts[shifts[:, 1] >= 0, 1].max())
super_pixel = np.zeros((top_pad + 1 + bottom_pad, left_pad + 1 + right_pad))
hr = np.tile(super_pixel, images[0].shape)
# hr[
#     top_pad :: (top_pad + 1 + bottom_pad), left_pad :: (left_pad + 1 + right_pad)
# ] = images[0]

for i, shift in enumerate(shifts):
    row_shift, column_shift = map(int, shift)
    hr[
        top_pad + row_shift :: (top_pad + 1 + bottom_pad),
        left_pad + column_shift :: (left_pad + 1 + right_pad),
    ] = images[i]

# plt.imshow(hr)
# plt.show()

obs_window = 5
dy, dx = super_pixel.shape
kk = len(hr[:dy, :dx][hr[:dy, :dx]>0])
print(k, kk)
wy, wx = obs_window * dy, obs_window * dx

k_i = obs_window * obs_window * k
g_i = np.random.rand(k_i)
W_i = np.random.rand(k_i, (dx*dy))
d_i = W_i.T @ g_i
# W_i = R_i^-1 * P_i
# R_i = rff + np.eye(*rff.shape)

sigma_n = sigma_d = 1
rho = 0.75
# x, y = np.arange(wy) - (wy-1)/2, np.arange(wx) - (wx-1)/2
# xx, yy = np.meshgrid(x, y, sparse=True)
# rdd needs to be zero where there are no pixels so i need hadamard product with a
# particular observation window
observation = hr[:wy,:wx]
xx, yy = np.nonzero(observation)
dists = cdist(np.vstack([xx, yy]).T, np.vstack([xx, yy]).T)
idxs = list(zip(*np.nonzero(observation)))

rdd = sigma_d ** 2 * np.power(rho, np.sqrt(xx ** 2 + yy ** 2)).T
rdd = np.where(observation>0, rdd, 0)
rdd = rdd[:, np.count_nonzero(rdd, axis=0)>0]
rdd = np.hstack([rdd[:, i][rdd[:, i]>0][:, np.newaxis] for i in range(rdd.shape[1])])
h = np.ones((3, 3)) / 9

rdf = convolve2d(rdd, h, "same")
rff = convolve2d(convolve2d(rdd, h, "same"), h, "same")

P = rdf
R = rff + np.eye(*rff.shape)
W = np.linalg.pinv(np.linalg.pinv(R)) * P


print(h)
