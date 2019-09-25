import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from more_itertools import windowed
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from skimage.feature import register_translation

from util.util import gkern

image_fps = [
    f"/home/maksim/data/DSIAC/dsiac_mad_tiffs/cegr01923_0011/{30 * i}.tiff"
    for i in range(25, 27)
]
images = [np.array(Image.open(image_fp).convert(mode="L")) for image_fp in image_fps]

# compute shifts between images
shifts, *_ = register_translation(images[0], images[0])
for image1, image2 in windowed(images, 2):
    shift, *_ = register_translation(image1, image2)
    shifts = np.vstack((shifts, shift))
_shifts = shifts
shifts = np.cumsum(shifts, axis=0).astype(int)

# how many unique shifts? if less than number of images then we have overlap. what to do? average?
k = len({*map(tuple, shifts)})
print(k, len(image_fps))

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

# odd multiple of lr_pixel width
obs_window = 3
dy, dx = lr_pixel.shape
wy, wx = obs_window * dy, obs_window * dx
num_lr_pixels = obs_window ** 2 * k

# find where the sample pixels are on the hr grid
observation = hr[:wy, :wx]
xx, yy = np.nonzero(observation)

estim_window_shift = ((obs_window - 1) // 2) * np.array([dy, dx])
estim_window_xx, estim_window_yy = (
    np.indices(lr_pixel.shape) + estim_window_shift[:, np.newaxis, np.newaxis]
)

# compute distances between sample pixels on hr grid
lr_pixel_coords = np.array([xx, yy]).T
estim_window_coords = np.array([estim_window_xx.squeeze(), estim_window_yy.squeeze()]).T
lr_pixel_dists = cdist(lr_pixel_coords, lr_pixel_coords)
coord_dist_idx = {coord: i for i, coord in enumerate(zip(*lr_pixel_coords.T))}

estim_window_lr_pixel_dists = cdist(estim_window_coords, lr_pixel_coords)
estim_window_coords_dist_idx = {
    coord: i for i, coord in enumerate(zip(*estim_window_coords.T))
}

# sample parametric model for cross correlation
blur_kernel = gkern(5, 1.5)
sigma_n = sigma_d = 1
rho = 0.75
max_dist = np.max(lr_pixel_dists)
rs = np.linspace(0, max_dist, 10000)
r_dd = sigma_d ** 2 * np.power(rho, rs)
r_df = interp1d(rs, gaussian_filter1d(r_dd, 1.5), axis=0)
r_ff = interp1d(rs, gaussian_filter1d(gaussian_filter1d(r_dd, 1.5), 1.5), axis=0)

R = r_ff(lr_pixel_dists) + sigma_n ** 2 * np.eye(*lr_pixel_dists.shape)
P = r_df(estim_window_lr_pixel_dists).T
W = np.linalg.pinv(R) @ P

plt.imshow(R)
plt.show()
