import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from skimage import data, img_as_float
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk

matplotlib.rcParams["font.size"] = 8


# [[114.35629928 111.561547   103.1545782 ]] [[4020.22681688 3690.08934962 4068.48610146]]


def equalize_hist(image, nbins=512, mask=None):
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        cdf, bin_centers = exposure.cumulative_distribution(image[mask], nbins)
    else:
        cdf, bin_centers = exposure.cumulative_distribution(image, nbins)
    out = np.interp(image.flat, bin_centers, cdf)
    return out.reshape(image.shape), cdf, bin_centers


def normalize_equalize_hist(image, bin_centers, loc, scale):
    bin_centers = bin_centers / bin_centers.max()
    cdf = norm.ppf(bin_centers, loc=loc, scale=scale)
    cdf[cdf == -np.inf] = cdf[cdf != -np.inf].min()
    cdf[cdf == np.inf] = cdf[cdf != np.inf].max()
    out = np.interp(image.flat, bin_centers, cdf)
    return out.reshape(image.shape), cdf, bin_centers


def dbpn_scale(img, loc=0.8687, scale=0.0005):
    img = exposure.rescale_intensity(img, out_range=(0, 255))

    # Equalization
    img_eq, uniform_cdf, bin_centers = equalize_hist(img)

    # Normal hist equalize
    img_normal_eq, normal_cdf, bin_centers = normalize_equalize_hist(
        img_eq, bin_centers, loc=loc, scale=scale
    )
    return img_normal_eq


def plot_img_and_hist(image, axes, bins=512):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype="bar", color="black")
    ax_hist.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_hist.set_xlabel("Pixel intensity")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, "r")
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def contrast_enhance(img):
    # # Load an example image
    # # img, *_ = linear_scale(a.get_frame_mat(0), rescale=255)

    img = exposure.rescale_intensity(img, out_range=(0, 255))
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(
        img, in_range=(p2, p98), out_range=(0, 255)
    )

    # Equalization
    img_eq, uniform_cdf, bin_centers = equalize_hist(img)

    # Normal hist equalize
    img_normal_eq, normal_cdf, bin_centers = normalize_equalize_hist(
        img_eq, bin_centers
    )

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Local Equalization
    selem = disk(30)
    img_local_eq = rank.equalize(img, selem=selem)

    # Display results
    fig = plt.figure(figsize=(12, 5))
    axes = np.zeros((2, 6), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 6, 1)
    for i in range(1, 6):
        axes[0, i] = fig.add_subplot(2, 6, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 6):
        axes[1, i] = fig.add_subplot(2, 6, 7 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Low contrast image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title("Contrast stretching")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title("Histogram equalization")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title("Adaptive equalization")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_local_eq, axes[:, 4])
    ax_img.set_title("Local equalization")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_normal_eq, axes[:, 5])
    ax_img.set_title("Normal equalization")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()


def histogram_matching():
    img = data.moon()

    img = exposure.rescale_intensity(img, out_range=(0, 255))
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(
        img, in_range=(p2, p98), out_range=(0, 255)
    )
    plt.hist(img)
    plt.show()

    # Equalization
    img_eq, uniform_cdf, bin_centers = equalize_hist(img)
    plt.hist(img_eq)
    plt.show()
    # Normal hist equalize
    img_normal_eq, normal_cdf, bin_centers = normalize_equalize_hist(
        img_eq, bin_centers
    )
    plt.hist(img_normal_eq)
    plt.show()


if __name__ == "__main__":
    # img = data.moon()
    # contrast_enhance(img)
    # # a = ARF("/home/maksim/data/DSIAC/cegr/arf/cegr01939_0001.arf")
    # arf_fps = glob.glob("/home/maksim/data/DSIAC/cegr/arf/*.arf")
    # for arf_fp in sorted(arf_fps)[:30:]:
    #     a = ARF(arf_fp)
    #     img = a.get_frame_mat(0)
    #     contrast_enhance(img)
    #     break
    histogram_matching()
