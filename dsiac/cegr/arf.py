import struct

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from config import DSIAC_DATA_DIR


class ARF:
    def __init__(self, fp):
        self.fptr, self.frames, self.rows, self.cols = read_arf(fp)

    def get_frame_mat(self, n):
        return self.fptr[n].byteswap()

    def _get_fig(self, n, title=None):
        im = self.get_frame_mat(n)
        fig = make_arf_frame_fig(im)
        if title is not None:
            ax = fig.axes[0]
            ax.text(
                320,
                15,
                title,
                horizontalalignment="center",
                verticalalignment="center",
                style="italic",
                bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
            )

        return fig

    def show_frame(self, n, title=None):
        self._get_fig(n, title).show()

    def save_frame(self, n, fp, title=None):
        fig = self._get_fig(n, title)
        fig.savefig(fp)
        plt.close(fig)
        Image.open(fp).save(fp, "JPEG")


def read_arf(arf_fp):
    f = open(arf_fp, "rb")
    header = f.read(8 * 4)
    header = list(struct.iter_unpack(">I", header))

    fptr = np.memmap(
        arf_fp,
        dtype="uint16",
        mode="r",
        shape=(header[5][0], header[2][0], header[3][0]),
        offset=32,
    )
    frames, rows, cols = fptr.shape
    return fptr, frames, rows, cols


def make_arf_frame_fig(im, dpi=80) -> plt.Figure:
    px, py = im.shape
    fig = plt.figure(figsize=(py / np.float(dpi), px / np.float(dpi)))
    vmin, vmax = np.percentile(im, [0.5, 99.5])
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], yticks=[], xticks=[], frame_on=False)
    ax.imshow(im, vmin=vmin, vmax=vmax, cmap="gray")

    return fig


if __name__ == "__main__":
    arf = ARF(f"{DSIAC_DATA_DIR}/cegr/arf/cegr02003_0009.arf")
    arf.show_frame(1, "")
    print(arf)
