import glob
import os
import struct
from multiprocessing import Process

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from config.config import DSIAC_ARFS_DIR
from util.util import grouper


class ARF:
    def __init__(self, fp):
        self.fptr, self.frames, self.rows, self.cols = read_arf(fp)

    def get_frame_mat(self, n):
        return self.fptr[n].byteswap()

    def _get_fig(self, n, dpi, title=None):
        im = self.get_frame_mat(n)
        fig = make_arf_frame_fig(im, dpi=dpi)
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

    def save_frame(self, n, fp, title=None, dpi=96):
        fig = self._get_fig(n, dpi, title)
        fig.savefig(fp, dpi=dpi, format="tiff")
        plt.close(fig)

    def save_mat(self, n, fp, dpi=96):
        im = self.get_frame_mat(n)
        vmin, vmax = np.percentile(im, [0.5, 99.5])
        plt.imsave(fp, im, vmin=vmin, vmax=vmax, format="tiff", dpi=dpi, cmap="gray")


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


def make_arf_frame_fig(im, dpi) -> plt.Figure:
    px, py = im.shape
    print(px, py)
    fig = plt.figure(figsize=(py / np.float(dpi), px / np.float(dpi)))
    vmin, vmax = np.percentile(im, [0.5, 99.5])
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], yticks=[], xticks=[], frame_on=False)
    ax.imshow(im, vmin=vmin, vmax=vmax, cmap="gray")

    return fig


def dump_all_frames_all_arfs(arfs_dir: str, dump_dir: str, frame_rate: int = 30):
    def dump_arf(arf_fp):
        arf = ARF(arf_fp)

        basename, _ext = os.path.splitext(os.path.basename(arf_fp))
        frames_dump_dir = os.path.join(dump_dir, basename)
        if not os.path.exists(frames_dump_dir):
            os.mkdir(frames_dump_dir)

        for n in tqdm(range(0, arf.frames, frame_rate), basename):
            arf.save_mat(n, os.path.join(frames_dump_dir, f"{n}.tiff"))

    num_processes = 16
    for arf_fps in grouper(
            sorted(glob.glob(f"{arfs_dir}/*.arf")), num_processes, None
    ):
        processes = []
        for rank in range(num_processes):
            p = Process(target=dump_arf, args=(arf_fps[rank],))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    dump_all_frames_all_arfs(DSIAC_ARFS_DIR, "/tmp")
