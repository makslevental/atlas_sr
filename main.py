import glob
import multiprocessing as mp
import os
from itertools import zip_longest

from tqdm import tqdm

from config import DSIAC_DATA_DIR
from dsiac.cegr.arf import ARF


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def dump_all_frames_all_arfs(dir_path):
    def dump_arf(arf_fp):
        arf = ARF(arf_fp)

        basename, _ext = os.path.splitext(os.path.basename(arf_fp))
        lr_path = os.path.join(dir_path, basename)
        if not os.path.exists(lr_path):
            os.mkdir(lr_path)
        if (
                len([name for name in os.listdir(lr_path) if os.path.isfile(name)])
                == arf.frames
        ):
            return

        for n in tqdm(range(0, arf.frames, 30), basename):
            arf.save_mat(n, os.path.join(lr_path, f"{basename}_{n}.tiff"))

    num_processes = 16
    for arf_fps in grouper(sorted(glob.glob(f"{DSIAC_DATA_DIR}/cegr/arf/*.arf")), 16, None):
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=dump_arf, args=(arf_fps[rank],))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    dump_all_frames_all_arfs("/home/maksim/dev_projects/SRGAN/dsiac_lr_images")
