import numpy
import torch
from dsiac.cegr.arf import ARF
from torch.utils.data import Dataset

from util.util import linear_scale


def torch_mad_normalize(x, scale=1.4826):
    med = torch.median(x)
    mad = scale * torch.median(torch.abs(x - med))
    return (x - torch.median(x)) / mad


class ARFDataset(Dataset):
    def __init__(self, arf_fp, rescale=1, bias=0):
        self.rescale = rescale
        self.bias = bias
        self.arf = ARF(arf_fp)

    def __len__(self):
        return self.arf.n_frames

    def __getitem__(self, idx):
        frame = self.arf.get_frame_mat(idx)
        frame, vmin, vmax = linear_scale(frame, rescale=self.rescale, bias=self.bias)
        frame = torch.from_numpy(frame.astype(numpy.float32))
        frame = torch.stack(3 * [frame])
        return frame, vmin, vmax
