import numpy
import torch
from dsiac.cegr.arf import ARF
from torch.utils.data import Dataset


def torch_mad_normalize(x, scale=1.4826):
    mad = scale * torch.median(torch.abs(x - torch.median(x)))
    return (x - torch.median(x)) / (mad if mad > 0 else 1)


class ARFDataset(Dataset):
    def __init__(self, arf_fp, transform=None):
        self.arf = ARF(arf_fp)
        self.transform = transform

    def __len__(self):
        return self.arf.n_frames

    def __getitem__(self, idx):
        frame = self.arf.get_frame_mat(idx)
        frame = torch.from_numpy(frame.astype(numpy.float32))
        if self.transform:
            frame = self.transform(frame)

        return frame


# transformed_dataset = ARFDataset(
#     "/home/maksim/data/DSIAC/cegr/arf/cegr01923_0011.arf",
#     transform=transforms.Compose(
#         [Lambda(lambda x: mad_normalize(x)), Lambda(lambda x: torch.clamp(x, -20, 20))]
#     ),
# )
#
# dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=1)
# fig, axs = pyplot.subplots(1, 1, tight_layout=True)
# for i_batch, sample_batched in enumerate(dataloader):
#     break
# all_pixels = mad_normalize(torch.from_numpy(a.fptr.astype(numpy.float32))).numpy().flatten()
# axs.hist(all_pixels, bins=1000)
# fig.show()
#
# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#
#     print(i, sample['image'].size(), sample['landmarks'].size())
#
#     if i == 3:
#         break
