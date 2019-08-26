from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


class DeviceDataLoader(DataLoader):
    def __init__(self, device: torch.device, *args, **kwargs):
        super(DeviceDataLoader, self).__init__(*args, **kwargs)
        self.device = device

    def __iter__(self):
        itr = super(DeviceDataLoader, self).__iter__()
        for x, y in itr:
            yield x.to(self.device), y.to(self.device)


@dataclass
class DataBunch:
    train_dl: DeviceDataLoader
    valid_dl: DeviceDataLoader = None
    test_dl: DeviceDataLoader = None
