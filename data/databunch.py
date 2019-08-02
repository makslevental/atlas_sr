from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class DataBunch:
    train_dl: DataLoader
    valid_dl: DataLoader
    test_dl: DataLoader = None
