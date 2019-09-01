from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_utils.dali import StupidDALIIterator, SRGANFilePipeline
from train.train_srgan_slow import TrainDatasetFromFolder


def compare_with_pytorch_loader():
    batch_size = 1
    world_size = 1
    workers = 1
    local_rank = 0
    crop_size = 88
    upscale_factor = 2

    train_pipe = SRGANFilePipeline(
        batch_size=batch_size,
        num_gpus=world_size,
        num_threads=workers,
        device_id=local_rank,
        crop=crop_size,
        dali_cpu=False,
        upscale_factor=upscale_factor,
        data_dir="/home/maksim/data/",
        file_list="/home/maksim/data/im_losing_my_mind.txt",
        random_shuffle=False,
    )
    train_pipe.build()
    dali_train_loader = StupidDALIIterator(
        pipelines=[train_pipe],
        output_map=["lr_image", "hr_image"],
        size=int(train_pipe.epoch_size("Reader") / world_size),
        auto_reset=False,
    )

    train_set = TrainDatasetFromFolder(
        "/home/maksim/data/im_losing_my_mind",
        crop_size=crop_size,
        upscale_factor=upscale_factor,
    )
    pytorch_train_loader = DataLoader(
        dataset=train_set, num_workers=workers, batch_size=batch_size, shuffle=False
    )

    for (a, b), (c, d) in zip(dali_train_loader, pytorch_train_loader):
        a = a.squeeze(0).squeeze(0).cpu()
        plt.imshow(a, cmap="gray")
        plt.show()
        # b = b.squeeze(0).squeeze(0).cpu()
        # plt.imshow(b, cmap="gray")
        # plt.show()
        # plt.imshow(c.squeeze(0).permute(1, 2, 0).cpu())
        # plt.show()
        # plt.imshow(d.squeeze(0).permute(1, 2, 0).cpu())
        # plt.show()


if __name__ == "__main__":
    compare_with_pytorch_loader()
