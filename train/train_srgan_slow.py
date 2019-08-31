from math import log10
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import torch.utils.data
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    ToTensor,
    ToPILImage,
    CenterCrop,
    Resize,
)
from tqdm import tqdm

from metrics.ssim import ssim
from models.SRGAN import Generator, Discriminator, GeneratorLoss


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    )


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([RandomCrop(crop_size), ToTensor()])


def train_lr_transform(crop_size, upscale_factor):
    return Compose(
        [
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ToTensor(),
        ]
    )


def display_transform():
    return Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + "/SRF_" + str(upscale_factor) + "/data/"
        self.hr_path = dataset_dir + "/SRF_" + str(upscale_factor) + "/target/"
        self.upscale_factor = upscale_factor
        self.lr_filenames = [
            join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)
        ]
        self.hr_filenames = [
            join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)
        ]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split("/")[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize(
            (self.upscale_factor * h, self.upscale_factor * w),
            interpolation=Image.BICUBIC,
        )
        hr_restore_img = hr_scale(lr_image)
        return (
            image_name,
            ToTensor()(lr_image),
            ToTensor()(hr_restore_img),
            ToTensor()(hr_image),
        )

    def __len__(self):
        return len(self.lr_filenames)


def train(netG, netD, optimizerG, optimizerD, generator_loss, train_loader):
    netG.train()
    netD.train()

    running_results = {
        "batch_sizes": 0,
        "d_loss": 0,
        "g_loss": 0,
    }

    for data, target in train_loader:
        batch_size = data.size(0)
        running_results["batch_sizes"] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        if torch.cuda.is_available():
            target = target.cuda()
            data = data.cuda()
        fake_img = netG(data)

        netD.zero_grad()
        real_out = netD(target).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_loss(fake_out, fake_img, target)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(data)
        fake_out = netD(fake_img).mean()

        g_loss = generator_loss(fake_out, fake_img, target)
        running_results["g_loss"] += g_loss.item() * batch_size
        d_loss = 1 - real_out + fake_out
        running_results["d_loss"] += d_loss.item() * batch_size

        print(
            running_results
        )

    return running_results


def validate(netG, val_loader):
    netG.eval()
    val_bar = tqdm(val_loader, "validate")
    valing_results = {"mse": 0, "ssims": 0, "psnr": 0, "ssim": 0, "batch_sizes": 0}
    for lr, val_hr_restore, hr in val_bar:
        batch_size = lr.size(0)
        valing_results["batch_sizes"] += batch_size
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results["mse"] += batch_mse * batch_size
        batch_ssim = ssim(sr, hr).item()
        valing_results["ssims"] += batch_ssim * batch_size
        valing_results["psnr"] = 10 * log10(
            1 / (valing_results["mse"] / valing_results["batch_sizes"])
        )
        valing_results["ssim"] = valing_results["ssims"] / valing_results["batch_sizes"]

    return valing_results


def test(model_pth_fp, upscale_factor, image_fp, out_image_fp):
    with torch.no_grad():
        model = Generator(scale_factor=upscale_factor, in_channels=3).eval()
        state_dict = {}
        for k, v in torch.load(model_pth_fp).items():
            state_dict[k.replace("module.", "")] = v
        model.load_state_dict(state_dict)

        image = Image.open(image_fp)
        image_tensor = ToTensor()(image)
        image = image_tensor[:3, :, :].unsqueeze(0)
        out = model(image)
        out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(out_image_fp)


def plot_metrics(metrics_csv_path):
    df = pd.read_csv(metrics_csv_path)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(df.index.values, df["Loss_G"].values, "b-", label="Generator Loss")
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax1.set_xlabel("Epoch")

    ax2 = ax1.twinx()
    lns2 = ax2.plot(df.index.values, df["PSNR"].values, 'r-', label="PSNR")
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax3 = ax1.twinx()
    lns3 = ax3.plot(df.index.values, df["SSIM"].values, 'g-', label="SSIM")
    for tl in ax3.get_yticklabels():
        tl.set_color('g')

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc=0, framealpha=1)
    plt.show()


def main(
        train_data_dir,
        val_data_dir,
        checkpoint_dir,
        metrics_csv_fp,
        crop_size=88,
        upscale_factor=2,
        num_epochs=100,
):
    train_set = TrainDatasetFromFolder(
        train_data_dir,
        crop_size=crop_size,
        upscale_factor=upscale_factor,
    )
    train_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=128, shuffle=True
    )
    val_set = ValDatasetFromFolder(
        val_data_dir,
        upscale_factor=upscale_factor,
    )
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = nn.DataParallel(Generator(scale_factor=upscale_factor, in_channels=3))
    netD = nn.DataParallel(Discriminator(in_channels=3))
    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {
        "d_loss": [],
        "g_loss": [],
        "psnr": [],
        "ssim": [],
    }

    for epoch in tqdm(range(1, num_epochs + 1), "epoch"):
        running_results = train(
            netG, netD, optimizerG, optimizerD, generator_criterion, train_loader
        )
        valing_results = validate(netG, val_loader)
        print(valing_results)
        torch.save(
            netG.state_dict(), f"{checkpoint_dir}/netG_epoch_{upscale_factor}_{epoch}.pth"
        )
        torch.save(
            netD.state_dict(), f"{checkpoint_dir}/netD_epoch_{upscale_factor}_{epoch}.pth"
        )
        # save loss\psnr\ssim
        results["d_loss"].append(
            running_results["d_loss"] / running_results["batch_sizes"]
        )
        results["g_loss"].append(
            running_results["g_loss"] / running_results["batch_sizes"]
        )
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

        if epoch != 0:
            data_frame = pd.DataFrame(
                data={
                    "Loss_D": results["d_loss"],
                    "Loss_G": results["g_loss"],
                    "PSNR": results["psnr"],
                    "SSIM": results["ssim"],
                },
                index=range(1, epoch + 1),
            )
            data_frame.to_csv(
                metrics_csv_fp,
                index_label="Epoch",
            )


if __name__ == "__main__":
    main(
        upscale_factor=4,
        train_data_dir="/home/maksim/data/VOC2012/train",
        val_data_dir="/home/maksim/data/VOC2012/val",
        checkpoint_dir="/home/maksim/dev_projects/atlas_sr/checkpoints/srgan",
        metrics_csv_fp="/home/maksim/dev_projects/atlas_sr/checkpoints/srgan/metrics.csv",
    )
