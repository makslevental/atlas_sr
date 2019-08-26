from math import log10
from os import listdir
from os.path import join

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
        "d_score": 0,
        "g_score": 0,
    }

    for data, target in tqdm(train_loader, "train"):
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
        running_results["d_score"] += real_out.item() * batch_size
        running_results["g_score"] += fake_out.item() * batch_size

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

    netG = nn.DataParallel(Generator(upscale_factor))
    netD = nn.DataParallel(Discriminator())
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
        "d_score": [],
        "g_score": [],
        "psnr": [],
        "ssim": [],
    }

    for epoch in tqdm(range(1, num_epochs + 1), "epoch"):
        running_results = train(
            netG, netD, optimizerG, optimizerD, generator_criterion, train_loader
        )
        valing_results = validate(netG, val_loader)

        torch.save(
            netG.state_dict(), f"{checkpoint_dir}/netG_epoch_{upscale_factor}_{epoch}.pth"
        )
        torch.save(
            netD.state_dict(), f"{checkpoint_dir}/netD_epoch_{upscale_factor}_{epoch}.pth"
        )
        # save loss\scores\psnr\ssim
        results["d_loss"].append(
            running_results["d_loss"] / running_results["batch_sizes"]
        )
        results["g_loss"].append(
            running_results["g_loss"] / running_results["batch_sizes"]
        )
        results["d_score"].append(
            running_results["d_score"] / running_results["batch_sizes"]
        )
        results["g_score"].append(
            running_results["g_score"] / running_results["batch_sizes"]
        )
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

        if epoch != 0:
            data_frame = pd.DataFrame(
                data={
                    "Loss_D": results["d_loss"],
                    "Loss_G": results["g_loss"],
                    "Score_D": results["d_score"],
                    "Score_G": results["g_score"],
                    "PSNR": results["psnr"],
                    "SSIM": results["ssim"],
                },
                index=range(1, epoch + 1),
            )
            data_frame.to_csv(
                metrics_csv_fp,
                index_label="Epoch",
            )
