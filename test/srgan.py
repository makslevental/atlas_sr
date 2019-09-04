import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from models.SRGAN import Generator


def test(model_pth_fp, upscale_factor, image_fp, out_image_fp, channels=3):
    model = Generator(scale_factor=upscale_factor, in_channels=channels).eval()
    state_dict = {}
    for k, v in torch.load(model_pth_fp).items():
        state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)

    image = Image.open(image_fp)
    image_tensor = ToTensor()(image)
    if channels == 1:
        image = image_tensor[1, :, :].unsqueeze(0).unsqueeze(0)
    else:
        image = image_tensor[:3, :, :].unsqueeze(0)

    with torch.no_grad():
        out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(out_image_fp)


if __name__ == "__main__":
    test(
        "/home/maksim/data/checkpoints/srgan_dali/netG_epoch_2_33.pth",
        upscale_factor=2,
        image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.tiff",
        out_image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.2x.imagenet.no_icnr.tiff",
        channels=1
    )
    test(
        "/home/maksim/data/checkpoints/srgan_dali_dp/netG_epoch_2_7.pth",
        upscale_factor=2,
        image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.tiff",
        out_image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.2x.imagenet.icnr.tiff",
    )
    test(
        "/home/maksim/dev_projects/atlas_sr/checkpoints/srgan_dali/netG_epoch_2_62.pth",
        upscale_factor=2,
        image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.tiff",
        out_image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.2x.pascal.no_icnr.tiff",
    )
    test(
        "/home/maksim/dev_projects/atlas_sr/checkpoints/srgan_dali_pascal_3_channel_icnr_dp/netG_epoch_2_59.pth",
        upscale_factor=2,
        image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.tiff",
        out_image_fp="/home/maksim/dev_projects/atlas_sr/test/cegr01923_0011.2x.pascal.icnr.tiff",
    )

