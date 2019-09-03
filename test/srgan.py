import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from models.SRGAN import Generator


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


if __name__ == "__main__":
    test(
        "/home/maksim/dev_projects/atlas_sr/checkpoints/srgan_dali/netG_epoch_2_62.pth",
        upscale_factor=2,
        image_fp="/tmp/cegr01923_0011.tiff",
        out_image_fp="/tmp/cegr01923_0011.2x.pascal.tiff",
    )
