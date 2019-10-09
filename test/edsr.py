import argparse
import os

import numpy
import pandas
import torch
from dsiac.cegr.arf import make_arf
from matplotlib import pyplot
from scipy.stats import median_absolute_deviation as mad
from sewar import psnr
from sewar.command_line import metrics
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Lambda

from data_utils.cegr import ARFDataset, torch_mad_normalize
from models.edsr import EDSR
from util.util import load_model_state, show_im, linear_scale


def make_dataloader(arf_fp):
    transformed_dataset = ARFDataset(
        arf_fp,
        transform=transforms.Compose(
            [
                # Lambda(lambda x: torch_mad_normalize(x)),
                # Lambda(lambda x: torch.clamp(x, -20, 20)),
                Lambda(
                    lambda x: linear_scale(x, vmin=0, vmax=2 ** 16 - 1, rescale=255)
                ),
                Lambda(lambda x: torch.stack([x, x, x])),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        ),
    )
    _, h, w = transformed_dataset[0].size()
    dataloader = DataLoader(
        transformed_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    return dataloader, w, h


def test_grid():
    upscale_factor = 2
    n_resblocks = 32
    n_feats = 256
    with torch.cuda.device(0):
        with torch.no_grad():
            model = EDSR(upscale_factor, n_resblocks, n_feats, res_scale=0.1)
            load_model_state(
                model, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt"
            )
            model = model.to("cuda")
            for k in range(1, 10):
                f, axs = pyplot.subplots(
                    10, 10, frameon=False, figsize=(30, 30), constrained_layout=True
                )
                # f.suptitle(f'checker={j}', fontsize=16)
                axs = numpy.concatenate(axs)
                steps = 100
                for j, i in enumerate(range(0, steps)):
                    scale = i * (255 / steps)
                    ax = axs[j]
                    ax.set_axis_off()

                    grid = (i/10)*interpolate(
                        torch.Tensor([[1, 0], [0, 1]]).unsqueeze(0).unsqueeze(0),
                        scale_factor=(k, k),
                    ).numpy().squeeze(0).squeeze(0)
                    grid2 = torch.from_numpy(numpy.tile(grid, (10, 10)))
                    grid3 = torch.stack([grid2, grid2, grid2])
                    
                    # grid3 = scale * torch.ones((3, 100, 100))
                    out = model(grid3.unsqueeze(0).to("cuda"))
                    sr = out.squeeze(0).mean(dim=0).cpu().numpy()
                    p = psnr(numpy.tile(grid, (20, 20))*scale, sr)
                    print(j, p, numpy.median(sr))
                    ax.set_title(f"{scale:.2f} {p:.3f}")
                    # print(sr.min(), sr.max())
                    ax.imshow(sr)
                    # show_im(sr, title=f"{i/10}", height=.9, cmap="gray")
                f.tight_layout()
                f.subplots_adjust(top=0.88)
                f.savefig(f"{steps}_steps_checker_{k}_edsr.png")


def main():
    df = pandas.read_csv("tgts.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--frame_rate", type=int, default=1)
    parser.add_argument("--upscale_factor", type=int, default=2)
    parser.add_argument(
        "--model_fp",
        type=str,
        default="/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt",
    )
    parser.add_argument(
        "--arf_dirp", type=str, default="/home/maksim/data/DSIAC/cegr/arf"
    )
    parser.add_argument(
        "--new_arf_dirp", type=str, default="/media/maksim/3125372135FE0CCE/edsr"
    )
    args = parser.parse_args()

    local_rank = args.local_rank
    upscale_factor = args.upscale_factor
    frame_rate = args.frame_rate
    model_fp = os.path.expanduser(args.model_fp)
    arf_dirp = os.path.expanduser(args.arf_dirp)
    new_arf_dirp = os.path.expanduser(args.new_arf_dirp)
    n_resblocks = 32
    n_feats = 256

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(args.local_rank, world_size)
    with torch.cuda.device(local_rank):
        with torch.no_grad():
            model = EDSR(
                upscale_factor,
                n_resblocks,
                n_feats,
                res_scale=0.1,
                rgb_range=2 ** 16 - 1,
                # rgb_mean=(0, 0, 0)
            )
            load_model_state(model, model_fp)
            model = model.to("cuda")

            for scenario in sorted(df["scenario"][args.local_rank :: world_size]):
                arf_fp = f"{arf_dirp}/{scenario}.arf"
                new_arf_fp = f"{new_arf_dirp}/{scenario}_{upscale_factor}x.arf"
                if os.path.exists(new_arf_fp):
                    continue
                print(new_arf_fp)
                dataloader, w, h = make_dataloader(arf_fp)
                n_frames = len(dataloader) // frame_rate
                if n_frames == 0:
                    continue
                print(n_frames)
                new_arf = make_arf(
                    new_arf_fp,
                    height=upscale_factor * h,
                    width=upscale_factor * w,
                    n_frames=n_frames,
                )
                for i_batch, frame in enumerate(dataloader):
                    if i_batch % frame_rate:
                        continue
                    print(i_batch)
                    frame = frame.to("cuda")
                    sr = model(frame)
                    sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
                    new_arf[i_batch // frame_rate] = (
                        (sr * 2 ** 16 - 1).astype(numpy.uint16).byteswap()
                    )
                    new_arf.flush()
                    break
                break
    show_im(frame.squeeze().cpu().numpy()[0])
    show_im(sr)


def experiment_dataset():
    metrics.pop("rmse_sw")
    n_resblocks = 32
    n_feats = 256
    upscale_factor = 2
    with torch.cuda.device(0):
        with torch.no_grad():
            model = EDSR(
                upscale_factor,
                n_resblocks,
                n_feats,
                res_scale=0.1,
                rgb_range=2 ** 16 - 1,
                # rgb_mean=(0, 0, 0)
            )
            load_model_state(
                model, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt"
            )
            model = model.to("cuda")
            transformed_dataset = ARFDataset(
                "/home/maksim/data/DSIAC/cegr/arf/cegr01937_0005.arf",
                transform=transforms.Compose(
                    [
                        # Lambda(lambda x: torch_mad_normalize(x)),
                        # Lambda(lambda x: torch.clamp(x, -20, 20)),
                        # Lambda(lambda x: linear_scale(x, vmin=0, vmax=2 ** 16 - 1)),
                        Lambda(lambda x: torch.stack([x, x, x])),
                        # transforms.Normalize(
                        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        # ),
                    ]
                ),
            )
            hr = transformed_dataset.arf.get_frame_mat(0)
            frame = transformed_dataset[0].to("cuda").unsqueeze(0)
            bicubic = interpolate(
                frame,
                scale_factor=(1 / upscale_factor, 1 / upscale_factor),
                mode="bicubic",
                align_corners=True,
            )
            sr = model(bicubic)
            sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
            for metric_name, metric in metrics.items():
                print(f"full range meanshift {metric_name} {metric(hr, sr)}")
            show_im(sr, title=f"full range meanshift", height=0.95)

            model = EDSR(
                upscale_factor,
                n_resblocks,
                n_feats,
                res_scale=0.1,
                rgb_range=255,
                rgb_mean=(0, 0, 0),
            )
            load_model_state(
                model, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt"
            )
            model = model.to("cuda")
            transformed_dataset = ARFDataset(
                "/home/maksim/data/DSIAC/cegr/arf/cegr01937_0005.arf",
                transform=transforms.Compose(
                    [
                        # Lambda(lambda x: torch_mad_normalize(x)),
                        # Lambda(lambda x: torch.clamp(x, -20, 20)),
                        Lambda(
                            lambda x: linear_scale(
                                x, vmin=0, vmax=2 ** 16 - 1, rescale=255
                            )
                        ),
                        Lambda(lambda x: torch.stack([x, x, x])),
                        # transforms.Normalize(
                        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        # ),
                    ]
                ),
            )
            frame = transformed_dataset[0].to("cuda").unsqueeze(0)
            bicubic = interpolate(
                frame,
                scale_factor=(1 / upscale_factor, 1 / upscale_factor),
                mode="bicubic",
                align_corners=True,
            )
            sr = model(bicubic)
            sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
            sr = sr / 255 + 2 ** 16 - 1
            for metric_name, metric in metrics.items():
                print(f"255 no meanshift {metric_name} {metric(hr, sr)}")
            show_im(sr, title=f"255 range no meanshift", height=0.95)

            model = EDSR(
                upscale_factor,
                n_resblocks,
                n_feats,
                res_scale=0.1,
                rgb_range=255,
                # rgb_mean=(0, 0, 0)
            )
            load_model_state(
                model, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt"
            )
            model = model.to("cuda")
            transformed_dataset = ARFDataset(
                "/home/maksim/data/DSIAC/cegr/arf/cegr01937_0005.arf",
                transform=transforms.Compose(
                    [
                        # Lambda(lambda x: torch_mad_normalize(x)),
                        # Lambda(lambda x: torch.clamp(x, -20, 20)),
                        Lambda(
                            lambda x: linear_scale(
                                x, vmin=0, vmax=2 ** 16 - 1, rescale=255
                            )
                        ),
                        Lambda(lambda x: torch.stack([x, x, x])),
                        # transforms.Normalize(
                        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        # ),
                    ]
                ),
            )
            frame = transformed_dataset[0].to("cuda").unsqueeze(0)
            bicubic = interpolate(
                frame,
                scale_factor=(1 / upscale_factor, 1 / upscale_factor),
                mode="bicubic",
                align_corners=True,
            )
            sr = model(bicubic)
            sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
            sr = sr / 255 + 2 ** 16 - 1
            for metric_name, metric in metrics.items():
                print(f"255 meanshift {metric_name} {metric(hr, sr)}")
            show_im(sr, title=f"255 range meanshift", height=0.95)

            im = transformed_dataset.arf.get_frame_mat(0)
            im = linear_scale(im, vmin=0, vmax=2 ** 16 - 1, rescale=255)
            m = im.mean()
            s = im.std()
            model = EDSR(
                upscale_factor,
                n_resblocks,
                n_feats,
                res_scale=0.1,
                rgb_range=255,
                rgb_mean=(m, m, m),
                rgb_std=(s, s, s),
            )
            load_model_state(
                model, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt"
            )
            model = model.to("cuda")
            transformed_dataset = ARFDataset(
                "/home/maksim/data/DSIAC/cegr/arf/cegr01937_0005.arf",
                transform=transforms.Compose(
                    [
                        # Lambda(lambda x: torch_mad_normalize(x)),
                        # Lambda(lambda x: torch.clamp(x, -20, 20)),
                        Lambda(
                            lambda x: linear_scale(
                                x, vmin=0, vmax=2 ** 16 - 1, rescale=255
                            )
                        ),
                        Lambda(lambda x: torch.stack([x, x, x])),
                        # transforms.Normalize(
                        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        # ),
                    ]
                ),
            )
            frame = transformed_dataset[0].to("cuda").unsqueeze(0)
            bicubic = interpolate(
                frame,
                scale_factor=(1 / upscale_factor, 1 / upscale_factor),
                mode="bicubic",
                align_corners=True,
            )
            sr = model(bicubic)
            sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
            sr = sr / 255 + 2 ** 16 - 1
            for metric_name, metric in metrics.items():
                print(f"255 std mean shift {metric_name} {metric(hr, sr)}")
            show_im(sr, title=f"255 std mean", height=0.95)

            model = EDSR(
                upscale_factor,
                n_resblocks,
                n_feats,
                res_scale=0.1,
                rgb_range=255,
                rgb_mean=(0, 0, 0),
            )
            load_model_state(
                model, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt"
            )
            model = model.to("cuda")
            transformed_dataset = ARFDataset(
                "/home/maksim/data/DSIAC/cegr/arf/cegr01937_0005.arf",
                transform=transforms.Compose(
                    [
                        Lambda(lambda x: torch_mad_normalize(x)),
                        Lambda(lambda x: torch.clamp(x, -20, 20)),
                        Lambda(lambda x: linear_scale(x, rescale=255)),
                        Lambda(lambda x: torch.stack([x, x, x])),
                        # transforms.Normalize(
                        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        # ),
                    ]
                ),
            )
            frame = transformed_dataset[0].to("cuda").unsqueeze(0)
            bicubic = interpolate(
                frame,
                scale_factor=(1 / upscale_factor, 1 / upscale_factor),
                mode="bicubic",
                align_corners=True,
            )
            med = numpy.median(hr)
            x_mad = mad(hr.flatten())
            hr_mad = numpy.clip((hr - med) / x_mad, -20, 20)
            vmin, vmax = hr_mad.min(), hr_mad.max()
            sr = model(bicubic)
            sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
            for metric_name, metric in metrics.items():
                print(
                    f"mad norm 255 range {metric_name} {metric(hr, ((sr / 255) * (vmax - vmin) + vmin) * x_mad + med)}"
                )
            show_im(sr, title=f"mad norm 255 range", height=0.95)


if __name__ == "__main__":
    # main()
    test_grid()
