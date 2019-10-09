import argparse
import os

import numpy
import pandas
import torch
from dsiac.cegr.arf import make_arf
from matplotlib import pyplot
from sewar.command_line import metrics
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from data_utils.cegr import ARFDataset
from models.dbpn import DBPNITER
from util.util import load_model_state, linear_unscale


# [[114.35629928 111.561547   103.1545782 ]] [[4020.22681688 3690.08934962 4068.48610146]]


def main():
    df = pandas.read_csv("tgts.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--frame_rate", type=int, default=1)
    parser.add_argument("--upscale_factor", type=int, default=2)
    parser.add_argument(
        "--model_fp",
        type=str,
        default="/home/maksim/data/checkpoints/dbpn_checkpoints/DBPN-RES-MR64-3_2x.pth",
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

    rescale = 1.04 - 0.79
    bias = 0.79

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(args.local_rank, world_size)
    with torch.cuda.device(local_rank):
        with torch.no_grad():
            model = DBPNITER(
                num_channels=3,
                base_filter=64,
                feat=256,
                num_stages=3,
                scale_factor=upscale_factor,
            )
            load_model_state(model, model_fp)
            model = model.to("cuda")
            for scenario in sorted(df["scenario"][args.local_rank :: world_size]):
                arf_fp = f"{arf_dirp}/{scenario}.arf"
                new_arf_fp = f"{new_arf_dirp}/{scenario}_{upscale_factor}x.arf"
                if os.path.exists(new_arf_fp):
                    continue
                print(new_arf_fp)

                transformed_dataset = ARFDataset(arf_fp, rescale=rescale, bias=bias)
                frame, _, _ = transformed_dataset[0]
                _, h, w = frame.size()
                dataloader = DataLoader(
                    transformed_dataset, batch_size=1, shuffle=False, num_workers=1
                )

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
                for i_batch, (frame, vmin, vmax) in enumerate(dataloader):
                    if i_batch % frame_rate:
                        continue
                    print(i_batch)
                    frame = frame.to("cuda")
                    bicubic = interpolate(
                        frame,
                        scale_factor=(upscale_factor, upscale_factor),
                        mode="bicubic",
                        align_corners=True,
                    )
                    out = model(frame)
                    sr = out + bicubic
                    sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
                    sr = linear_unscale(sr, bias, rescale, vmin.item(), vmax.item())
                    new_arf[i_batch // frame_rate] = sr.astype(numpy.uint16).byteswap()
                    new_arf.flush()


def experiment_dataset():
    metrics.pop("rmse_sw")
    upscale_factor = 2
    transformed_dataset = ARFDataset(
        # "/home/maksim/data/DSIAC/cegr/arf/cegr01937_0005.arf",
        "/home/maksim/data/DSIAC/cegr/arf/cegr01939_0001.arf",
        rescale=1.04 - 0.79,
        bias=0.79,
    )
    with torch.cuda.device(0):
        with torch.no_grad():
            model = DBPNITER(
                num_channels=3,
                base_filter=64,
                feat=256,
                num_stages=3,
                scale_factor=upscale_factor,
            )
            load_model_state(
                model,
                "/home/maksim/data/checkpoints/dbpn_checkpoints/DBPN-RES-MR64-3_2x.pth",
            )
            model = model.to("cuda")

            rescale = 1.04 - 0.79
            bias = 0.79

            frame, vmin, vmax = transformed_dataset[0]
            frame = frame.unsqueeze(0)

            bicubic_up_up = interpolate(
                frame,
                scale_factor=(upscale_factor, upscale_factor),
                mode="bicubic",
                align_corners=True,
            )

            out = model(frame.to("cuda"))
            sr = (
                out.squeeze(0).mean(dim=0).cpu().numpy()
                + bicubic_up_up.squeeze(0).mean(dim=0).cpu().numpy()
            )
            # show_im(out.squeeze(0).mean(dim=0).cpu().numpy())
            # show_im(sr)
            sr = linear_unscale(sr, bias, rescale, vmin, vmax)
            # show_im(sr)


def test_grid():
    upscale_factor = 2
    with torch.cuda.device(0):
        with torch.no_grad():
            model = DBPNITER(
                num_channels=3,
                base_filter=64,
                feat=256,
                num_stages=3,
                scale_factor=upscale_factor,
            )
            load_model_state(
                model,
                "/home/maksim/data/checkpoints/dbpn_checkpoints/DBPN-RES-MR64-3_2x.pth",
            )
            min_std = numpy.float("inf")
            model = model.to("cuda")
            for k in range(1, 10):
                f, axs = pyplot.subplots(
                    10, 10, frameon=False, figsize=(30, 30), constrained_layout=True
                )
                f.suptitle(f"checker={k}", fontsize=16)
                stds = []
                axs = numpy.concatenate(axs)
                for j, i in enumerate(range(1, 100)):
                    scale = i / 100
                    ax = axs[j]
                    ax.set_title(f"{scale}")
                    ax.set_axis_off()

                    grid = (i / 10) * interpolate(
                        torch.Tensor([[1, 0], [0, 1]]).unsqueeze(0).unsqueeze(0),
                        scale_factor=(k, k),
                    ).numpy().squeeze(0).squeeze(0)
                    grid2 = torch.from_numpy(numpy.tile(grid, (10, 10)))
                    grid3 = torch.stack(3 * [grid2])
                    grid4 = interpolate(grid3.unsqueeze(0), scale_factor=(2, 2))

                    out = model(grid3.unsqueeze(0).to("cuda"))
                    sr = out.squeeze(0).mean(dim=0).cpu().numpy()
                    ax.imshow(sr + grid4.squeeze(0).mean(dim=0).cpu().numpy())
                print(min_std)
                f.tight_layout()
                f.subplots_adjust(top=0.88)
                f.savefig(f"100_checker_{k}_step_dbpn.png")


if __name__ == "__main__":
    # experiment_dataset()
    # test_grid()
    main()
