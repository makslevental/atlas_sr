import argparse
import os

import numpy
import pandas
import torch
from dsiac.cegr.arf import make_arf
from torch.utils.data import DataLoader

from data_utils.cegr import ARFDataset
from models.srgan import Generator
from util.util import load_model_state, linear_unscale, show_im


def main():
    df = pandas.read_csv("tgts.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--frame_rate", type=int, default=1)
    parser.add_argument("--upscale_factor", type=int, default=2)
    parser.add_argument(
        "--model_fp",
        type=str,
        default="/home/maksim/data/checkpoints/dbpn_checkpoints/srresnet_imagenet_2x_psnr_0076.pth",
    )
    parser.add_argument(
        "--arf_dirp", type=str, default="/home/maksim/data/DSIAC/cegr/arf"
    )
    parser.add_argument(
        "--new_arf_dirp", type=str, default="/media/maksim/3125372135FE0CCE/srresnet"
    )
    args = parser.parse_args()

    local_rank = args.local_rank
    upscale_factor = args.upscale_factor
    frame_rate = args.frame_rate
    model_fp = os.path.expanduser(args.model_fp)
    arf_dirp = os.path.expanduser(args.arf_dirp)
    new_arf_dirp = os.path.expanduser(args.new_arf_dirp)
    rescale = 1
    bias = 0

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(args.local_rank, world_size)
    with torch.cuda.device(local_rank):
        with torch.no_grad():
            model = Generator(scale_factor=upscale_factor)
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
                new_arf = make_arf(
                    new_arf_fp,
                    height=upscale_factor * h,
                    width=upscale_factor * w,
                    n_frames=n_frames,
                )
                for i_batch, (frame, vmin, vmax) in enumerate(dataloader):
                    if i_batch % frame_rate:
                        continue
                    if not i_batch % 20:
                        print(scenario, i_batch)
                    print(i_batch)
                    frame = frame.to("cuda")
                    sr = model(frame)
                    sr = sr.squeeze(0).mean(dim=0).cpu().numpy()
                    sr = linear_unscale(sr, bias, rescale, vmin.item(), vmax.item())
                    new_arf[i_batch // frame_rate] = sr.astype(numpy.uint16).byteswap()
                    new_arf.flush()


if __name__ == "__main__":
    main()
