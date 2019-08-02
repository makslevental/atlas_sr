import glob
import json
import os
from typing import List

import pandas as pd

from dsiac.cegr.cegr import CEGR
from dsiac.util import make_dsiac_paths, map, FilePaths, find_nearest, join_and_mkdir


def compare_at_multiple_ranges():
    df = print_sorted_by_type_range()

    TARGET_N = 0
    FRAME_N = 100

    for group_name, group_df in df.groupby("target_type"):
        paths: List[FilePaths] = map(make_dsiac_paths, group_df["basename"].values)

        cegr = CEGR(*paths[0])
        target = cegr.agt.targets(FRAME_N)[TARGET_N]
        aspect = target["Aspect"]
        print(f"aspect: {aspect}")

        fp = os.path.join(
            join_and_mkdir(
                "/home/maksim/dev_projects/dsiac/multi_range", target["TgtType"]
            ),
            f"{target['Range']:.0f}m.jpg",
        )

        title = f"{target['TgtType']} @ {target['Range']}"
        cegr.save_frame(FRAME_N, fp, title=title)

        for path in paths[1:]:
            try:
                cegr = CEGR(*path)
                frame_n = find_nearest(cegr.agt.aspects[TARGET_N], aspect)
                target = cegr.agt.targets(FRAME_N)[TARGET_N]
                title = f"{target['TgtType']} @ {target['Range']}"
                fp = os.path.join(
                    join_and_mkdir(
                        "/home/maksim/dev_projects/dsiac/multi_range", target["TgtType"]
                    ),
                    f"{target['Range']:.0f}m.jpg",
                )
                cegr.save_frame(frame_n, fp, title=title)
            except Exception as e:
                print(e)


def print_sorted_by_type_range():
    covar_jsons = [
        json.load(open(fp))
        for fp in glob.glob(
            "/home/maksim/dev_projects/dsiac/DSIAC/annotated-jsons/*.json"
        )
    ]
    stats = [
        (
            covar_json["PrjSect"]["TargetName"],
            covar_json["PrjSect"]["Range"],
            covar_json["PrjSect"]["Name"],
        )
        for covar_json in covar_jsons
    ]
    df = pd.DataFrame(
        {
            "target_type": [s[0] for s in stats],
            "range": [s[1] for s in stats],
            "basename": [s[2] for s in stats],
        }
    ).sort_values(by=["target_type", "range", "basename"])

    return df


def agt_aspects(basename):
    *_, covar_json_path = make_dsiac_paths(basename)
    covar_json = json.load(open(covar_json_path))
    return [
        tgt_update["Tgt"][0]["Aspect"]
        for tgt_update in covar_json["TgtSect"]["TgtUpd"]
        if "Tgt" in tgt_update
    ]
