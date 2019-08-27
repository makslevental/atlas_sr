import copy
import csv
import glob
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter

import numpy as np
import pandas as pd
import yaml
from matplotlib import patches
from pymongo import MongoClient
from tqdm import tqdm
from yaml import CLoader

from config.config import DSIAC_AGTS_DIR
from util.util import basename


class AGT:
    def __init__(self, agt_path, bbox_met_path, covar_json_path=None):
        self.basename = basename(agt_path)
        self.bbox_met_df = read_bbox_met(bbox_met_path)
        self.agt_dict = read_agt(agt_path)
        if covar_json_path is not None:
            self.covar_json = json.load(open(covar_json_path))

        self._aspects = None

    @property
    def aspects(self):
        if self._aspects is None:
            self._aspects = defaultdict(list)
            for tgt_update in self.agt_dict["TgtSect"]:
                if "Targets" in tgt_update:
                    for tgt_n, tgt in enumerate(tgt_update["Targets"]):
                        self._aspects[tgt_n].append(tgt["Aspect"])
        return self._aspects

    def targets(self, frame_n):
        return self.agt_dict["TgtSect"][frame_n]["Targets"]


def read_agt(fp):
    f = open(fp).read()
    r = re.sub(r"\s+{", ":", f).replace("}", "").replace("\t", "    ").replace('"', "")

    conts = ["SenUpd", "TgtUpd", "Tgt"]
    for cont in conts:
        i = 0
        while True:
            s = re.search(f"{cont}:", r)
            if not s:
                break
            s, e = s.span()
            r = r[:s] + f"{cont}.{i:05.0f}:" + r[e:]
            i += 1
    idents = [
        "Comment",
        "Aspect",
        "Azimuth",
        "Elevation",
        "Name",
        "Obscuration",
        "PlyId",
        "Pitch",
        "Range",
        "Roll",
        "TgtType",
        "Scenario",
        "Site",
        "Stake",
        "Time",
        "PixLoc",
        "Utm",
        "LatLong",
        "Fov",
        "PixBox",
        "PixRange",
    ]
    for ident in idents:
        r = re.sub(rf"^(\s*){ident}", rf"\1{ident}:", r, 0, re.MULTILINE)

    r = re.sub(r"Keyword\s*(\w*)\s*(-?\d*\.?\d*).*", r"\1: \2", r)

    y = yaml.load(r, Loader=CLoader)
    # ordering

    agt = copy.deepcopy(y["Agt"])
    agt["TgtSect"] = [
        tgt for _, tgt in sorted(list(agt["TgtSect"].items()), key=itemgetter(0))
    ]

    sensor_updates = []
    for sen in sorted(agt["SenSect"].keys()):
        if "SenUpd." in sen:
            sensor_updates.append(agt["SenSect"].pop(sen))
    agt["SenSect"] = sensor_updates

    for tgt_upt in agt["TgtSect"]:
        tgt_upt["Targets"] = []
        for tgt in sorted(tgt_upt.keys()):
            if "Tgt." in tgt:
                tgt_upt["Targets"].append(tgt_upt.pop(tgt))

    return agt


def read_bbox_met(bbox_met_path):
    with open(bbox_met_path) as csvfile:
        reader = np.array(list(csv.reader(csvfile)))[:, :11]
        df = pd.DataFrame(
            reader,
            columns=[
                "site",
                "unknown1",
                "unknown2",
                "sensor",
                "scenario",
                "frame",
                "ply_id",
                "unknown3",
                "unknown4",
                "upperx",
                "uppery",
            ],
        )

    df["frame"] = pd.to_numeric(df["frame"])
    df["ply_id"] = pd.to_numeric(df["ply_id"])
    df["upperx"] = pd.to_numeric(df["upperx"])
    df["uppery"] = pd.to_numeric(df["uppery"])

    return df


@dataclass
class BoundingBox:
    upper_left_x: int
    upper_left_y: int
    tgtx: int
    tgty: int

    @property
    def tgt_width(self) -> int:
        return 2 * (self.tgtx - self.upper_left_x)

    @property
    def tgt_height(self) -> int:
        return 2 * (self.tgty - self.upper_left_y)


class BoundingBoxes:
    def __init__(self, agt: AGT):
        self.agt = agt

    def mine(self, n):
        bboxes = []
        for i, tgt in enumerate(
                self.agt.agt_dict["Agt"]["TgtSect"][f"TgtUpd.{n}"]["Targets"]
        ):
            tgtx, tgty = map(int, tgt["PixLoc"].split())
            upper_left_x, upper_left_y = self.agt.bbox_met_df[
                self.agt.bbox_met_df["frame"] == n + 1
                ][["upperx", "uppery"]].iloc[i]
            bboxes.append(BoundingBox(upper_left_x, upper_left_y, tgtx, tgty))
        return bboxes

    def covar(self, n):
        if "BBoxes" in self.agt.covar_json["TgtSect"]["TgtUpd"][n]:
            len_bb = len(self.agt.covar_json["TgtSect"]["TgtUpd"][n]["BBoxes"])

            return [
                (
                    # fmt: off
                    self.agt.covar_json["TgtSect"]["TgtUpd"][n]["BBoxes"][tgt]["x0"],
                    self.agt.covar_json["TgtSect"]["TgtUpd"][n]["BBoxes"][tgt]["y0"],
                    self.agt.covar_json["TgtSect"]["TgtUpd"][n]["Tgt"][tgt]["PixLoc"][0],
                    self.agt.covar_json["TgtSect"]["TgtUpd"][n]["Tgt"][tgt]["PixLoc"][1],
                    self.agt.covar_json["TgtSect"]["TgtUpd"][n]["BBoxes"][tgt]["w"],
                    self.agt.covar_json["TgtSect"]["TgtUpd"][n]["BBoxes"][tgt]["h"],
                    # fmt: on
                )
                for tgt in range(len_bb)
            ]
        else:
            print(f"no bounding boxes {self.agt.basename} {n}")
            return []


def create_bbox_patches(bbox: BoundingBox, color="red"):
    rect = patches.Rectangle(
        xy=(bbox.upper_left_x, bbox.upper_left_y),
        width=bbox.tgt_width,
        height=bbox.tgt_height,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    circle = patches.Circle(xy=(bbox.tgtx, bbox.tgty), radius=2)

    return rect, circle


def read_all_agts():
    client = MongoClient("localhost", 27017)
    db = client["dsiac"]
    agts_table = db.agts
    for agt_file_fp in tqdm(glob.glob(f"{DSIAC_AGTS_DIR}/*.agt")):
        agts_table.insert_one(read_agt(agt_file_fp))


def query_all_agts():
    client = MongoClient("localhost", 27017)
    db = client["dsiac"]
    agts_table = db.agts
    df = pd.DataFrame(columns=["scenario", "tgt_type", "slant_range"])
    i = 0
    for agt in tqdm(agts_table.find(
            {
                "TgtSect": {
                    "$elemMatch": {
                        "Targets": {
                            "$elemMatch": {
                                # "TgtType": "SUV",
                                "SlantRange": {"$lte": 50000, "$gte": 1}
                            }
                        }
                    }
                }
            },
            {"PrjSect": 1, "TgtSect": 1},
    ), "agt"):
        for tgt_upd in tqdm(agt["TgtSect"], "tgt_upd"):
            for tgt in tgt_upd["Targets"]:
                df.loc[i] = [
                    agt["PrjSect"]["Name"],
                    tgt["TgtType"],
                    int(tgt["Range"]),
                ]
                i += 1
            break

    df.sort_values(["tgt_type", "slant_range"]).to_csv("tgts.csv", index=False)


def query_agts():
    df = pd.read_csv("tgts.csv")
    df.sort_values(["TgtType", "Range"]).to_csv("tgts.csv", index=False)


if __name__ == "__main__":
    # query_all_agts()
    query_agts()