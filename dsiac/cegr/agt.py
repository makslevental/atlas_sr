import copy
import csv
import json
import re
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd
import yaml
from yaml import CLoader

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
    r = re.sub(r"\s+{", ":", f).replace("}", "").replace("\t", "    ")

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
        "Keyword",
    ]
    for ident in idents:
        r = r.replace(ident, f"{ident}:")
    y = yaml.load(r, Loader=CLoader)
    # ordering

    agt = copy.deepcopy(y["Agt"])
    agt["TgtSect"] = [
        tgt for _, tgt in sorted(list(agt["TgtSect"].items()), key=itemgetter(0))
    ]

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
