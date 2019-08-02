from dataclasses import dataclass

from matplotlib import patches as patches

from dsiac.cegr.agt import AGT
from dsiac.util import basename


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
            print(f"no bounding boxes {basename} {n}")
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
