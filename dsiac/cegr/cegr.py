from typing import Dict

from dsiac.bounding_boxes.bounding_boxes import BoundingBoxes
from dsiac.cegr.agt import AGT
from dsiac.cegr.arf import ARF


class CEGR:
    def __init__(self, arf_path, agt_path, bbox_met_path, covar_json_path=None):
        self.arf_path = arf_path
        self.agt_path = agt_path
        self.bbox_met_path = bbox_met_path
        self.covar_json_path = covar_json_path

        self._project_sect = None
        self._arf = None
        self._agt = None
        self._bboxes: BoundingBoxes = None

    @property
    def agt(self) -> AGT:
        if self._agt is None:
            self._agt = AGT(self.agt_path, self.bbox_met_path, self.covar_json_path)
            self._bboxes = BoundingBoxes(self._agt)
        return self._agt

    @property
    def arf(self) -> ARF:
        if self._arf is None:
            self._arf = ARF(self.arf_path)
        return self._arf

    @property
    def project_sect(self) -> Dict:
        if self._project_sect is None:
            self._project_sect = self.agt.agt_dict["PrjSect"]
        return self._project_sect

    def bboxes(self, n):
        return self._bboxes.mine(n), self._bboxes.covar(n)

    def show_frame(self, n, title=None):
        self.arf.show_frame(n, title)

    def save_frame(self, n, fp, title=None):
        self.arf.save_frame(n, fp, title)
