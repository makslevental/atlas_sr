from dsiac.cegr import cegr

c = cegr.CEGR(
    arf_path="/home/maksim/data/yuma/arf/avco10425_1034.arf",
    agt_path="/home/maksim/data/yuma/agt/avco10425_1034.agt",
    bbox_met_path="/home/maksim/data/yuma/metric/avco10425_1034.bbox_met"
)

c.bboxes(100)

for i in range(1800):
    print(i)
    c.save_frame(i, f"/home/maksim/data/yuma/mad_tiffs/avco10425_1034/{i}.tiff")