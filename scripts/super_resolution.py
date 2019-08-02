import numpy as np
import glob
import os
from PIL import Image

from ISR.models import RDN

rdn = RDN(arch_params={"C": 6, "D": 20, "G": 64, "G0": 64, "x": 2})
rdn.model.load_weights(
    "/home/maksim/dev_projects/dsiac/image-super-resolution/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5"
)


def super_resolve(fp):
    img = Image.open(fp)
    lr_img = np.array(img)
    sr_img = rdn.predict(lr_img)
    im = Image.fromarray(sr_img)
    im.save(fp + "super.jpg")


if __name__ == "__main__":
    for im_fp in glob.glob("../multi_range/**/*.jpg"):
        super_resolve(im_fp)
