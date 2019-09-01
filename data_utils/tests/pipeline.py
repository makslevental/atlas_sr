# this pipeline is for 1-1 testing against a pytorch loader
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import gridspec
from nvidia.dali import ops, types
from nvidia.dali.pipeline import Pipeline

from data_utils.dali import (
    calculate_valid_crop_size,
    SRGANMXNetPipeline,
    StupidDALIIterator,
)


class TestingSRGANPipeline(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, crop, upscale_factor, dali_cpu=False
    ):
        super(TestingSRGANPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        crop = calculate_valid_crop_size(crop, upscale_factor)
        decoder_device = "cpu" if dali_cpu else "mixed"
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.ImageDecoderCrop(
            device=decoder_device,
            output_type=types.DALIImageType.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            crop_pos_x=0,
            crop_pos_y=0,
            crop=crop,
        )
        dali_device = "cpu" if dali_cpu else "gpu"
        self.cpm = ops.CropMirrorNormalize(
            device=dali_device,
            crop_pos_x=0,
            crop_pos_y=0,
            crop=crop,
            mean=[0, 0, 0],
            std=[255, 255, 255],
            output_layout=types.NHWC,
            mirror=0,
            output_dtype=types.DALIDataType.FLOAT,
        )
        self.res = ops.Resize(
            device=dali_device,
            resize_x=crop // upscale_factor,
            resize_y=crop // upscale_factor,
            interp_type=types.DALIInterpType.INTERP_CUBIC,
            # image_type=types.GRAY,
        )
        self.u_rng = ops.Uniform(device=dali_device, range=[0, 1])
        # self.cast = ops.Cast(device=dali_device, dtype=types.DALIDataType.UINT8)

    def define_graph(self):
        jpegs, _labels = self.input(name="Reader")
        hr_images = self.decode(jpegs)
        lr_images = self.res(hr_images)
        hr_images = self.cpm(hr_images)

        return [lr_images, hr_images]


def show_images(image_batch, batch_size):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        img = image_batch.at(j)
        print(img.dtype)
        img = np.transpose(img, (1, 2, 0))
        # img = img.squeeze(2)
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show()


def test_mxnet_pipeline():
    batch_size = 16
    pipe = SRGANMXNetPipeline(
        batch_size=batch_size,
        num_gpus=1,
        num_threads=4,
        device_id=0,
        crop=88,
        upscale_factor=2,
        mx_path="/home/maksim/data/VOC2012/voc_train.rec",
        mx_index_path="/home/maksim/data/VOC2012/voc_train.idx",
    )
    pipe.build()
    lr_images, hr_images = pipe.run()
    show_images(hr_images.as_cpu(), batch_size)
    show_images(lr_images.as_cpu(), batch_size)


def image_reses():
    image_rezes = []
    for img_fp in glob.glob(
        "/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/val/*.JPEG"
    ):
        h, w = Image.open(img_fp).size
        if h < 224 or w < 224:
            print(h, w)
            os.remove(img_fp)
        else:
            image_rezes.append(img_fp)
    print(len(image_rezes))


def test_iter():
    train_pipe = SRGANMXNetPipeline(
        batch_size=16,
        num_gpus=1,
        num_threads=1,
        device_id=0,
        crop=88,
        dali_cpu=False,
        mx_path="/home/maksim/data/VOC2012/voc_val.rec",
        mx_index_path="/home/maksim/data/VOC2012/voc_val.idx",
        upscale_factor=2,
    )
    train_pipe.build()
    train_loader = StupidDALIIterator(
        pipelines=[train_pipe],
        output_map=["lr_image", "hr_image"],
        size=int(train_pipe.epoch_size("Reader") / 1),
    )

    for lr, hr in train_loader:
        print(lr.shape, hr.shape)


if __name__ == "__main__":
    test_mxnet_pipeline()
    # image_reses()
    # test_iter()
