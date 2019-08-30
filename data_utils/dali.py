import nvidia.dali.ops as ops
import nvidia.dali.types as types
from matplotlib import gridspec, pyplot as plt
from nvidia.dali.pipeline import Pipeline


class ResNetImageNetPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        num_gpus,
        num_threads,
        device_id,
        crop,
        mx_path,
        mx_index_path,
        dali_cpu=False,
    ):
        super(ResNetImageNetPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        self.input = ops.MXNetReader(
            path=[mx_path],
            index_path=[mx_index_path],
            random_shuffle=True,
            shard_id=device_id,
            num_shards=num_gpus,
        )
        # let user decide which pipeline works him bets for RN version he runs
        dali_device = "cpu" if dali_cpu else "gpu"
        decoder_device = "cpu" if dali_cpu else "mixed"
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.ImageDecoderRandomCrop(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100,
        )
        self.res = ops.Resize(
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_TRIANGULAR,
        )
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class SRGANPipeline(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, crop, upscale_factor, dali_cpu=False
    ):
        super(SRGANPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        self.pos_rng_x = ops.Uniform(range=(0.0, 1.0))
        self.pos_rng_y = ops.Uniform(range=(0.0, 1.0))
        crop = calculate_valid_crop_size(crop, upscale_factor)
        decoder_device = "cpu" if dali_cpu else "mixed"
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.ImageDecoderCrop(
            device=decoder_device,
            output_type=types.GRAY,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            crop=crop,
        )

        dali_device = "cpu" if dali_cpu else "gpu"
        self.res = ops.Resize(
            device=dali_device,
            resize_x=crop // upscale_factor,
            resize_y=crop // upscale_factor,
            interp_type=types.INTERP_CUBIC,
            image_type=types.GRAY,
        )

    def define_graph(self):
        jpegs, _labels = self.input()
        pos_x = self.pos_rng_x()
        pos_y = self.pos_rng_y()
        hr_images = self.decode(jpegs, crop_pos_x=pos_x, crop_pos_y=pos_y)
        lr_images = self.res(hr_images)

        return [lr_images, hr_images]


class SRGANMXNetPipeline(SRGANPipeline):
    def __init__(
        self,
        batch_size,
        num_gpus,
        num_threads,
        device_id,
        crop,
        upscale_factor,
        mx_path,
        mx_index_path,
        dali_cpu=False,
    ):
        super(SRGANMXNetPipeline, self).__init__(
            batch_size, num_threads, device_id, crop, upscale_factor, dali_cpu
        )
        self.input = ops.MXNetReader(
            path=[mx_path],
            index_path=[mx_index_path],
            random_shuffle=True,
            shard_id=device_id,
            num_shards=num_gpus,
        )


class SRGANFilePipeline(SRGANPipeline):
    def __init__(
        self,
        batch_size,
        num_gpus,
        num_threads,
        device_id,
        crop,
        upscale_factor,
        data_dir,
        file_list,
        dali_cpu=False,
    ):
        super(SRGANFilePipeline, self).__init__(
            batch_size, num_threads, device_id, crop, upscale_factor, dali_cpu
        )
        self.input = ops.FileReader(
            file_root=data_dir,
            file_list=file_list,
            shard_id=device_id,
            num_shards=num_gpus,
            random_shuffle=True,
        )


def show_images(image_batch, batch_size):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j).squeeze(2), cmap="gray")
    plt.show()


def test_pipeline():
    batch_size = 16
    pipe = SRGANFilePipeline(
        batch_size,
        1,
        4,
        0,
        32,
        2,
        "/home/maksim/dev_projects/atlas_sr/data/tiny-imagenet-200/val/images",
        "/home/maksim/dev_projects/atlas_sr/data/tiny-imagenet-200/val/val_annotations.1.txt",
    )
    pipe.build()
    lr_images, hr_images = pipe.run()
    show_images(lr_images.as_cpu(), batch_size)
    show_images(hr_images.as_cpu(), batch_size)
