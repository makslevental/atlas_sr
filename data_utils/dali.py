import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


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


class StupidDALIIterator:
    # https://github.com/NVIDIA/DALI/issues/1227
    def __init__(self, *args, **kwargs):
        self.dali_iter = DALIGenericIterator(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        n = next(self.dali_iter)
        lr_image, hr_image = n[0]["lr_image"], n[0]["hr_image"]
        hr_image = hr_image.to(torch.float).div(255)
        lr_image = lr_image.to(torch.float).div(255)
        hr_image = hr_image.permute(0, 3, 1, 2)
        lr_image = lr_image.permute(0, 3, 1, 2)
        return lr_image, hr_image

    @property
    def size(self):
        return self.dali_iter._size

    def reset(self):
        self.dali_iter.reset()


class SRGANPipeline(Pipeline):
    def __init__(
            self, batch_size, num_threads, device_id, crop, upscale_factor, image_type, dali_cpu=False
    ):
        super(SRGANPipeline, self).__init__(
            batch_size, num_threads, device_id
        )
        crop = calculate_valid_crop_size(crop, upscale_factor)
        decoder_device = "cpu" if dali_cpu else "mixed"
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.ImageDecoderCrop(
            device=decoder_device,
            output_type=image_type,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            crop=crop,
        )
        dali_device = "cpu" if dali_cpu else "gpu"
        self.res = ops.Resize(
            device=dali_device,
            resize_x=crop // upscale_factor,
            resize_y=crop // upscale_factor,
            interp_type=types.DALIInterpType.INTERP_CUBIC,
            image_type=image_type,
        )
        self.uniform_rng = ops.Uniform(range=(0.0, 1.0))
        self.cast = ops.Cast(device=dali_device, dtype=types.DALIDataType.FLOAT)

    def define_graph(self):
        jpegs, _labels = self.input(name="Reader")
        hr_images = self.decode(
            jpegs, crop_pos_x=self.uniform_rng(), crop_pos_y=self.uniform_rng()
        )
        lr_images = self.res(hr_images)

        return [lr_images, hr_images]


class SRGANMXNetPipeline(SRGANPipeline):
    def __init__(
            self,
            *,
            batch_size,
            num_gpus,
            num_threads,
            device_id,
            crop,
            upscale_factor,
            image_type,
            mx_path,
            mx_index_path,
            dali_cpu=False,
            random_shuffle=True
    ):
        super(SRGANMXNetPipeline, self).__init__(
            batch_size, num_threads, device_id, crop, upscale_factor, image_type, dali_cpu
        )
        self.input = ops.MXNetReader(
            path=[mx_path],
            index_path=[mx_index_path],
            random_shuffle=random_shuffle,
            shard_id=device_id,
            num_shards=num_gpus,
        )


class SRGANFilePipeline(SRGANPipeline):
    def __init__(
            self,
            *,
            batch_size,
            num_gpus,
            num_threads,
            device_id,
            crop,
            upscale_factor,
            image_type,
            data_dir,
            file_list,
            dali_cpu=False,
            random_shuffle=True
    ):
        super(SRGANFilePipeline, self).__init__(
            batch_size, num_threads, device_id, crop, upscale_factor, image_type, dali_cpu
        )
        self.input = ops.FileReader(
            file_root=data_dir,
            file_list=file_list,
            shard_id=device_id,
            num_shards=num_gpus,
            random_shuffle=random_shuffle,
        )


class SRGANVOCPipeline(Pipeline):
    def __init__(
            self,
            *,
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
        super(SRGANVOCPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        self.input = ops.MXNetReader(
            path=[mx_path],
            index_path=[mx_index_path],
            random_shuffle=True,
            shard_id=device_id,
            num_shards=num_gpus,
        )
        self.u_rng = ops.Uniform(range=(0.0, 1.0))
        crop = calculate_valid_crop_size(crop, upscale_factor)
        decoder_device = "cpu" if dali_cpu else "mixed"
        self.decode = ops.ImageDecoderCrop(device=decoder_device, crop=crop)
        dali_device = "cpu" if dali_cpu else "gpu"
        self.res = ops.Resize(
            device=dali_device,
            resize_x=crop // upscale_factor,
            resize_y=crop // upscale_factor,
            interp_type=types.INTERP_CUBIC,
        )
        self.cmp = ops.CropMirrorNormalize(
            device=dali_device, output_layout=types.NHWC, output_dtype=types.FLOAT
        )
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        c = self.coin()
        jpegs, _labels = self.input(name="Reader")
        hr_images = self.decode(jpegs, crop_pos_x=self.u_rng(), crop_pos_y=self.u_rng())
        lr_images = self.res(hr_images)

        return [lr_images, hr_images]
