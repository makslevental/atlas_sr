from abc import ABC

from nvidia.dali.pipeline import Pipeline


class CommonPipeline(Pipeline, ABC):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(
            device="gpu", image_type=types.RGB, interp_type=types.INTERP_LINEAR
        )
        self.cmn = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            crop=(227, 227),
            image_type=types.RGB,
            mean=[128.0, 128.0, 128.0],
            std=[1.0, 1.0, 1.0],
        )
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.resize_rng = ops.Uniform(range=(256, 480))

    def base_define_graph(self, inputs, labels):
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter=self.resize_rng())
        output = self.cmn(images, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        return output, labels


class HybridTrainPipe(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False
    ):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        self.input = ops.FileReader(
            file_root=data_dir,
            # shard_id=args.local_rank,
            # num_shards=args.world_size,
            random_shuffle=True,
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


class FileReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, file_root):
        super(FileReadPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root=file_root)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)


import nvidia.dali.ops as ops
import nvidia.dali.types as types


class MXNetReaderPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(MXNetReaderPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.MXNetReader(
            path=[
                "/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_rec.rec"
            ],
            index_path=[
                "/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_rec.idx"
            ],
            random_shuffle=True,
            shard_id=device_id,
            num_shards=num_gpus,
        )

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)
