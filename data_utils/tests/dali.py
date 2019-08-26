import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from data.dali import FileReadPipeline, MXNetReaderPipeline


def test_dali():
    image_dir = "/home/maksim/Downloads/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/train"

    # TFRecord

    N = 3  # number of GPUs
    BATCH_SIZE = 128  # batch size per GPU
    ITERATIONS = 32

    pipe_name, label_range = MXNetReaderPipeline, (0, 1)
    print("RUN: " + pipe_name.__name__)
    pipes = [
        pipe_name(
            batch_size=BATCH_SIZE, num_threads=4, device_id=device_id, num_gpus=3
        )
        for device_id in range(N)
    ]
    pipes[0].build()
    dali_iter = DALIGenericIterator(
        pipes, ["data", "label"], pipes[0].epoch_size("Reader")
    )

    for i, data in enumerate(dali_iter):
        if i >= ITERATIONS:
            break
        # Testing correctness of labels
        for d in data:
            # label = d["label"]
            image = d["data"]
            print(image)
            ## labels need to be integers
            # assert np.equal(np.mod(label, 1), 0).all()
            # ## labels need to be in range pipe_name[2]
            # assert (label >= label_range[0]).all()
            # assert (label <= label_range[1]).all()
    print("OK : " + pipe_name.__name__)


if __name__ == "__main__":
    test_dali()