from train.train_srgan import main, test, plot_metrics
from train.train_resnet_dali import main

if __name__ == "__main__":
    # main(
    #     upscale_factor=4,
    #     train_data_dir="/home/maksim/dev_projects/SRGAN/data/VOC2012/train",
    #     val_data_dir="/home/maksim/dev_projects/SRGAN/data/VOC2012/val",
    #     checkpoint_dir="/home/maksim/dev_projects/atlas_sr/checkpoints/srgan",
    #     metrics_csv_fp="/home/maksim/dev_projects/atlas_sr/checkpoints/srgan/metrics.csv",
    # )
    # test(
    #     "/home/maksim/dev_projects/atlas_sr/checkpoints/srgan/netG_epoch_4_60.pth",
    #     4,
    #     "/home/maksim/dev_projects/SRGAN/dsiac_lr_images/cegr02002_0001/cegr02002_0001_120.tiff",
    #     "/home/maksim/dev_projects/SRGAN/dsiac_lr_images/cegr02002_0001/cegr02002_0001_120.4x.tiff",
    # )
    # test(
    #     "/home/maksim/dev_projects/SRGAN/epochs/netG_epoch_2_20.pth",
    #     2,
    #     "/home/maksim/dev_projects/SRGAN/dsiac_lr_images/cegr01923_0013/cegr01923_0013_120.tiff",
    #     "/home/maksim/dev_projects/SRGAN/dsiac_lr_images/cegr01923_0013/cegr01923_0013_120.2x.tiff",
    # )
    plot_metrics("/home/maksim/dev_projects/atlas_sr/checkpoints/srgan/metrics.csv")