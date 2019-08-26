from train.train_srgan import main

if __name__ == "__main__":
    main(
        train_data_dir="/home/maksim/dev_projects/SRGAN/data/VOC2012/train",
        val_data_dir="/home/maksim/dev_projects/SRGAN/data/VOC2012/val",
        checkpoint_dir="/home/maksim/dev_projects/atlas_sr/checkpoints/srgan",
        metrics_csv_fp="/home/maksim/dev_projects/atlas_sr/checkpoints/srgan/metrics.csv",
    )
